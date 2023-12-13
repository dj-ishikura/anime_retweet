from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BatchEncoding, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

import src.utils as utils
from src.models_prediction import Model

import pandas as pd
import numpy as np

import os

class Args(Tap):
    model_name: str = "rinna/japanese-gpt-neox-3.6b"
    input_path: Path = ""
    output_path: Path = ""

    batch_size: int = 32 # 32, 64, 128 # メモリオーバー　gliotq or オプション gradient_accumurate 
    epochs: int = 2 # 2, 3, 4
    num_warmup_epochs: int = 1

    template_type: int = 2

    lr: float = 5e-5 # 1e-5, 2e-5, 5e-5
    weight_decay: float = 0.01
    max_seq_len: int = 256
    gradient_checkpointing: bool = True

    seed: int = 42

    labels: list[int] = list(range(-1, 2))

class Experiment:
    def __init__(self, args: Args):
        self.args: Args = args

        use_fast = not ("japanese-gpt-neox" in args.model_name)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=args.max_seq_len,
            use_fast=use_fast,
        )

        self.model: PreTrainedModel = Model(
            model_name=args.model_name,
            num_labels=len(args.labels),
            gradient_checkpointing=args.gradient_checkpointing,
        ).eval()
        self.model.write_trainable_params()

        self.test_dataloader = self.load_dataset()
        steps_per_epoch: int = len(self.test_dataloader)

        self.accelerator = Accelerator()
        (
            self.model,
            self.test_dataloader,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.test_dataloader,
            *self.create_optimizer(steps_per_epoch),
        )

    def create_optimizer(
        self,
        steps_per_epoch: int,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [
                    param for name, param in self.model.named_parameters() if not name in no_decay
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    param for name, param in self.model.named_parameters() if name in no_decay
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=steps_per_epoch * self.args.num_warmup_epochs,
            num_training_steps=steps_per_epoch * self.args.epochs,
        )

        return optimizer, lr_scheduler

    def run(self):
        model_backbone = "/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/model_emo_polarity_fine_to01.backbone"
        model_classifier = "/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/model_emo_polarity_fine_to01.classifier"
        saved_state={"backbone": torch.load(model_backbone),"classifier": torch.load(model_classifier)}
        print("model loading ...")
        self.model.load_state_dict(saved_state)
        print("model eval ...")
        self.model.eval()
        print("model prediction ...")
        test_predictions = self.evaluate(self.test_dataloader)

        utils.save_json(test_predictions, args.output_path)

    def load_dataset(
        self,
    ) -> DataLoader:
        dataset: pd.DataFrame = pd.read_json(args.input_path, lines=True)
        # dataset: pd.DataFrame = pd.read_csv(args.input_path, sep='\t')
        print(dataset)
        return self.create_loader(dataset.to_dict(orient="records"))

    def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        texts = [d["text"] for d in data_list]
        tweet_id = [d["tweet_id"] for d in data_list]

        inputs: BatchEncoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=args.max_seq_len,
        )
        return BatchEncoding({**inputs, "tweet_id": tweet_id})

    def create_loader(
        self,
        dataset,
        batch_size=None,
        shuffle=False,
    ):
        print("create_loader ...")
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size or self.args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    @torch.inference_mode()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        predictions = []
        all_batch_texts, all_batch_tweet_ids, pred_labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
                all_batch_tweet_ids += batch.pop("tweet_id")  # tweet_id を取得し、バッチから削除
                out: SequenceClassifierOutput = self.model(**batch)
                # テキストのデコード
                batch_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]
                all_batch_texts.extend(batch_texts)

                pred_labels += out.logits.argmax(dim=-1).tolist()
            
            print("prediction ...")
            # 各感情カテゴリーに対して、予測、実際のラベル、テキストを関連付ける
            for text, tweet_id, pred in zip(all_batch_texts, all_batch_tweet_ids, pred_labels):
                predictions.append({
                    "tweet_id": tweet_id,
                    "text": text,
                    "predictions": pred
                })

        return predictions

def main(args: Args):
    exp = Experiment(args=args)
    exp.run()
    
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = Args().parse_args()
    utils.init(seed=args.seed)
    main(args)