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
from src.models import Model

import pandas as pd
import numpy as np

import os

class Args(Tap):
    model_name: str = "rinna/japanese-gpt-neox-3.6b"
    dataset_dir: Path = "./datasets/wrime"

    batch_size: int = 32
    # batch_size: int = 64 # 動作テスト用
    epochs: int = 3
    num_warmup_epochs: int = 1
    # num_warmup_epochs: int = 0 # 動作テスト用

    template_type: int = 2

    lr: float = 2e-5
    lora_r: int = 32
    weight_decay: float = 0.01
    max_seq_len: int = 256
    # max_seq_len: int = 128  # 動作テスト用
    gradient_checkpointing: bool = True 
    # gradient_checkpointing: bool = False # 動作テスト用

    seed: int = 42

    def process_args(self):
        self.emotion_labels = list(range(4))
        self.polarity_labels = list(range(-2, 3))
        # self.labels: list[int] = self.polarity_labels

        date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
        self.output_dir = self.make_output_dir(
            "outputs",
            self.model_name,
            date,
            time,
        )

    def make_output_dir(self, *args) -> Path:
        args = [str(a).replace("/", "__") for a in args]
        output_dir = Path(*args)
        output_dir.mkdir(parents=True)
        return output_dir


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
            lora_r=args.lora_r,
            gradient_checkpointing=args.gradient_checkpointing,
        ).eval()
        self.model.write_trainable_params()

        self.train_dataloader = self.load_dataset(split="train", shuffle=True)
        steps_per_epoch: int = len(self.train_dataloader)

        self.accelerator = Accelerator()
        (
            self.model,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader,
            self.load_dataset(split="dev", shuffle=False),
            self.load_dataset(split="test", shuffle=False),
            *self.create_optimizer(steps_per_epoch),
        )

    def load_dataset(
        self,
        split: str,
        shuffle: bool = False,
    ) -> DataLoader:
        path: Path = self.args.dataset_dir / f"{split}.tsv"
        dataset: pd.DataFrame = pd.read_csv(path, sep='\t')
        return self.create_loader(dataset.to_dict(orient="records"), shuffle=shuffle)

    def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        texts = [d["Sentence"] for d in data_list]

        inputs: BatchEncoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=args.max_seq_len,
        )

        labels = torch.tensor([[
            d['Avg. Readers_Joy'],
            d['Avg. Readers_Sadness'],
            d['Avg. Readers_Anticipation'],
            d['Avg. Readers_Surprise'],
            d['Avg. Readers_Anger'],
            d['Avg. Readers_Fear'],
            d['Avg. Readers_Disgust'],
            d['Avg. Readers_Trust'],
            d['Avg. Readers_Sentiment']+2
        ] for d in data_list], dtype=torch.float)
        

        return BatchEncoding({**inputs, "labels": labels})

    def create_loader(
        self,
        dataset,
        batch_size=None,
        shuffle=False,
    ):
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size or self.args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
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
        val_metrics = {"epoch": None, **self.evaluate(self.val_dataloader)}
        best_epoch, best_val_mae = None, val_metrics["mae"]
        best_state_dict = self.model.clone_state_dict()
        self.log(val_metrics)

        for epoch in trange(self.args.epochs, dynamic_ncols=True):
            self.model.train()

            for batch in tqdm(
                self.train_dataloader,
                total=len(self.train_dataloader),
                dynamic_ncols=True,
                leave=False,
            ):
                self.optimizer.zero_grad()
                out: SequenceClassifierOutput = self.model(**batch)
                loss: torch.FloatTensor = out.loss
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()

            self.model.eval()
            val_metrics = {"epoch": epoch, **self.evaluate(self.val_dataloader)}
            self.log(val_metrics)

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_epoch = epoch
                best_state_dict = self.model.clone_state_dict()

        self.model.load_state_dict(best_state_dict)
        self.model.eval()

        val_metrics = {"best-epoch": best_epoch, **self.evaluate(self.val_dataloader)}
        test_metrics = self.evaluate(self.test_dataloader)

        return val_metrics, test_metrics

    def predict_test_data(self):
        self.model.eval()
        predictions = []
        all_batch_texts, all_pred_labels, all_gold_labels = [], [], []

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, total=len(self.test_dataloader), dynamic_ncols=True):
                out: SequenceClassifierOutput = self.model(**batch)

                # テキストのデコード
                batch_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]
                all_batch_texts.extend(batch_texts)

                # 各感情と感情極性に対する予測ラベルの取得
                batch_pred_labels = [logit.argmax(dim=-1).tolist() for logit in out.logits]
                batch_pred_labels = [label for sublist in batch_pred_labels for label in sublist]
                all_pred_labels.extend(batch_pred_labels)

                # 実際のラベル
                batch_gold_labels = [label for sublist in batch.labels.tolist() for label in sublist]
                all_gold_labels.extend(batch_gold_labels)

            all_pred_labels_ = [all_pred_labels[i:i + 9] for i in range(0, len(all_pred_labels), 9)]
            all_gold_labels_ = [all_gold_labels[i:i + 9] for i in range(0, len(all_gold_labels), 9)]

            # 各感情カテゴリーに対して、予測、実際のラベル、テキストを関連付ける
            for text, pred, gold in zip(all_batch_texts, all_pred_labels_, all_gold_labels_):
                predictions.append({
                    "text": text,
                    "predictions": pred,
                    "gold_labels": gold
                })

        return predictions

    @torch.inference_mode()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss, all_pred_labels, all_gold_labels = 0, [], []

        # MAEを計算するための変数を初期化
        # total_mae = [0.0 for _ in range(num_labels)]  # num_labels はラベルの総数
        num_samples = 0

        total_correct = 0  # Accuracy用
        all_preds = []  # QWK用
        all_trues = []  # QWK用

        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            out: SequenceClassifierOutput = self.model(**batch)

            batch_size: int = batch.input_ids.size(0)
            num_samples += batch_size

            total_loss += out.loss.item() * batch_size

            # 各感情と感情極性に対する予測ラベルの取得
            batch_pred_labels = [logit.argmax(dim=-1).tolist() for logit in out.logits]
            batch_pred_labels = [label for sublist in batch_pred_labels for label in sublist]
            all_pred_labels.extend(batch_pred_labels)

            # 実際のラベル
            batch_gold_labels = [label for sublist in batch.labels.tolist() for label in sublist]
            all_gold_labels.extend(batch_gold_labels)

        avg_loss = total_loss / num_samples

        accuracy = accuracy_score(all_gold_labels, all_pred_labels)

        # MAEの計算
        mae = mean_absolute_error(all_gold_labels, all_pred_labels)

        # QWKの計算
        qwk = cohen_kappa_score(all_gold_labels, all_pred_labels, weights='quadratic')

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "mae": mae,
            "qwk": qwk,
        }

    def log(self, metrics: dict) -> None:
        utils.log(metrics, self.args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']} \t"
            f"loss: {metrics['loss']:.4f} \t"
            f"accuracy: {metrics['accuracy']:.4f} \t"
            f"mae: {metrics['mae']:.4f} \t"
            f"qwk: {metrics['qwk']:.4f} \t"
        )


def main(args: Args):
    exp = Experiment(args=args)
    val_metrics, test_metrics = exp.run()

    # テストデータの予測、テキスト、正解ラベルを取得
    test_predictions = exp.predict_test_data()

    # 結果をファイルに保存
    utils.save_json(test_predictions, args.output_dir / "test-predictions.json")

    utils.save_json(val_metrics, args.output_dir / "val-metrics.json")
    utils.save_json(test_metrics, args.output_dir / "test-metrics.json")
    utils.save_config(args, args.output_dir / "config.json")

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = Args().parse_args()
    utils.init(seed=args.seed)
    main(args)
