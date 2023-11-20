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
import sklearn.metrics as metrics

import src.utils as utils
from src.models import Model
import pandas as pd
import numpy as np


class Args(Tap):
    model_name: str = "rinna/japanese-gpt-neox-3.6b"
    dataset_dir: Path = "./datasets/wrime"

    batch_size: int = 32
    epochs: int = 1
    num_warmup_epochs: int = 1

    template_type: int = 2

    lr: float = 2e-5
    lora_r: int = 32
    weight_decay: float = 0.01
    max_seq_len: int = 512
    gradient_checkpointing: bool = True

    seed: int = 42

    def process_args(self):
        # 各感情カテゴリーのラベルの範囲を設定 (0, 1, 2, 3)
        self.emotion_labels = list(range(4))  # 8つの感情カテゴリーそれぞれについて

        # 感情極性のラベルの範囲を設定 (-2, -1, 0, 1, 2)
        self.polarity_labels = list(range(-2, 3))

        # ラベルの設定
        self.labels = {
            'emotion': self.emotion_labels,  # 8つの感情カテゴリーのラベル
            'polarity': self.polarity_labels  # 感情極性のラベル
        }

        # 出力ディレクトリの設定
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
            num_labels=len(args.labels),
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

    def load_dataset(self, split: str, shuffle: bool = False) -> DataLoader:
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
            max_length=self.args.max_seq_len,
        )

        # ラベルをTensorに変換
        labels = torch.tensor([[
            d['Avg. Readers_Joy'],
            d['Avg. Readers_Sadness'],
            d['Avg. Readers_Anticipation'],
            d['Avg. Readers_Surprise'],
            d['Avg. Readers_Anger'],
            d['Avg. Readers_Fear'],
            d['Avg. Readers_Disgust'],
            d['Avg. Readers_Trust'],
            d['Avg. Readers_Sentiment']
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
        # 最初のバリデーション
        val_metrics = {"epoch": None, **self.evaluate(self.val_dataloader)}
        best_epoch, best_val_mae = None, val_metrics["mae"]
        best_state_dict = self.model.clone_state_dict()
        self.log(val_metrics)
        for param in self.model.parameters():
            param.requires_grad = True

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
                print(f'loss : {loss}')
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()

            # エポックごとのバリデーション
            self.model.eval()
            val_metrics = {"epoch": epoch, **self.evaluate(self.val_dataloader)}
            self.log(val_metrics)

            # 最良モデルの更新
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_epoch = epoch
                best_state_dict = self.model.clone_state_dict()

        # 最良モデルのロードとテスト
        self.model.load_state_dict(best_state_dict)
        self.model.eval()
        val_metrics = {"best-epoch": best_epoch, **self.evaluate(self.val_dataloader)}
        test_metrics = self.evaluate(self.test_dataloader)
        return val_metrics, test_metrics

    def get_predictions(self, dataloader: DataLoader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.accelerator.device) for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = logits.detach().cpu().numpy()
                predictions.extend(preds)
        return predictions

    def display_predictions(self, dataloader: DataLoader):
        for batch in dataloader:
            inputs = {k: v.to(self.accelerator.device) for k, v in batch.items() if k != 'labels'}
            logits_emotion, logits_polarity = self.model(**inputs)
            preds_emotion = logits_emotion.detach().cpu().numpy()
            preds_polarity = logits_polarity.detach().cpu().numpy()

            # 予測結果を適切な形式に変換
            for pred_emotion, pred_polarity in zip(preds_emotion, preds_polarity):
                predicted_label_emotion = np.argmax(pred_emotion, axis=-1)
                predicted_label_polarity = np.argmax(pred_polarity, axis=-1) - 2  # -2 から 2 の範囲に調整
                print(f"Emotion: {predicted_label_emotion}, Polarity: {predicted_label_polarity}")


    @torch.inference_mode()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        num_labels = 9  # ラベルの総数
        self.model.eval()
        total_loss = 0

        # MAEを計算するための変数を初期化
        total_mae = [0.0 for _ in range(num_labels)]  # num_labels はラベルの総数
        num_samples = 0

        total_correct = 0  # Accuracy用
        all_preds = []  # QWK用
        all_trues = []  # QWK用

        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            out: SequenceClassifierOutput = self.model(**batch)

            batch_size: int = batch['input_ids'].size(0)
            num_samples += batch_size

            for i in range(num_labels):
                true = batch['labels'][:, i]
                # print(f"batch['labels'][:, i] : \n {batch['labels'][:, i]}")
                # print(f"out.logits : \n {out.logits}")
                pred = out.logits[i]
                
                total_mae[i] += torch.sum(torch.abs(pred - true)).item()

                # 正解率の計算のために予測と実際のラベルを保存
                all_preds.append(pred.cpu())
                all_trues.append(true.cpu())

                # 正解率の計算
                total_correct += (pred.round() == true).sum().item()

        # 各ラベルのMAEを平均化
        avg_mae = sum(total_mae) / (num_samples * num_labels)

        # Accuracyを計算
        accuracy = total_correct / (num_samples * num_labels)

        # QWKを計算
        qwk = metrics.cohen_kappa_score(
            np.concatenate(all_trues), 
            np.concatenate(all_preds).round(), 
            weights="quadratic"
        )
        print(f'loss : {total_loss / num_samples}')
        return {
            "loss": total_loss / num_samples,
            "mae": avg_mae,
            "accuracy": accuracy,
            "qwk": qwk
        }
    
    def log(self, metrics: dict) -> None:
        utils.log(metrics, self.args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']} \t"
            f"loss: {metrics['loss']:2.6f}   \t"
            f"accuracy: {metrics['accuracy']:.4f} \t"
            f"mae: {metrics['mae']:.4f} \t"
        )

def main(args: Args):
    exp = Experiment(args=args)
    val_metrics, test_metrics = exp.run()
    exp.display_predictions(exp.val_dataloader)

    utils.save_json(val_metrics, args.output_dir / "dev-metrics.json")
    utils.save_json(test_metrics, args.output_dir / "test-metrics.json")
    utils.save_config(args, args.output_dir / "config.json")

if __name__ == "__main__":
    args = Args().parse_args()
    utils.init(seed=args.seed)
    main(args)
