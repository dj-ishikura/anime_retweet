from pathlib import Path
from tap import Tap
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import pipeline
from transformers.pipelines.base import KeyDataset
import pandas as pd
from datasets import load_dataset

import src.utils as utils

import torch
import numpy as np
import random

class Args(Tap):
    model_name: str = "rinna/japanese-gpt-neox-3.6b"
    dataset_dir: Path = "./datasets/wrime"
    output_file: Path = "./zeroshot.json"
    device: int = 0 if torch.cuda.is_available() else -1
    batch_size: int = 1

    def process_args(self):
        self.labels: List[str] = ["強いネガティブ", "ネガティブ", "ニュートラル", "ポジティブ", "強いポジティブ"]
        self.test_file: Path = self.dataset_dir / "test.tsv"

def main(args):
    dataset = load_dataset('csv', data_files={'test': str(args.test_file)}, delimiter='\t', split='test')
    def dsconv(x):
        x['text'] = x['Sentence']
        return x
    # dataset = dataset.map(dsconv)
    print(f'dataset: \n {dataset}')
    print(f'args.labels: \n {args.labels}')

    classifier = pipeline("zero-shot-classification", model=args.model_name, device=args.device)

    gold_labels = []
    pred_labels = []
    results = []
    for example, result in zip(dataset, classifier(KeyDataset(dataset, 'Sentence'), batch_size=args.batch_size, candidate_labels=args.labels)):
        print(f'result: \n {result}')
        print(f'example: \n {example}')

        p = result['labels'][0]
        pred_labels.append(p)
        gold_labels.append(args.labels[example['Avg. Readers_Sentiment']+2])
        results.append({'gold_label': args.labels[example['Avg. Readers_Sentiment']+2],
                        'predicted_label': p})

    from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score

    # 正解率、平均絶対誤差、QWKの計算
    label_to_index = {label: index for index, label in enumerate(args.labels)}
    gold_labels_numeric = [label_to_index[label] for label in gold_labels]
    pred_labels_numeric = [label_to_index[label] for label in pred_labels]

    accuracy: float =  accuracy_score(gold_labels_numeric, pred_labels_numeric)
    mae: float = mean_absolute_error(gold_labels_numeric, pred_labels_numeric)
    qwk: float = cohen_kappa_score(gold_labels_numeric, pred_labels_numeric, weights='quadratic')

    # 統計情報を辞書に格納
    stat = {
        "accuracy": accuracy,
        "mae": mae,
        "qwk": qwk,
        "results": results
    }

    if args.output_file:
        utils.save_json(stat, args.output_file)
    else:
        print(stat)

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)