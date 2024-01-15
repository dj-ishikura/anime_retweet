from pathlib import Path
from tap import Tap
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import pipeline
from transformers.pipelines.base import KeyDataset
from datasets import load_dataset

import src.utils as utils

class Args(Tap):
    model_name: str = "rinna/japanese-gpt-neox-3.6b"
    dataset_dir: Path = "./datasets/wrime_to01"
    output_file: Path = "./zeroshot_prompt.json"
    device: int = 0 if torch.cuda.is_available() else -1
    batch_size: int = 16
    ignore_premise: bool = False

    def process_args(self):
        self.labels: List[str] = ["negative", "neutral", "positive"]
        self.test_file: Path = self.dataset_dir / "test.tsv"

def main(args):
    dataset = load_dataset('csv', data_files={'test': str(args.test_file)}, delimiter='\t', split='test')
    def dsconv(x):
        x['text'] = """question: Classify the sentiment of the following tweet into three categories, such as negative, neutral, or positive. 
tweet: 最高にろっくなアニメやった ありがとうBTR 速攻でロスになってますわ  #ぼっち・ざ・ろっく
sentiment classification: positive

question: Classify the sentiment of the following tweet into three categories, such as negative, neutral, or positive. 
tweet: 【新作予約情報】吸血鬼すぐ死ぬ  アクリルチャーム付き缶バッジ  ¥1000+税  ※12月10日より店頭予約受付開始 ※1月下旬頃入荷予定   #秋田書店ストア #吸血鬼すぐ死ぬ https://t.co/xHZnT8jpgi
sentiment classification: neutral

question: Classify the sentiment of the following tweet into three categories, such as negative, neutral, or positive.
tweet: """ + x['Sentence'] + "\n sentiment classification: "
        return x
    dataset = dataset.map(dsconv)

    pipe = pipeline(model=args.model_name,
                    task='text-generation',
                    device=args.device)

    gold_labels = []
    pred_labels = []
    results = []
    label_dict = {-1: "negative", 0: "neutral", 1: "positive"}
    for e,r in zip(dataset,
                   pipe(KeyDataset(dataset, 'text'),
                        batch_size=args.batch_size)):

        print(f"r[0]['generated_text'] : {r[0]['generated_text']}")

        pred = r[0]['generated_text']
        pred_labels.append(pred)
        gold_labels.append(e['Avg. Readers_Sentiment'])
        results.append({'Sentence': e['Sentence'],
                        'gold_label': label_dict[e['Avg. Readers_Sentiment']],
                        'predicted_label': pred})

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