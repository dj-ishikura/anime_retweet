import json
import os
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
import pandas as pd

# 先行研究の注釈者のアノテーション精度と, ファインチューニングしたモデルの性能を比較

def calculate_metrics(predictions, gold_labels):
    accuracy = accuracy_score(gold_labels, predictions)
    mae = mean_absolute_error(gold_labels, predictions)
    qwk = cohen_kappa_score(gold_labels, predictions, weights='quadratic')
    result = {
        "accuracy": accuracy,
        "mae": mae,
        "qwk": qwk
    }
    print(result)

file = "/work/n213304/learn/anime_retweet_2/work_emo_analyze/wrime/wrime-ver2.tsv"

df = pd.read_csv(file, delimiter="\t")
df = df[df["Train/Dev/Test"] == "test"]

Reader_A = df["Reader1_Sentiment"].replace([-2, 2], [-1, 1]).tolist()
Reader_B = df["Reader2_Sentiment"].replace([-2, 2], [-1, 1]).tolist()
Reader_C = df["Reader3_Sentiment"].replace([-2, 2], [-1, 1]).tolist()
Reader_Avg = df["Avg. Readers_Sentiment"].replace([-2, 2], [-1, 1]).tolist()

# メトリクスを計算
print("Reader_A - Reader_Avg")
calculate_metrics(Reader_A, Reader_Avg)

print("Reader_B - Reader_Avg")
calculate_metrics(Reader_B, Reader_Avg)

print("Reader_C - Reader_Avg")
calculate_metrics(Reader_C, Reader_Avg)



