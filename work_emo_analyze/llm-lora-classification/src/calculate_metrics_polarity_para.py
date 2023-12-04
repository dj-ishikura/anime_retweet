import json
import os
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score

def calculate_metrics(predictions, gold_labels):
    accuracy = accuracy_score(gold_labels, predictions)
    mae = mean_absolute_error(gold_labels, predictions)
    qwk = cohen_kappa_score(gold_labels, predictions, weights='quadratic')
    return accuracy, mae, qwk

# ファイル名のパターンに基づいてファイルを探す
output_dir = "/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/outputs/rinna__japanese-gpt-neox-3.6b/2023-12-01/4"

# 結果を格納するためのリスト
results = []

# ディレクトリ内の各ファイルに対して処理を実行
for file_name in os.listdir(output_dir):
    if file_name.startswith("test-predictions_") and file_name.endswith(".json"):
        file_path = os.path.join(output_dir, file_name)

        # ファイル名からバッチ数、エポック数、学習率を抽出
        parts = file_name[len("test-predictions_"):-len(".json")].split("_")
        batch_size, epochs, learning_rate = parts[0], parts[1], parts[2]

        # ファイルを開いてデータを読み込む
        with open(file_path, "r") as file:
            data = json.load(file)
            predictions = [item["predictions"] for item in data]
            gold_labels = [item["gold_labels"] for item in data]

            # メトリクスを計算
            accuracy, mae, qwk = calculate_metrics(predictions, gold_labels)

            # 結果をリストに追加
            results.append({
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "accuracy": accuracy,
                "mae": mae,
                "qwk": qwk
            })

# CSVファイルに結果を書き込む
import csv

# TSVファイルに結果を書き込む
output_tsv_path = os.path.join(output_dir, "result-test-predictions.tsv")

results.sort(key=lambda x: x['mae'])

with open(output_tsv_path, 'w', newline='', encoding='utf-8') as tsvfile:
    fieldnames = ['batch_size', 'epochs', 'learning_rate', 'accuracy', 'mae', 'qwk']
    writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')

    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"結果が {output_tsv_path} に出力されました。")