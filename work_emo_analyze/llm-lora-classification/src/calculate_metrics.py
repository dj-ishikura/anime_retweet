import json
from sklearn.metrics import accuracy_score, mean_absolute_error

def calculate_metrics(predictions, gold_labels):
    category_metrics = {}

    # 感情カテゴリの名前
    categories = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"]

    # 各感情強度カテゴリに対する正解率とMAEの計算
    sum_accuracy = 0.0
    sum_mae = 0.0
    for i, category in enumerate(categories):
        preds = [pred[i] for pred in predictions]
        golds = [gold[i] for gold in gold_labels]
        accuracy = accuracy_score(golds, preds)
        mae = mean_absolute_error(golds, preds)
        sum_accuracy += accuracy
        sum_mae += mae
        category_metrics[category] = {
            "accuracy": accuracy,
            "mae": mae
        }

    # 感情強度全体に対する正解率とMAEの計算
    category_metrics["感情強度全体"] = {
        "accuracy": sum_accuracy / 8,
        "mae": sum_mae / 8
    }

    # 感情極性に対する正解率とMAEの計算
    if len(predictions[0]) > 8:
        preds = [pred[8] for pred in predictions]
        golds = [gold[8] for gold in gold_labels]
        accuracy = accuracy_score(golds, preds)
        mae = mean_absolute_error(golds, preds)
        sum_accuracy += accuracy
        sum_mae += mae
        category_metrics["感情極性"] = {
            "accuracy": accuracy,
            "mae": mae
        }

        # 感情強度全体に対する正解率とMAEの計算
        category_metrics["全体"] = {
            "accuracy": sum_accuracy / 9,
            "mae": sum_mae / 9
        }

    return category_metrics

# JSONファイルからデータを読み込む
path = "/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/outputs/rinna__japanese-gpt-neox-3.6b/2023-11-23/03-44-39.473696/test-predictions.json"
path = "/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/outputs/rinna__japanese-gpt-neox-3.6b/2023-11-23/05-05-45.506225/test-predictions.json"
with open(path, "r") as file:
    data = json.load(file)

# 予測と正解ラベルの抽出
predictions = [item["predictions"] for item in data]
gold_labels = [item["gold_labels"] for item in data]

# メトリクスの計算
metrics = calculate_metrics(predictions, gold_labels)

# 結果の出力
for category, values in metrics.items():
    print(f"{category}: Accuracy = {values['accuracy']:.3f}, MAE = {values['mae']:.3f}")
