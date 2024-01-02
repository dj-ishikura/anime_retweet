import os
import pandas as pd
import json

# ディレクトリとファイルパス
input_dir = "anime_retweet_concat"
remove_file_path = "anime_data_loss_data.csv"

# 削除するアニメのIDリストを取得
df_remove_anime = pd.read_csv(remove_file_path)
remove_anime_ids = df_remove_anime['id'].tolist()

# 集計用の変数
total_retweets = 0
unique_original_tweet_ids = set()

# 各ファイルを処理
for file_name in os.listdir(input_dir):
    id = os.path.splitext(file_name)[0]  # ファイル名から拡張子を除外
    if id in remove_anime_ids:
        continue
    print(id)
    file_path = os.path.join(input_dir, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split(',', 2)
            if len(parts) < 3:
                continue

            tweet_id, hashtag, tweet_json_str = parts
            try:
                total_retweets += 1
                tweet_json = json.loads(tweet_json_str)
                if "retweeted_status" in tweet_json:
                    original_tweet_id = tweet_json["retweeted_status"]["id_str"]
                    unique_original_tweet_ids.add(original_tweet_id)
            except json.JSONDecodeError as e:
                print(f"JSON デコードエラー: {e}, 行: {line}")
                continue

# 結果の出力
total_original_tweets = len(unique_original_tweet_ids)
print(f"合計リツイートデータ数: {total_retweets}")
print(f"ユニークなオリジナルツイート数: {total_original_tweets}")

# 平均値の計算
average_retweets = total_retweets / len(os.listdir(input_dir)) if os.listdir(input_dir) else 0
average_original_tweets = total_original_tweets / len(os.listdir(input_dir)) if os.listdir(input_dir) else 0
print(f"平均リツイートデータ数: {average_retweets}")
print(f"平均オリジナルツイート数: {average_original_tweets}")
