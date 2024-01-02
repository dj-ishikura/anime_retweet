import os
import json

input_dir = 'tweet_data_randam'
output_file_path = 'tweet_data_randam_text.jsonl'

tweet_data = {}  # ツイート ID とテキストを保存するための辞書

# ディレクトリ内のすべてのファイルを処理
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split('\t')  # タブで分割
            tweet_id, tweet_json_str = parts
            try:
                tweet_json = json.loads(tweet_json_str)
                if "retweeted_status" in tweet_json:
                    tweet_json = tweet_json["retweeted_status"]
                
                if "text" in tweet_json:
                    tweet_data[tweet_id] = tweet_json["text"]
                elif "full_text" in tweet_json:
                    tweet_data[tweet_id] = tweet_json["full_text"]

            except json.JSONDecodeError as e:
                print(f"JSON デコードエラー: {e}, 行: {line}")
                continue

# JSONL ファイルとして保存
with open(output_file_path, 'w', encoding='utf-8') as file:
    for tweet_id, text in tweet_data.items():
        # 各ツイートを JSON オブジェクトに変換
        json_object = json.dumps({"tweet_id": tweet_id, "text": text}, ensure_ascii=False)
        # JSONL ファイルに書き出し
        file.write(json_object + '\n')

print(f"データが {output_file_path} に保存されました。")
