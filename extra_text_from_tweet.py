import json
import csv
import sys
from typing import Dict, Iterator

def process_tsv_file(file_path: str) -> Iterator[Dict]:
    """TSVファイルを処理し、各行をイテレートする"""
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # ヘッダーをスキップ
        for row in reader:
            if len(row) >= 2:  # IDとJSONデータが存在することを確認
                tweet_id, tweet_json = row[0], row[1]
                yield json.loads(tweet_json)

def extract_text(tweet: Dict) -> str:
    """ツイートからテキストを抽出する"""
    # 'full_text' フィールドがある場合はそれを使用し、なければ 'text' フィールドを使用
    return tweet.get('full_text', tweet.get('text', ''))

def process_tweets(input_file: str, output_file: str):
    """ツイートを処理し、tweet_idとtextを抽出してJSONL形式で出力する"""
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for tweet in process_tsv_file(input_file):
            tweet_id = tweet['id_str']
            text = extract_text(tweet)
            output_json = {
                "tweet_id": tweet_id,
                "text": text
            }
            json.dump(output_json, out_file, ensure_ascii=False)
            out_file.write('\n')

    print(f"テキストの抽出が完了しました。結果: {output_file}")

if __name__ == "__main__":
    input_file = "random_tweets_2022_7-9.tsv"
    output_file = "random_tweets_text_2022_7-9.jsonl"
    process_tweets(input_file, output_file)