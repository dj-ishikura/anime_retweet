import json
import csv
import sys
from typing import Dict, Iterator

def process_tsv_file(file_path: str) -> Iterator[Dict]:
    """TSVファイルを処理し、各行をイテレートする"""
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) >= 2:  # IDとJSONデータが存在することを確認
                tweet_id, tweet_json = row[0], row[1]
                yield json.loads(tweet_json)

def extract_original_tweet(tweet: Dict) -> Dict:
    """リツイートの場合、オリジナルツイートを抽出する"""
    return tweet.get('retweeted_status', tweet)

def process_tweets(input_file: str, output_file: str):
    """ツイートを処理し、オリジナルツイートを抽出して出力する"""
    with open(output_file, 'w', encoding='utf-8', newline='') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerow(['original_tweet_id', 'original_tweet_json'])

        for tweet in process_tsv_file(input_file):
            original_tweet = extract_original_tweet(tweet)
            original_id = original_tweet['id_str']
            writer.writerow([original_id, json.dumps(original_tweet, ensure_ascii=False)])

    print(f"オリジナルツイートの抽出が完了しました。結果: {output_file}")

if __name__ == "__main__":
    input_file = "random_retweets_unique_2022_7-9.tsv"
    output_file = "random_tweets_2022_7-9.tsv"
    process_tweets(input_file, output_file)