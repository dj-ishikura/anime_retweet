import os
import pandas as pd
import json
import sys
from dateutil import parser
import pytz
from datetime import datetime, timedelta

def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']
    start_date = df.loc[id, '開始日']
    start_date = datetime.strptime(start_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))
    end_date = df.loc[id, '終了日']
    end_date = datetime.strptime(end_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))

    return title, start_date, end_date

# extra_anime_retweet_concat ディレクトリ内のすべてのファイルをループ処理
output_dir = 'extra_anime_retweet_concat'
tweet_data = {}  # ツイート ID とテキストを保存するための辞書

file_path = sys.argv[1]
output_file_path = sys.argv[2]
id = sys.argv[3]
title, start_date, end_date = get_info_from_csv(id)

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.split(',', 2)  # 最初の2つのコンマで分割
        if len(parts) < 3:
            continue  # データが不完全な行はスキップ

        tweet_id, hashtag, tweet_json_str = parts
        try:
            tweet_json = json.loads(tweet_json_str)
            if "retweeted_status" in tweet_json:
                tweet_json = tweet_json["retweeted_status"]

                if "full_text" in tweet_json:
                    tweet_data[tweet_id] = tweet_json["full_text"]
                elif "text" in tweet_json:
                    tweet_data[tweet_id] = tweet_json["text"]

        except json.JSONDecodeError as e:
            print(f"JSON デコードエラー: {e}, 行: {row}")
            continue  

import json

# JSONL ファイルとして保存
with open(output_file_path, 'w', encoding='utf-8') as file:
    for tweet_id, text in tweet_data.items():
        # 各ツイートを JSON オブジェクトに変換
        json_object = json.dumps({"tweet_id": tweet_id, "text": text}, ensure_ascii=False)
        # JSONL ファイルに書き出し
        file.write(json_object + '\n')


