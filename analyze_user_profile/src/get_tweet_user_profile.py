# -*- coding: utf-8 -*-
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob
import matplotlib.pyplot as plt
import pytz
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib
import csv

def read_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # タブで区切られた行を分割し、2列目（0から数えて1）をJSONとして解析
            line = line.replace(",", "\t", 2)
            json_string = line.split('\t')[2]
            tweet = json.loads(json_string.rstrip('\n|\r'))
            # ツイートのみを取得
            if "retweeted_status" in tweet:
                tweet = tweet["retweeted_status"]
                
            if "created_at" in tweet:
                if "user" in tweet and "id_str" in tweet["user"] and 'description' in tweet['user']:
                    ruduced_data = {
                        'created_at': tweet['created_at'],
                        'user_id': tweet['user']['id_str'],
                        'profile': tweet['user']['description']
                    }
                    data.append(ruduced_data)

    return data


def get_user_profile(input_file, start_date, end_date, output_file):
    tweets = read_jsonl(input_file)
    tweets_df = pd.DataFrame(tweets)
    print(f'データ数 : {len(tweets_df)}')
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')  # convert to JST

    current_date = start_date
    
    tweets_df = tweets_df[(tweets_df['created_at'] >= start_date) & (tweets_df['created_at'] < end_date + timedelta(weeks=1))]

    tweets_df = tweets_df.drop_duplicates()

    tweets_df['created_at'] = tweets_df['created_at'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    print(f'取得したデータ数 : {len(tweets_df)}')

    tweets_df.to_json(output_file, orient='records', lines=True, force_ascii=False, date_format='iso')

def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']
    start_date = df.loc[id, '開始日']
    start_date = datetime.strptime(start_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))
    end_date = df.loc[id, '終了日']
    end_date = datetime.strptime(end_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))
    # end_date = end_date + timedelta(days=7)

    return title, start_date, end_date

if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    id = sys.argv[3]
    title, start_date, end_date = get_info_from_csv(id)

    get_user_profile(input_file, start_date, end_date, output_file)