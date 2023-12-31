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

def read_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # タブで区切られた行を分割し、2列目（0から数えて1）をJSONとして解析
            line = line.replace(",", "\t", 2)
            json_string = line.split('\t')[2]
            tweet = json.loads(json_string.rstrip('\n|\r'))
            # ツイートのみを取得
            try:
                ruduced_data = {
                    'created_at': tweet.get('created_at'),
                    'user': tweet['user']['id']
                }
                if ruduced_data['created_at'] is not None:
                    data.append(ruduced_data)
            except KeyError:
                pass

    return data


def count_tweet_users(input_file, start_date, end_date, output_txt, title, id):

    tweets = read_jsonl(input_file)
    tweets_df = pd.DataFrame(tweets)
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')  # convert to JST

    current_date = start_date
    counts = []

    weekly_tweets = tweets_df[(tweets_df['created_at'] >= start_date) & (tweets_df['created_at'] < end_date)]
    user_list = tweets_df['user'].unique().tolist()

    with open(output_txt, 'w') as f:
        for item in user_list:
            f.write("%s\n" % item)

def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('./anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']
    start_date = df.loc[id, '開始日']
    start_date = datetime.strptime(start_date, "%Y年%m月%d日").replace(tzinfo=pytz.UTC)
    end_date = df.loc[id, '終了日']
    end_date = datetime.strptime(end_date, "%Y年%m月%d日").replace(tzinfo=pytz.UTC)
    end_date = end_date + timedelta(days=7)

    return title, start_date, end_date

if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_txt = sys.argv[2]
    id = sys.argv[3]
    title, start_date, end_date = get_info_from_csv(id)

    count_tweet_users(input_file, start_date, end_date, output_txt, title, id)
