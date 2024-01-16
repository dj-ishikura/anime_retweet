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

def read_csv_tweet(input_path):
    data = []
    tweet_ids = set()  # 重複チェック用のセット
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace(",", "\t", 2)
            json_string = line.split('\t')[2]
            tweet = json.loads(json_string.rstrip('\n|\r'))

            # リツイートされたツイートは除外
            if "retweeted_status" in tweet:
                tweet = tweet["retweeted_status"]
            if "created_at" in tweet and "id_str" in tweet and "user" in tweet and "id_str" in tweet["user"]:
                # tweet_idが重複していないことを確認
                if tweet['id_str'] not in tweet_ids:
                    tweet_ids.add(tweet['id_str'])
                    reduced_data = {
                        'user_id': tweet['user']['id_str'],  # ユーザID
                        'created_at': tweet['created_at'],
                        'tweet_id': tweet['id_str']
                    }
                    data.append(reduced_data)
    tweets_df = pd.DataFrame(data)
    return tweets_df

def tweet_interval(tweet_object_file, start_date, end_date, output_csv):
    period = timedelta(weeks=1)
    tweets_df = read_csv_tweet(tweet_object_file)

    # 日時をパースしてタイムゾーンを変換
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')

    # 指定された期間でフィルタリング
    tweets_df = tweets_df[(tweets_df['created_at'] >= start_date) & (tweets_df['created_at'] <= (end_date + period))]

    # ユーザごとにソート
    tweets_df.sort_values(by=['user_id', 'created_at'], inplace=True)

    # 各ユーザのツイート間隔を計算
    tweets_df['interval'] = tweets_df.groupby('user_id')['created_at'].diff()

    # ツイート間隔を秒単位に変換
    tweets_df['interval'] = tweets_df['interval'].dt.total_seconds()

    # ツイート間隔が30分（1800秒）以上のものを除外
    tweets_df = tweets_df[tweets_df['interval'] < 1800]

    # ユーザごとに平均ツイート間隔を計算し、ツイートが1つだけのユーザを除外
    avg_intervals = tweets_df.groupby('user_id').agg(
        average_interval=('interval', 'mean'),
        tweet_count=('interval', 'count')
    ).reset_index()
    avg_intervals = avg_intervals[avg_intervals['tweet_count'] > 1]

    # 結果をCSVファイルに出力
    avg_intervals.to_csv(output_csv, index=False)
    

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

    tweet_object_file = sys.argv[1]
    output_csv = sys.argv[2]
    id = sys.argv[3]
    title, start_date, end_date = get_info_from_csv(id)

    tweet_interval(tweet_object_file, start_date, end_date, output_csv)