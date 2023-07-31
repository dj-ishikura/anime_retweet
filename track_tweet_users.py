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
import matplotlib.cm as cm
import numpy as np

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
                    if "user" in tweet:
                        reduced_data = {
                            'created_at': tweet['created_at'],
                            'user_id': tweet['user']['id']
                        }
            data.append(reduced_data)
    return data

def get_first_tweets(tweets_df, start_date, end_date, period):
    first_tweets = {} 
    week_number = 1
    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + period
        weekly_tweets = tweets_df[(tweets_df['created_at'] >= current_date) & (tweets_df['created_at'] < next_date)]
        for user_id, tweet_date in weekly_tweets[['user_id', 'created_at']].values:
            if user_id not in first_tweets:
                first_tweets[user_id] = week_number

        current_date = next_date
        week_number += 1

    first_tweets_df = pd.DataFrame(list(first_tweets.items()), columns=['user_id', 'week_number'])
    df = pd.merge(tweets_df, first_tweets_df, on='user_id')
    
    return df


def count_tweet_users(input_file, start_date, end_date, period_weeks, output_csv, output_png, title, id):
    period = timedelta(weeks=int(period_weeks))

    tweets = read_jsonl(input_file)
    tweets_df = pd.DataFrame(tweets)
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')  # convert to JST

    tweets_df = get_first_tweets(tweets_df, start_date, end_date, period)  # Dictionary to hold the first tweet of each user
    weekly_counts = pd.DataFrame(columns=['date', 'count', 'first_time_users'])

    current_date = start_date
    counts = []

    while current_date <= end_date:
        next_date = current_date + period
        weekly_tweets = tweets_df[(tweets_df['created_at'] >= current_date) & (tweets_df['created_at'] < next_date)]
        weekly_tweets = pd.get_dummies(weekly_tweets.week_number)
        # 列名を整数型に変換
        weekly_tweets.columns = weekly_tweets.columns.astype(int)
        # 列名をソート
        weekly_tweets = weekly_tweets.sort_index(axis=1)
        count_tweet = weekly_tweets.sum()
        count_tweet_dict = count_tweet.to_dict()
        count_tweet_dict['date'] = current_date.strftime('%Y-%m-%d')
        counts.append(count_tweet_dict)

        current_date = next_date

    weekly_counts = pd.DataFrame(counts)
    weekly_counts.set_index('date', inplace=True)
    weekly_counts.fillna(0, inplace=True)
    weekly_counts.to_csv(output_csv)

    # コラム名を整数に変換し、ソートします
    weekly_counts.columns = weekly_counts.columns.astype(int)
    weekly_counts = weekly_counts.sort_index(axis=1)

    # colormap の生成
    colors = cm.inferno(np.linspace(0, 1, len(weekly_counts.columns)))

    # プロットの作成
    ax = weekly_counts.plot(kind='area', stacked=True, color=colors, alpha=0.7)

    plt.title(f'{id}\n{title} : Tweet Users Count, period {period_weeks}')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend().set_visible(False)
    plt.savefig(output_png)


   

def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('./anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']
    start_date = df.loc[id, '開始日']
    start_date = datetime.strptime(start_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))
    end_date = df.loc[id, '終了日']
    end_date = datetime.strptime(end_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))

    return title, start_date, end_date

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print('Usage: python count_tweet_users.py $input_file $output_csv $output_png $output_dir $id')
        sys.exit(1)

    input_file = sys.argv[1]
    period_weeks = sys.argv[2]
    output_csv = sys.argv[3]
    output_png = sys.argv[4]
    id = sys.argv[5]
    title, start_date, end_date = get_info_from_csv(id)

    count_tweet_users(input_file, start_date, end_date, period_weeks, output_csv, output_png, title, id)
