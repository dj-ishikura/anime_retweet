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
                    ruduced_data = {
                        'created_at': tweet['created_at'],
                        'user': tweet['user'],
                        'tweet_id': tweet['id_str']
                    }
                    data.append(ruduced_data)
    return data

def count_tweet_users(input_file, start_date, end_date, output_csv, output_png, title, id, broadcast_weeks):
    period = timedelta(weeks=1)

    tweets = read_jsonl(input_file)
    tweets_df = pd.DataFrame(tweets)
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')

    current_date = start_date
    counts = []
    week_number = 1

    while current_date <= end_date:
        next_date = current_date + period
        weekly_tweets = tweets_df[(tweets_df['created_at'] >= current_date) & (tweets_df['created_at'] < next_date)]
        user_count = weekly_tweets['user'].apply(lambda x: x['id']).nunique()
        tweet_count = weekly_tweets['tweet_id'].nunique()

        counts.append({'date': current_date.strftime('%Y-%m-%d'), 
                       'tweet_users_count': user_count, 
                       'tweet_count': tweet_count, 
                       'tweet_ratio': tweet_count / user_count if user_count != 0 else 0,
                       'week': f'Week {week_number}'})
        current_date = next_date
        week_number += 1

    df = pd.DataFrame(counts)
    df.set_index('date', inplace=True)

    df.to_csv(output_csv)

    # プロットの作成と保存
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # 放送終了週を特定
    final_broadcast_week = f'Week {broadcast_weeks}'
    
    for column in ['tweet_users_count', 'tweet_count']:
        color = 'blue' if column == 'tweet_users_count' else 'green'
        ax.plot(df['week'], df[column], marker='o', linestyle='-', color=color, label=column)
        
        # 放送終了週のポイントを赤色で表示
        ax.plot(final_broadcast_week, df.loc[df['week'] == final_broadcast_week, column], 
                marker='o', markersize=10, color='red')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('週数', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.legend()

    # 放送期間を灰色の背景で表示
    ax.axvspan('Week 1', final_broadcast_week, alpha=0.2, color='gray')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def get_info_from_csv(id):
    df = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)
    title = df.loc[id, '作品名']
    start_date = df.loc[id, '開始日']
    start_date = datetime.strptime(start_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))
    end_date = df.loc[id, '終了日']
    end_date = datetime.strptime(end_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))
    broadcast_weeks = (end_date - start_date).days // 7 + 1  # 放送週数を計算
    end_date = start_date + timedelta(weeks=broadcast_weeks * 2)  # 放送週の2倍の期間を設定

    return title, start_date, end_date, broadcast_weeks

if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_csv = sys.argv[2]
    output_png = sys.argv[3]
    id = sys.argv[4]
    title, start_date, end_date, broadcast_weeks = get_info_from_csv(id)

    count_tweet_users(input_file, start_date, end_date, output_csv, output_png, title, id, broadcast_weeks)
