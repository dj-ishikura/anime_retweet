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
                    if "user" in tweet and "id_str" in tweet["user"] and 'description' in tweet['user']:
                        ruduced_data = {
                            'created_at': tweet['created_at'],
                            'user_id': tweet['user']['id_str'],
                            'profile': tweet['user']['description']
                        }
                        data.append(ruduced_data)
    return data

def count_target_users(input_file, start_date, end_date, output_csv, id, keyword):
    period = timedelta(weeks=int(1))

    tweets = read_jsonl(input_file)
    tweets_df = pd.DataFrame(tweets)
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')  # convert to JST

    current_date = start_date
    counts = []
    
    while current_date <= end_date:
        next_date = current_date + period
        print(f'current_date: {current_date}, next_date: {next_date}')
        weekly_tweets = tweets_df[(tweets_df['created_at'] >= current_date) & (tweets_df['created_at'] < next_date)]
        weekly_tweets_unique = weekly_tweets.drop_duplicates(subset=['user_id'])
        user_count = weekly_tweets_unique.shape[0]

        keyword_count = weekly_tweets_unique[
            weekly_tweets_unique['profile'].str.contains('|'.join(keywords), case=False, na=False)
        ].shape[0]

        counts.append({'date': current_date.strftime('%Y-%m-%d'), 'user_count': user_count, 'keyword_count': keyword_count})
        current_date = next_date

    df = pd.DataFrame(counts)
    df.set_index('date', inplace=True)

    df.to_csv(output_csv)
    df['month_day'] = pd.to_datetime(df.index).strftime('%m-%d')

    return df

def plot_tweet_data(df, output_png, title, keywords):
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df['month_day']))
    width = 0.35

    ax.bar([i - width/2 for i in x], df['user_count'], width, label='週間ツイートユーザ数', color='blue', alpha=0.7)
    ax.bar([i + width/2 for i in x], df['keyword_count'], width, label=f'キーワードを含むユーザ数', color='red', alpha=0.7)

    ax.set_xlabel('放送日', fontsize=14)
    ax.set_ylabel('ユーザ数', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['month_day'], rotation=45, ha='right')
    
    ax.legend()

    # 2つ目の軸を追加（パーセンテージ表示用）
    ax2 = ax.twinx()
    percentage = df['keyword_count'] / df['user_count'] * 100
    ax2.plot(x, percentage, color='green', linestyle='--', marker='o', label='割合 (%)')
    ax2.set_ylabel('割合 (%)', color='green', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax2.set_ylim(0, 100)

    # 両方の凡例を統合
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f"{title}\nキーワード: {', '.join(keywords)}", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

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
    output_csv = sys.argv[2]
    output_png = sys.argv[3]
    id = sys.argv[4]
    keywords = sys.argv[5].split(',')
    title, start_date, end_date = get_info_from_csv(id)

    df = count_target_users(input_file, start_date, end_date, output_csv, id, keywords)
    plot_tweet_data(df, output_png, title, keywords)
