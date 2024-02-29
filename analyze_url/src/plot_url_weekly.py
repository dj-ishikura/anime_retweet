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
from urllib.parse import urlparse

def count_tweet_url(tweet_url_file, start_date, end_date, output_csv):
    period = timedelta(weeks=int(1))

    tweets_df = pd.read_json(tweet_url_file, lines=True, dtype={'tweet_id': str})
    
    current_date = start_date
    counts = []

    while current_date <= end_date:
        next_date = current_date + period

        weekly_tweets = tweets_df[(tweets_df['created_at'] >= current_date) & (tweets_df['created_at'] < next_date)]
        tweet_count = len(weekly_tweets)
        media_count = weekly_tweets['media'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
        photo_count = weekly_tweets['media'].apply(lambda x: any(item['type'] == 'photo' for item in x) if isinstance(x, list) else False).sum()
        video_count = weekly_tweets['media'].apply(lambda x: any(item['type'] == 'video' for item in x) if isinstance(x, list) else False).sum()
        url_count = weekly_tweets['urls'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
        url_counts = weekly_tweets['urls'].explode().dropna().apply(lambda x: urlparse(x).netloc).value_counts().to_dict()
        pixiv_count = 0
        youtube_count = 0
        for url, count in url_counts.items():
            if url in ['www.pixiv.net']:
                pixiv_count += count
            elif url in ["youtu.be", "www.youtube.com", "youtube.com"]:
                youtube_count += count

        counts.append({
            'date': current_date.strftime('%Y-%m-%d'), 
            'tweet_count': tweet_count, 
            'media_count': media_count,
            'photo_count': photo_count,
            'video_count': video_count,
            'url_count': url_count,
            'pixiv_count': pixiv_count,
            'youtube_count': youtube_count,
            'othoer_tweet_count': tweet_count - media_count - video_count,
        })
        current_date = next_date

    df = pd.DataFrame(counts)
    df.set_index('date', inplace=True)

    df.to_csv(output_csv)
    df['month_day'] = df.index.str.split('-').str[1] + '-' + df.index.str.split('-').str[2]

    return df

def plot_tweet_weekly(df, output_png, title, id):
    # プロットの作成と保存
    df.plot(kind='bar', stacked=True, y=['media_count', 'url_count', 'othoer_tweet_count'], color=['lightblue', 'lightcoral', 'lightgreen'])

    plt.title(f'{title} \n {id}')
    plt.xticks(ticks=range(len(df['month_day'])), labels=df['month_day'], rotation=45)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlabel('放送日', fontsize=16)
    plt.ylabel('週間ツイート数', fontsize=16)
    plt.legend(labels=['media', 'url', 'other tweet'], fontsize=14)  # 凡例のラベルを指定
    plt.tight_layout()  # ラベルが画像の外に出ないように調整
    plt.savefig(output_png)

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

    tweet_url_file = sys.argv[1]
    output_csv = sys.argv[2]
    output_png = sys.argv[3]
    id = sys.argv[4]
    title, start_date, end_date = get_info_from_csv(id)

    df = count_tweet_url(tweet_url_file, start_date, end_date, output_csv)
    plot_tweet_weekly(df, output_png, title, id)