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
    tweet_ids = set()  # 重複チェック用のセット
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace(",", "\t", 2)
            json_string = line.split('\t')[2]
            tweet = json.loads(json_string.rstrip('\n|\r'))
            if "retweeted_status" in tweet:
                tweet = tweet["retweeted_status"]
                if "created_at" in tweet and "id_str" in tweet:
                    # tweet_idが重複していないことを確認
                    if tweet['id_str'] not in tweet_ids:
                        tweet_ids.add(tweet['id_str'])
                        reduced_data = {
                            'created_at': tweet['created_at'],
                            'tweet_id': tweet['id_str']
                        }
                        data.append(reduced_data)
    tweets_df = pd.DataFrame(data)
    return tweets_df

def load_predictions_as_df(file_path):
    return pd.read_json(file_path, dtype={'tweet_id': str})

def count_tweet(tweet_object_file, tweet_emo_file, start_date, end_date, output_csv):
    period = timedelta(weeks=int(1))

    tweets_df = read_jsonl(tweet_object_file)
    print(tweets_df)
    tweet_emo_df = load_predictions_as_df(tweet_emo_file)
    print(tweet_emo_df)
    # tweets_df と tweet_emo_df を tweet_id 列で結合する
    tweets_df = pd.merge(tweets_df, tweet_emo_df, on='tweet_id')

    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')  # convert to JST

    current_date = start_date
    counts = []

    while current_date <= end_date:
        next_date = current_date + period
        # print(f'current_date: {current_date}, next_date: {next_date}')
        weekly_tweets = tweets_df[(tweets_df['created_at'] >= current_date) & (tweets_df['created_at'] < next_date)]
        tweet_count = len(weekly_tweets)
        positive_tweet_count = len(weekly_tweets[weekly_tweets['predictions']==2])
        neutral_tweet_count = len(weekly_tweets[weekly_tweets['predictions']==1])
        negative_tweet_count = len(weekly_tweets[weekly_tweets['predictions']==0])

        counts.append({'date': current_date.strftime('%Y-%m-%d'), 'tweet_count': tweet_count, 
        'positive': positive_tweet_count, 'neutral': neutral_tweet_count, 'negative': negative_tweet_count})
        current_date = next_date

    df = pd.DataFrame(counts)
    df.set_index('date', inplace=True)

    df.to_csv(output_csv)
    df['month_day'] = df.index.str.split('-').str[1] + '-' + df.index.str.split('-').str[2]

    return df

def plot_tweet_weekly(df, output_png, title, id):
    # プロットの作成と保存
    df.plot(kind='bar', stacked=True, y=['positive', 'neutral', 'negative'], color=['lightcoral', 'khaki', 'lightblue'])
    # plt.title(f'{id}\n{title} : Tweet Emo Count')
    plt.title(f'{title}')
    plt.xticks(ticks=range(len(df['month_day'])), labels=df['month_day'], rotation=45)
    plt.xlabel('放送日', fontsize=14)
    plt.ylabel('週間ツイート数', fontsize=14)
    plt.legend(labels=['ポジティブ', 'ニュートラル', 'ネガティブ'])  # 凡例のラベルを指定
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

    tweet_object_file = sys.argv[1]
    tweet_emo_file = sys.argv[2]
    output_csv = sys.argv[3]
    output_png = sys.argv[4]
    id = sys.argv[5]
    title, start_date, end_date = get_info_from_csv(id)

    df = count_tweet(tweet_object_file, tweet_emo_file, start_date, end_date, output_csv)
    plot_tweet_weekly(df, output_png, title, id)