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
    processed_tweet_ids = set()  # 処理済みのツイートIDを追跡するためのセット

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # タブで区切られた行を分割し、2列目（0から数えて1）をJSONとして解析
            line = line.replace(",", "\t", 2)
            json_string = line.split('\t')[2]
            tweet = json.loads(json_string.rstrip('\n|\r'))

            # ツイートのみを取得
            if "retweeted_status" in tweet:
                tweet = tweet["retweeted_status"]
                if "created_at" in tweet and "user" in tweet and "id_str" in tweet["user"]:
                    tweet_id = tweet['id_str']
                    if tweet_id not in processed_tweet_ids:  # 重複チェック
                        reduced_data = {
                            'created_at': tweet['created_at'],
                            'user': tweet['user'],
                            'tweet_id': tweet_id
                        }
                        data.append(reduced_data)
                        processed_tweet_ids.add(tweet_id)  # 処理済みとしてIDを追加

    return data

def count_tweet_users(input_file, start_date, end_date, period_weeks, output_csv, output_png, title, id):
    period = timedelta(weeks=int(period_weeks))

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
        user_count = weekly_tweets['user'].apply(lambda x: x['id']).nunique()

        counts.append({'date': current_date.strftime('%Y-%m-%d'), 'count': user_count})
        current_date = next_date

    df = pd.DataFrame(counts)
    df.set_index('date', inplace=True)

    df.to_csv(output_csv)
    df['month_day'] = df.index.str.split('-').str[1] + '-' + df.index.str.split('-').str[2]

    # プロットの作成と保存
    df.plot(kind='line', x='month_day', y='count', marker='o', label='週間ツイートユーザ数')
    plt.xticks(rotation=45)
    # plt.title(f'{id}\n{title} : 週間ツイートユーザ数の推移',  fontsize=16)
    plt.title(f'{title}', fontsize=16)
    plt.xlabel('放送日', fontsize=14)
    plt.ylabel('週間ツイートユーザ数', fontsize=14)
    plt.tight_layout()  # ラベルが画像の外に出ないように調整
    plt.legend()
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
    # end_date = end_date + timedelta(days=7)

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
