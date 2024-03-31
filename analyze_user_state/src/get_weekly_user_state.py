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
                        }
                        data.append(ruduced_data)
    return data


def fetch_weekly_user_lists(input_file, start_date, end_date):

    tweets = read_jsonl(input_file)
    tweets_df = pd.DataFrame(tweets)
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')  # convert to JST

    current_date = start_date
    weekly_user_lists = []

    while current_date < end_date:
        next_date = current_date + timedelta(weeks=1)
        weekly_tweets = tweets_df[(tweets_df['created_at'] >= current_date) & (tweets_df['created_at'] < next_date)]
        
        # その週にツイートしたユーザーIDの一意のリストを取得
        weekly_users = weekly_tweets['user_id'].unique().tolist()
        weekly_user_lists.append(weekly_users)
        
        current_date = next_date
    
    return weekly_user_lists

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

def analyze_user_transitions(weekly_user_lists):
    all_users = set(user for week in weekly_user_lists for user in week)
    user_objects = {user: AnimeTweetUser(user) for user in all_users}
    
    for week in weekly_user_lists:
        for user in all_users:
            if user in week:
                user_objects[user].tweet()
            else:
                user_objects[user].no_tweet()
    
    return user_objects

def save_transitions_to_csv(user_objects, weekly_user_lists, output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['user_id'] + [f'week {week+1}' for week in range(len(weekly_user_lists))]
        writer.writerow(header)

        for user, obj in user_objects.items():
            states = [obj.state for _ in range(len(weekly_user_lists))]
            writer.writerow([user] + states)


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_csv = sys.argv[2]
    id = sys.argv[3]
    title, start_date, end_date = get_info_from_csv(id)

    weekly_user_lists = fetch_weekly_user_lists(input_file, start_date, end_date)
    user_objects = analyze_user_transitions(weekly_user_lists)
    save_transitions_to_csv(user_objects, weekly_user_lists, output_csv)