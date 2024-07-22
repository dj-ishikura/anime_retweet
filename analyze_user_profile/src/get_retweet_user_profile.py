# -*- coding: utf-8 -*-
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob
import matplotlib.pyplot as plt
import pytz
from matplotlib.font_manager import FontProperties
import japanize_matplotlib
import csv

def read_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace(",", "\t", 2)
            json_string = line.split('\t')[2]
            tweet = json.loads(json_string.rstrip('\n|\r'))
            
            if tweet in "user":
                user_data = tweet["user"]
            
                if "created_at" in tweet and "id_str" in user_data and 'description' in user_data:
                    reduced_data = {
                        'created_at': tweet['created_at'],
                        'user_id': user_data['id_str'],
                        'profile': user_data['description']
                    }
                    data.append(reduced_data)
    return data

def get_user_profile(input_file, start_date, end_date, output_file):
    tweets = read_jsonl(input_file)
    tweets_df = pd.DataFrame(tweets)
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')  # convert to JST

    tweet_df = tweets_df[(tweets_df['created_at'] >= start_date) & (tweets_df['created_at'] < end_date + timedelta(weeks=1))]

    tweet_df.to_json(output_file, orient='records', lines=True, force_ascii=False)

def get_info_from_csv(id):
    df = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)

    title = df.loc[id, '作品名']
    start_date = df.loc[id, '開始日']
    start_date = datetime.strptime(start_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))
    end_date = df.loc[id, '終了日']
    end_date = datetime.strptime(end_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))

    return title, start_date, end_date

if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    id = sys.argv[3]
    title, start_date, end_date = get_info_from_csv(id)

    get_user_profile(input_file, start_date, end_date, output_file)