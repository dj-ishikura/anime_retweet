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

def create_tweet_user_dict(input_path, tweet_ids_file):
    # tweet_idsを読み込み、セットを作成
    tweet_ids_df = pd.read_json(tweet_ids_file, dtype={'tweet_id': str})
    tweet_ids_set = set(tweet_ids_df['tweet_id'])

    # tweet_ids_setに含まれる全てのtweet_idに対して空文字列を持つ辞書を初期化
    tweet_user_dict = {tweet_id: '' for tweet_id in tweet_ids_set}

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace(",", "\t", 2)
            json_string = line.split('\t')[2]
            tweet = json.loads(json_string.rstrip('\n|\r'))

            if "retweeted_status" in tweet:
                tweet = tweet["retweeted_status"]

            tweet_id = tweet.get('id_str')
            user_id = tweet['user'].get('id_str') or str(tweet['user'].get('id'))

            if tweet_id in tweet_user_dict:
                tweet_user_dict[tweet_id] = user_id

    return tweet_user_dict

def export_dict_to_csv(tweet_user_dict, output_csv):
    # 辞書をDataFrameに変換
    df = pd.DataFrame(list(tweet_user_dict.items()), columns=['tweet_id', 'user_id'])
    
    # DataFrameをCSVファイルとして出力
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    import sys

    tweet_object_file = sys.argv[1]
    tweet_emo_file = sys.argv[2]
    output_csv = sys.argv[3]

    tweet_user_dict = create_tweet_user_dict(tweet_object_file, tweet_emo_file)
    export_dict_to_csv(tweet_user_dict, output_csv)
