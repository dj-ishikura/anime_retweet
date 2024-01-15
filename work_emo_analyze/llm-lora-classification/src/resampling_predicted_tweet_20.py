import json
import os
import pandas as pd
import random

df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

directory = '/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/prediction'

file_list = [
    '2022-10-582.json',
    '2021-10-369.json',
    '2022-01-417.json',
    '2022-10-588.json',
    '2020-10-136.json',
    '2022-10-547.json',
    '2021-10-350.json',
    '2021-01-205.json',
    '2022-04-484.json',
    '2020-10-135.json',
    '2022-04-452.json',
]

# 各ファイルからランダムに20件のツイートを取得
max_tweets = 20
all_tweets = []

for file_name in file_list:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = json.load(file)
        selected_tweets = random.sample(tweets, min(len(tweets), max_tweets))

        # アニメIDをファイル名から抽出
        anime_id = os.path.splitext(file_name)[0]
        anime_title = df_title.loc[anime_id, '作品名'] if anime_id in df_title.index else '不明'

        # 各ツイートにアニメタイトルとIDを追加
        for tweet in selected_tweets:
            tweet['anime_id'] = anime_id
            tweet['anime_title'] = anime_title

        all_tweets.extend(selected_tweets)

# すでに取得している all_tweets をDataFrameに変換
df_tweets = pd.DataFrame(all_tweets)

# ツイートのテキスト中のタブをスペースに置換して問題を防ぐ
df_tweets['text'] = df_tweets['text'].str.replace('\t', ' ', regex=False)

# CSVファイルに出力
output_file_path = 'resampling_predicted_tweet.csv'  # 出力ファイル名
df_tweets.to_csv(output_file_path, sep='\t', index=False, encoding='utf-8')
