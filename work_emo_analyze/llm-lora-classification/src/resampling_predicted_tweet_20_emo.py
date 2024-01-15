import json
import os
import pandas as pd
import random

def extract_tweets(directory, file_list, max_tweets, predictions_value, output_file_path):
    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)
    all_tweets = []

    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            tweets = json.load(file)
            # 指定されたpredictions値のツイートをフィルタリング
            filtered_tweets = [tweet for tweet in tweets if tweet['predictions'] == predictions_value]
            
            # ランダムにツイートを抽出
            selected_tweets = random.sample(filtered_tweets, min(len(filtered_tweets), max_tweets))

            # アニメIDをファイル名から抽出
            anime_id = os.path.splitext(file_name)[0]
            anime_title = df_title.loc[anime_id, '作品名'] if anime_id in df_title.index else '不明'

            # 各ツイートにアニメタイトルとIDを追加
            for tweet in selected_tweets:
                tweet['anime_id'] = anime_id
                tweet['anime_title'] = anime_title

            all_tweets.extend(selected_tweets)

    # DataFrameに変換し、CSVファイルに出力
    df_tweets = pd.DataFrame(all_tweets)
    df_tweets['text'] = df_tweets['text'].str.replace('\t', ' ', regex=False)
    df_tweets.to_csv(output_file_path, sep='\t', index=False, encoding='utf-8')

    return all_tweets

# 使用例
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

# predictionsが0のツイートを抽出
max_tweets = 20
output_file_path = "resampling_predicted_tweet_nega.tsv"
extracted_tweets = extract_tweets(directory, file_list, max_tweets, 0, output_file_path)
output_file_path = "resampling_predicted_tweet_neut.tsv"
extracted_tweets = extract_tweets(directory, file_list, max_tweets, 1, output_file_path)
output_file_path = "resampling_predicted_tweet_posi.tsv"
extracted_tweets = extract_tweets(directory, file_list, max_tweets, 2, output_file_path)


