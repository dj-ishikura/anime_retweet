import pandas as pd
import os

tweet_emo_dir = 'tweet_emo_weekly'

# ディレクトリ内のすべてのCSVファイルに対して処理を行う
for file_name in os.listdir(tweet_emo_dir):
    if file_name.endswith('.csv'):
        print(f'{file_name}')
        file_path = os.path.join(tweet_emo_dir, file_name)
        df = pd.read_csv(file_path)

        total_positive = df['positive'].sum()
        total_neutral = df['neutral'].sum()
        total_negative = df['negative'].sum()

        # 総ツイート数を計算
        total_tweet_count = df['tweet_count'].sum()

        # 割合を計算
        total_positive_ratio = total_positive / total_tweet_count
        total_neutral_ratio = total_neutral / total_tweet_count
        total_negative_ratio = total_negative / total_tweet_count
        print(f'ボジティブ : {total_positive_ratio}')
        print(f'ニュートラル : {total_neutral_ratio}')
        print(f'ネガティブ : {total_negative_ratio}\n')


