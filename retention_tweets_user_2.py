import pandas as pd
import sys

# input_file = 'track_tweet/2020-01-0_1_week_tweet_track.csv'
input_file = sys.argv[1]

# データの読み込み
df = pd.read_csv(input_file)  # ファイルパスは適宜変更してください。

# 初回放送時のツイート数
max_first_tweet_count = max(df.iloc[1:, 1])  # 初回放送時のツイート数は1列目
sum_first_tweet_count = sum(df.iloc[1:, 1])
total_weeks = len(df.columns) - 2

retention_rate = sum_first_tweet_count / (max_first_tweet_count * (total_weeks))
id = input_file.split('/')[2].split('_')[0]
if retention_rate < 1.0:
    print(f'{id},{retention_rate}')
