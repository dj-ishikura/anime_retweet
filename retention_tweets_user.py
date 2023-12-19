import pandas as pd
import sys
import numpy as np

# input_file = './track_tweet/2020-01-0_1_week_tweet_track.csv'
# output_file = 
input_file = sys.argv[1]
output_file = sys.argv[2]

# データの読み込み
df = pd.read_csv(input_file) 

id = input_file.split('/')[2].split('_')[0]
total_weeks = len(df.columns) - 1

# print(f'id: {id}')  # idを出力するのだ
# print(f'total_weeks: {total_weeks}')  # total_weeksを出力するのだ

# 初回放送時のツイート数
first_tweet_counts = df.iloc[:, 1:-1].max()  # 最後の週は含まないのだ
# print(f'first_tweet_counts: {first_tweet_counts}')  # first_tweet_countsを出力するのだ

sum_first_tweet_counts = df.iloc[:, 1:-1].sum()  # 各週のツイート数の合計を計算するのだ
# print(f'sum_first_tweet_counts: {sum_first_tweet_counts}')  # sum_first_tweet_countsを出力するのだ

# 定着率の計算
retention_rates = sum_first_tweet_counts / (first_tweet_counts * (total_weeks - np.arange(0, total_weeks-1))) 

# 定着率をCSVファイルに保存するのだ
retention_rates.to_csv(output_file, header=True, index=True)

# 定着率の平均を計算するのだ
average_rate = retention_rates.mean()
print(f'{id},{average_rate}')
