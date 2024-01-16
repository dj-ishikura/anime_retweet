import pandas as pd
import os

# 空のデータフレームを作成
anime_stats = pd.DataFrame(columns=['anime_id', 'avg_interval', 'user_count'])

directory = 'user_tweet_interval_anime'

# 各CSVファイルからデータを集計
for file_name in os.listdir(directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)

        # 平均ツイート間隔を計算
        avg_interval = df['average_interval'].mean()

        # ユーザ数を計算
        user_count = df['user_id'].nunique()

        # アニメIDを取得
        anime_id = os.path.splitext(file_name)[0]

        # 集計データを追加
        anime_stats = anime_stats.append({'anime_id': anime_id, 'avg_interval': avg_interval, 'user_count': user_count}, ignore_index=True)

# 相関を計算
# データ型の確認と必要に応じて変換
anime_stats['avg_interval'] = pd.to_numeric(anime_stats['avg_interval'], errors='coerce')
anime_stats['user_count'] = pd.to_numeric(anime_stats['user_count'], errors='coerce')

# NaN値の除去
anime_stats.dropna(subset=['avg_interval', 'user_count'], inplace=True)

# 相関を計算
correlation = anime_stats['avg_interval'].corr(anime_stats['user_count'])

print("相関係数:", correlation)


print("相関係数:", correlation)