import pandas as pd
import os

# 結合したいCSVファイルが保存されているディレクトリのパス
directory_path = './model_tweet/'

# 結合するCSVファイルのリストを作成
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 空のDataFrameを用意
merged_df = pd.DataFrame()

# 各CSVファイルを読み込み、DataFrameに追加
for csv_file in csv_files:
    df = pd.read_csv(directory_path + csv_file)
    merged_df = pd.concat([merged_df, df])

# 結合した結果を新しいCSVファイルに保存
merged_df.to_csv('model_tweet_concat.csv', index=False)
