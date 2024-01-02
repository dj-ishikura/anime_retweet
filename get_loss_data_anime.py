import pandas as pd
import datetime

# ファイルパス
file_path = 'anime_data_updated.csv'

# 日付範囲
start_date = datetime.datetime(2021, 4, 14, 4, 0)
end_date = datetime.datetime(2021, 4, 20, 18, 0)

# CSVファイルを読み込み
df = pd.read_csv(file_path)

# 日付の列を datetime オブジェクトに変換
df['開始日'] = pd.to_datetime(df['開始日'], format='%Y年%m月%d日')
df['終了日'] = pd.to_datetime(df['終了日'], format='%Y年%m月%d日')

# 重複範囲を持つ行をフィルタリング
filtered_df = df[(df['開始日'] <= end_date) & (df['終了日'] >= start_date)]

# フィルタリングされた結果をCSVファイルに保存
output_file_path = 'anime_data_loss_data.csv'
filtered_df.to_csv(output_file_path, index=False)

print(f"データが {output_file_path} に保存されました。")
