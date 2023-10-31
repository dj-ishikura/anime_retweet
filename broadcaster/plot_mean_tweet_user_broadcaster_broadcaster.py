import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib
import numpy as np

# anime_data_updated.csvからデータを取得
anime_data = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv')

# 平均週間ツイートユーザ数をresult/tweet_mean.csvからとる。
user_mean_data = pd.read_csv('../result/tweet_mean.csv')

# anime_dataとuser_mean_dataをidで結合する
merged_data = pd.merge(anime_data, user_mean_data, on='id', how='inner')
merged_data = merged_data.set_index('id')

filtered_rows = []
data_dir = 'data/'

# 行数を保持するためのリスト
rows_count = {}

for anime_id, row in merged_data.iterrows():
    # デバック用
    # anime_id = '2020-04-51'
    # row = merged_data.loc[anime_id].to_dict()

    file_path = os.path.join(data_dir, f"{anime_id}.csv")

    
    if os.path.exists(file_path):
        start = pd.to_datetime(row['開始日'], format='%Y年%m月%d日')
        end = pd.to_datetime(row['終了日'], format='%Y年%m月%d日')
        # startの値を1週間後ろにする
        start -= pd.Timedelta(weeks=1)

        # on_bad_linesを使用する
        data = pd.read_csv(file_path, on_bad_lines='skip')

        # '開始'の列が存在するか確認
        if '開始' in data.columns and 'タイプ' in data.columns:
            data = data[data['タイプ'] == '放送']
            data_date = pd.to_datetime(data['開始'], errors='coerce', format='%Y-%m-%d')
            mask = (data_date >= start) & (data_date <= end)
            unique_data = data[mask].drop_duplicates(subset='放送局')
            filtered_rows.append(unique_data)
            rows_count[anime_id] = len(unique_data)


user_mean_for_plot = merged_data.loc[rows_count.keys(), 'user_mean']

# 相関係数と回帰直線を計算する
correlation_coefficient = np.corrcoef(list(rows_count.values()), user_mean_for_plot)[0, 1]
slope, intercept = np.polyfit(list(rows_count.values()), user_mean_for_plot, 1)

# プロットする
plt.scatter(rows_count.values(), user_mean_for_plot, label='データポイント')
plt.plot(list(rows_count.values()), slope * np.array(list(rows_count.values())) + intercept, color='red', label=f'回帰直線: y={slope:.2f}x+{intercept:.2f}')
plt.xlabel('放送局数')
plt.ylabel('平均週間ツイートユーザ数')
plt.title(f'放送局数と平均週間ツイートユーザ数の散布図\nピアソンの相関係数: {correlation_coefficient:.2f}')
plt.legend()

plt.savefig("plot_mean_tweet_user_broadcaster_broadcaster.png")

# データをDataFrameとしてまとめる
result_df = pd.DataFrame({
    'id': list(rows_count.keys()),
    'rows_count': list(rows_count.values()),
    'user_mean': user_mean_for_plot
})

result_df = result_df.sort_values(by=['rows_count', 'user_mean'], ascending=False)

# CSVとして保存する
result_df.to_csv('plot_mean_tweet_user_broadcaster_broadcaster.csv', index=False)
