import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

# データを格納するディレクトリのパス
directory_path = 'count_tweet_users'

# データを格納する辞書
data_dict = {}
anime_weekly_tweet_list = []
anime_ids = []

# 時系列データの長さを格納するリスト
lengths = []

# ディレクトリ内の各ファイルを処理するが、最初の10件のファイルのみを処理する
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        anime_id = filename.split('_')[0]
        anime_ids.append(anime_id)
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, index_col=0)
        series_length = len(df[["count"]])
        lengths.append(series_length)

# 最頻値を計算するのだ
counter = Counter(lengths)
most_common_length, most_common_frequency = counter.most_common(1)[0]
print(f'Most common length: {most_common_length}, Frequency: {most_common_frequency}')

# ヒストグラムをプロットするのだ
plt.figure(figsize=(15, 9))  # ここで12は幅、6は高さを指定するのだ
# plt.hist(lengths, bins=np.arange(min(lengths), max(lengths)+1) - 0.5, edgecolor='black')
plt.hist(lengths, bins=np.arange(1, 28) - 0.5, edgecolor='black')
# plt.title('Distribution of Series Lengths')
plt.xlabel('放送期間', fontsize=20)
plt.ylabel('アニメ作品数', fontsize=20)
plt.xticks(np.arange(1, 27), fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.savefig("plot_week_length.png")

