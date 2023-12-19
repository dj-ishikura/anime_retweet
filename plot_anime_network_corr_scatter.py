import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CSVファイルが格納されているディレクトリのパスを指定するのだ
dir_path = "weekly_anime_network_corr"
anime_class_df = pd.read_csv('result/class_anime_list.csv')

# クラスと色の対応
color_dict = {'miner': 'blue', 'hit': 'green', 'trend': 'red'}

# 指定ディレクトリ内のすべてのCSVファイルを読み込む
data_frames = []
for file_name in os.listdir(dir_path):
    if file_name.endswith('.csv'):
        filepath = os.path.join(dir_path, file_name)
        df = pd.read_csv(filepath)
        df['id'] = file_name.replace(".csv", "")
        data_frames.append(df)

# すべてのデータフレームを連結するのだ
merged_df = pd.concat(data_frames)
merged_df = pd.merge(anime_class_df, merged_df, on='id')

# パラメータのリストを取得するのだ
params = merged_df['param'].unique()

# 3x3のグリッドでグラフをプロットするのだ
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# 各パラメータについて散布図を作成するのだ
for i, param in enumerate(params):
    ax = axes[i // 3, i % 3]

    # パラメータに関するデータを抽出するのだ
    param_df = merged_df[merged_df['param'] == param]
    for label in color_dict.keys():
        subset = param_df[param_df['class'] == label]

        # 散布図をプロットするのだ
        ax.scatter(subset['id'], subset['correlation'], label=param, color=color_dict[label], alpha=0.8)

        # x軸の線を削除するのだ
        ax.xaxis.set_visible(False)
        
        # タイトルを設定するのだ
        ax.set_title(param)

# グラフを保存するのだ
plt.savefig('plot_anime_network_corr_scatter.png')
