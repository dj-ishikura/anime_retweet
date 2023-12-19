

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_distribution_stat(dir_path):
    stats = []
    
    # 指定ディレクトリ内のすべてのCSVファイルを読み込む
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.csv'):
            filepath = os.path.join(dir_path, file_name)
            df = pd.read_csv(filepath, index_col=0)
            stats.append(df.loc[['mean', '50%']].values.flatten())  # 平均と中央値を集める

    # 統計データをデータフレームに変換
    stats_df = pd.DataFrame(stats, columns=['followers_mean', 'following_mean', 'followers_median', 'following_median'])

    # ヒストグラムをプロット
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    bins = np.logspace(np.log10(1), np.log10(10**7), num=8)

    ax[0, 0].hist(stats_df['followers_mean'], bins=bins, edgecolor='black', log=True)
    ax[0, 0].set_title('Followers Mean Distribution')
    ax[0, 0].set_xscale('log')
    
    ax[0, 1].hist(stats_df['following_mean'], bins=bins, edgecolor='black', log=True)
    ax[0, 1].set_title('Following Mean Distribution')
    ax[0, 1].set_xscale('log')

    ax[1, 0].hist(stats_df['followers_median'], bins=bins, edgecolor='black', log=True)
    ax[1, 0].set_title('Followers Median Distribution')
    ax[1, 0].set_xscale('log')

    ax[1, 1].hist(stats_df['following_median'], bins=bins, edgecolor='black', log=True)
    ax[1, 1].set_title('Following Median Distribution')
    ax[1, 1].set_xscale('log')
    
    # 保存するのだ
    plt.tight_layout()
    plt.savefig(f'plot_mean_follow.png')

    # 統計情報をCSVファイルとして保存するのだ
    stat_stat = stats_df.describe()
    stat_stat.to_csv('plot_mean_follow.csv', index=True)

# ディレクトリパスを指定して関数を呼び出す
plot_distribution_stat('follow_user_distribution')
