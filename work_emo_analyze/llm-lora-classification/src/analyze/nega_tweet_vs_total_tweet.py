import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 日本語対応
import japanize_matplotlib  # 日本語ラベル対応

def calculate_correlations_and_plot(tweet_emo_dir, df_class, df_title):
    cluster_colors = {
        0: 'red',   # 上昇
        1: 'blue',  # 下降
        2: 'green', # U型(横ばい)
        3: 'purple' # W型(山型)
    }

    plt.figure(figsize=(12, 8))
    cluster_data = {0: {'x': [], 'y': []}, 1: {'x': [], 'y': []}, 2: {'x': [], 'y': []}, 3: {'x': [], 'y': []}}

    for file_name in os.listdir(tweet_emo_dir):
        if file_name.endswith('.csv'):
            id = os.path.splitext(file_name)[0]
            file_path = os.path.join(tweet_emo_dir, file_name)
            df = pd.read_csv(file_path)

            negative_counts = df['negative'].sum()
            total_tweet_counts = df['tweet_count'].sum()
            
            weekly_tweet_user_clusters = df_class.loc[id, 'weekly_tweet_user_clusters']
            color = cluster_colors[weekly_tweet_user_clusters]

            cluster_data[weekly_tweet_user_clusters]['x'].append(negative_counts)
            cluster_data[weekly_tweet_user_clusters]['y'].append(total_tweet_counts)

            plt.scatter(negative_counts, total_tweet_counts, color=color)
            # plt.text(negative_counts, total_tweet_counts, df_title.loc[id, '作品名'], fontsize=9, alpha=0.7)
    
    # 各クラスタごとに最小二乗法で回帰線を計算しプロット
    for cluster in cluster_data:
        x_values = cluster_data[cluster]['x']
        y_values = cluster_data[cluster]['y']
        if x_values and y_values:
            A = np.vstack([x_values, np.ones(len(x_values))]).T
            m, c = np.linalg.lstsq(A, y_values, rcond=None)[0]
            plt.plot(x_values, m*np.array(x_values) + c, color=cluster_colors[cluster], alpha=0.5)

    # 凡例を作成
    plt.xlabel('ネガティブツイート数')
    plt.ylabel('ツイート総数')
    plt.title('アニメ作品別 ネガティブツイート数 vs ツイート総数')
    legend_labels = ['上昇', '下降', 'U型(横ばい)', 'W型(山型)']
    legend_colors = [cluster_colors[i] for i in range(4)]
    plt.legend([plt.Line2D([0], [0], color=color, marker='o', linestyle='') for color in legend_colors], legend_labels)
    
    # グラフを保存
    plt.savefig("./src/analyze/nega_tweet_vs_total_tweet.png")
    plt.close()

def main():
    tweet_emo_dir = 'tweet_emo_weekly'
    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)
    path = "/work/n213304/learn/anime_retweet_2/anime_class.csv"
    df_class = pd.read_csv(path, index_col="id")

    calculate_correlations_and_plot(tweet_emo_dir, df_class, df_title)

if __name__ == "__main__":
    main()
