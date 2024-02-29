import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 日本語対応
import japanize_matplotlib  # 日本語ラベル対応
import json
from scipy import stats

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def calculate_and_plot(count_url_data, df_class):
    cluster_colors = {
        0: 'red',   # 上昇
        1: 'blue',  # 下降
        2: 'green', # W型(山型)
        3: 'purple' # U型(横ばい)
    }

    plt.figure(figsize=(12, 8))
    cluster_data = {0: {'x': [], 'y': []}, 1: {'x': [], 'y': []}, 2: {'x': [], 'y': []}, 3: {'x': [], 'y': []}}

    amzn_counts = []
    for anime in count_url_data:
        id = anime['id']
        title = df_class.loc[id, 'title']
        amzn_count = anime['urls'].get('amzn.to', 0)
        mean_tweet_user_count = df_class.loc[id, 'mean_tweet_user_count']
        
        weekly_tweet_user_clusters = df_class.loc[id, 'weekly_tweet_user_clusters']
        cluster_data[weekly_tweet_user_clusters]['x'].append(amzn_count)
        cluster_data[weekly_tweet_user_clusters]['y'].append(mean_tweet_user_count)

        color = cluster_colors[weekly_tweet_user_clusters]
        plt.scatter(amzn_count, mean_tweet_user_count, color=color)
        # plt.text(amzn_count, mean_tweet_user_count, title, fontsize=9, alpha=0.7)
        
    """
    for cluster in cluster_data:
        x_values = cluster_data[cluster]['x']
        y_values = cluster_data[cluster]['y']
        if x_values and y_values:
            A = np.vstack([x_values, np.ones(len(x_values))]).T
            m, c = np.linalg.lstsq(A, y_values, rcond=None)[0]
            plt.plot(x_values, m*np.array(x_values) + c, color=cluster_colors[cluster], alpha=0.5)
    """
    
    # 凡例を作成
    plt.xlabel('amznのURLを含むツイートの数')
    plt.ylabel('平均週間ツイートユーザ数')
    legend_labels = ['上昇', '下降', '山型', '横ばい']
    legend_colors = [cluster_colors[i] for i in range(4)]
    plt.legend([plt.Line2D([0], [0], color=color, marker='o', linestyle='') for color in legend_colors], legend_labels)
    
    # グラフを保存
    plt.savefig("./results/amzn_vs_mean_tweet_users.png")
    plt.close()
    
    # 相関係数と有意性の計算
    all_x_values = []
    all_y_values = []
    for cluster in cluster_data:
        x_values = cluster_data[cluster]['x']
        y_values = cluster_data[cluster]['y']
        all_x_values.extend(x_values)
        all_y_values.extend(y_values)

        if x_values and y_values:
            # 相関係数の計算
            correlation, p_value = stats.pearsonr(x_values, y_values)
            cluster = legend_labels[cluster]
            print(f"クラスター {cluster} - 相関係数: {correlation:.2f}, p値: {p_value:.4f}")
            # p値が0.05以下の場合、統計的に有意と判断
            if p_value < 0.05:
                print(f"クラスター {cluster}: 統計的に有意")
            else:
                print(f"クラスター {cluster}: 統計的に有意ではない")

    if all_x_values and all_y_values:
        # 相関係数の計算
        correlation, p_value = stats.pearsonr(all_x_values, all_y_values)
        print(f"全体 - 相関係数: {correlation:.2f}, p値: {p_value:.4f}")
        # p値が0.05以下の場合、統計的に有意と判断
        if p_value < 0.05:
            print("全体: 統計的に有意")
        else:
            print("全体: 統計的に有意ではない")

def main():
    count_url_data = load_jsonl('/work/n213304/learn/anime_retweet_2/analyze_url/results/count_url.jsonl')
    path = "/work/n213304/learn/anime_retweet_2/anime_class.csv"
    df_class = pd.read_csv(path, index_col="id")

    calculate_and_plot(count_url_data, df_class)

if __name__ == "__main__":
    main()
