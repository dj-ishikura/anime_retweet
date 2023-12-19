# 全体のフォロワー数とフォロー数の分布
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

def plot_distribution():
    follower_data_path = "tweet_user_follower_number.jsonl"
    txt_path = "tweet_user_list.txt"

    # データを読み込むのだ
    with open(txt_path, 'r') as file:
        user_ids = [int(line.strip()) for line in file]
        
    # ユーザIDのセットを作成するのだ
    user_ids_set = set(user_ids)

    # フォロワー/フォロー数データを読み込むのだ
    follower_data = pd.read_json(follower_data_path, lines=True)
    print(follower_data)

    # ユーザIDセットを利用してマージするデータをフィルタリングするのだ
    merged_data = follower_data[follower_data['user_id'].isin(user_ids_set)]
    print(merged_data)

    # フィギュアを初期化するのだ
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))
    # フィギュア全体のタイトルを設定するのだ

    # ビンのエッジを定義するのだ
    bins = np.logspace(np.log10(1), np.log10(10**7), num=8)

    # フォロワー数の分布をプロットするのだ
    ax[0,0].hist(merged_data['followers_count'], bins=bins)
    ax[0,0].set_xscale('log')
    ax[0,0].set_title('Follower count distribution')
    ax[0,0].set_xlabel('Follower count')
    ax[0,0].set_ylabel('Frequency')

    # フォロー数の分布をプロットするのだ
    ax[0,1].hist(merged_data['following_count'], bins=bins)
    ax[0,1].set_xscale('log')
    ax[0,1].set_title('Following count distribution')
    ax[0,1].set_xlabel('Following count')
    ax[0,1].set_ylabel('Frequency')

    # フォロワー数とフォロー数の散布図をプロットするのだ
    ax[1,0].scatter(merged_data['followers_count'], merged_data['following_count'], alpha=0.5, color='blue')
    ax[1,0].set_title('Followers vs Following')
    ax[1,0].set_xlabel('Follower count')
    ax[1,0].set_ylabel('Following count')

    # フォロー比率のヒストグラムをプロットするのだ
    merged_data['follow_ratio'] = merged_data['followers_count'] / merged_data['following_count']
    
    ax[1,1].hist(merged_data['follow_ratio'], bins=np.logspace(np.log10(10**-7), np.log10(10**7), num=15))
    ax[1,1].set_xscale('log')
    ax[1,1].set_title('Follow ratio distribution follower / following')
    ax[1,1].set_xlabel('Follow ratio')
    ax[1,1].set_ylabel('Frequency')


    # 保存するのだ
    plt.tight_layout()
    plt.savefig(f'plot_follow_user_distribution.png')

    # 統計情報をCSVファイルとして保存するのだ
    stats = merged_data[['followers_count', 'following_count', 'follow_ratio']].describe()
    stats.to_csv('plot_follow_user_distribution.csv', index=True)

if __name__ == "__main__":
    plot_distribution()

