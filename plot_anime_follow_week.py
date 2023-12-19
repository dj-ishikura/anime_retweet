import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('./anime_data_updated.csv', index_col=0)

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']

    return title

def plot_follow_week(jsonl_path, output_png):
    follower_data_path = "tweet_user_follower_number.jsonl"

    # データを読み込むのだ
    weekly_data = pd.read_json(jsonl_path, lines=True)
    
    # 週ごとのデータを集めるのだ
    weekly_followers_counts = []
    weekly_following_counts = []
    weekly_follow_ratio = []


    # フォロワー/フォロー数データを読み込むのだ
    follower_data = pd.read_json(follower_data_path, lines=True)
    
    for index, row in weekly_data.iterrows():
        user_ids_set = set(row['user_ids'])

        # ユーザIDセットを利用してマージするデータをフィルタリングするのだ
        merged_data = follower_data[follower_data['user_id'].isin(user_ids_set)]

        # 週ごとのデータを保存するのだ
        weekly_followers_counts.append(merged_data['followers_count'])
        weekly_following_counts.append(merged_data['following_count'])
        weekly_follow_ratio.append(merged_data['followers_count'] / merged_data['following_count'])

    
    id = os.path.basename(jsonl_path).replace('.jsonl', '')
    title = get_info_from_csv(id)

    # フィギュアを初期化するのだ
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # フィギュア全体のタイトルを設定するのだ
    fig.suptitle(f'{title}\n{id}', fontsize=16)

    # 箱ひげ図をプロットするのだ
    ax[0].boxplot(weekly_followers_counts, labels=[row['date'] for index, row in weekly_data.iterrows()])
    ax[0].set_title('Weekly Follower Count Distribution')
    ax[0].set_xlabel('Week')
    ax[0].set_ylabel('Follower Count')
    ax[0].set_yscale('log')

    ax[1].boxplot(weekly_following_counts, labels=[row['date'] for index, row in weekly_data.iterrows()])
    ax[1].set_title('Weekly Following Count Distribution')
    ax[1].set_xlabel('Week')
    ax[1].set_ylabel('Following Count')
    ax[1].set_yscale('log')

    ax[2].boxplot(weekly_follow_ratio, labels=[row['date'] for index, row in weekly_data.iterrows()])
    ax[2].set_title('Weekly Follow Ratio Distribution')
    ax[2].set_xlabel('Week')
    ax[2].set_ylabel('Follow Ratio')
    ax[2].set_yscale('log')
    ax[2].axhline(y=1, color='r', linestyle='--')  # この行を追加するのだ

    # 保存するのだ
    plt.tight_layout()
    plt.savefig(output_png)
    
if __name__ == "__main__":
    plot_follow_week(sys.argv[1], sys.argv[2])
