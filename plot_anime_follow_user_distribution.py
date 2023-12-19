import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib
import numpy as np

def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('./anime_data_updated.csv', index_col=0)

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']

    return title

def plot_distribution(jsonl_path, output_dir):
    follower_data_path = "tweet_user_follower_number.jsonl"
    
    # データを読み込むのだ
    user_data_df = pd.read_json(jsonl_path, lines=True)
    
    # ユーザIDのセットを作成するのだ
    user_ids_set = set()
    for user_ids in user_data_df['user_ids']:
        user_ids_set.update(user_ids)
    
    # フォロワー/フォロー数データを読み込むのだ
    follower_data = pd.read_json(follower_data_path, lines=True)

    # ユーザIDセットを利用してマージするデータをフィルタリングするのだ
    merged_data = follower_data[follower_data['user_id'].isin(user_ids_set)]

    id = os.path.basename(jsonl_path).replace('.jsonl', '')
    title = get_info_from_csv(id)
    output_prefix = os.path.join(output_dir, id)

    # フィギュアを初期化するのだ
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))
    # フィギュア全体のタイトルを設定するのだ
    fig.suptitle(f'{title}\n{id}', fontsize=16)

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
    ax[1,1].set_title('Follow ratio distribution')
    ax[1,1].set_xlabel('Follow ratio')
    ax[1,1].set_ylabel('Frequency')

    # 保存するのだ
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png')

    # CSVファイルを保存するのだ
    merged_data.to_csv(f'{output_prefix}.csv', index=False)

    # 統計情報をCSVファイルとして保存するのだ
    stats = merged_data[['followers_count', 'following_count', 'follow_ratio']].describe()
    stats.to_csv(f'{output_prefix}.csv', index=True)

if __name__ == "__main__":
    plot_distribution(sys.argv[1], sys.argv[2])
