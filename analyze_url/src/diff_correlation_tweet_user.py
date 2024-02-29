import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 日本語対応
import japanize_matplotlib  # 日本語ラベル対応

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr
import statsmodels.api as sm

def load_data(tweet_url_dir, tweet_user_dir):
    # データをロードして結合する
    url_and_user_set = {}

    for file_name in os.listdir(tweet_url_dir):
        if file_name.endswith('.csv'):
            id = os.path.splitext(file_name)[0]

            # 感情データとユーザ数データのパスを取得
            url_file_path = os.path.join(tweet_url_dir, file_name)
            user_file_path = os.path.join(tweet_user_dir, id + "_1_week_tweet_counts.csv")

            # データを読み込む
            url_df = pd.read_csv(url_file_path)
            user_df = pd.read_csv(user_file_path)

            # 列名を変更して重複を解消
            url_df.rename(columns={'tweet_count': 'url_tweet_count'}, inplace=True)
            user_df.rename(columns={'tweet_count': 'user_tweet_count'}, inplace=True)

            # 'date' 列を基にデータを結合
            merged_df = pd.merge(url_df, user_df, on='date', how='inner')

            url_and_user_set[id] = merged_df

    return url_and_user_set

def calculate_cluster_correlation(url_and_user_set, df_class):
    all_dfs = []
    
    weekly_tweet_user_cluster_name = {0: "上昇", 1: "下降", 2: "山型", 3: "横ばい"}

    for id, df in url_and_user_set.items():
        df['next_week_user'] = df['tweet_users_count'].shift(-1)
        df['weekly_cluster'] = df_class.loc[id, 'weekly_tweet_user_clusters']
        df['weekly_cluster'] = df['weekly_cluster'].apply(lambda x: weekly_tweet_user_cluster_name[x])

        all_dfs.append(df)

    combined_df = pd.concat(all_dfs)

    # クラスタごとにデータを分割
    cluster_groups = combined_df.groupby('weekly_cluster')
    print_correlation(cluster_groups)

def print_correlation(cluster_groups):
    # 各クラスタの相関係数を計算
    variables = ['media_count', 'photo_count', 'video_count', 'url_count', 'pixiv_count', 'youtube_count']
    
    
    for cluster, group in cluster_groups:
        correlations = {}
        for variable in variables:
            temp_df = pd.DataFrame({
                'variable': group[variable],
                'next_week_user': group['next_week_user']
            }).dropna()

            # temp_dfから相関とp値を計算
            corr, p_value = pearsonr(temp_df['variable'], temp_df['next_week_user'])
            correlations[variable] = (corr, p_value)

        print(cluster)
        for variable, (corr, p_value) in correlations.items():
            print(f"{variable} - Correlation: {corr}, p-value: {p_value}")

def calculate_overall_correlation(url_and_user_set):
    all_dfs = []

    for id, df in url_and_user_set.items():
        # 前週とのユーザ数の増減を計算
        df['prev_week_user_count_change'] = df['tweet_users_count'].diff()
        # 次週とのユーザ数の増減を計算
        df['next_week_user_count_change'] = df['prev_week_user_count_change'].shift(-1)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs)

    # 増減の相関を計算
    temp_df = combined_df[['prev_week_user_count_change', 'next_week_user_count_change']].dropna()
    corr, p_value = pearsonr(temp_df['prev_week_user_count_change'], temp_df['next_week_user_count_change'])

    print(f"前週との増減と次週との増減の相関係数: {corr:.2f}, p値: {p_value:.2e}")

    # 散布図をプロット
    plt.figure(figsize=(10, 6))
    plt.scatter(temp_df['prev_week_user_count_change'], temp_df['next_week_user_count_change'], alpha=0.5)
    plt.title(f"前週と次週のユーザ数の増減の相関\n相関係数: {corr:.2f}, p値: {p_value:.2e}")
    plt.xlabel('前週のユーザ数の増減')
    plt.ylabel('次週のユーザ数の増減')
    plt.grid(True)
    plt.savefig("results/diff_correlation_tweet_user.png")
    plt.show()


def main():
    tweet_url_dir = 'tweet_url_weekly'
    tweet_user_dir = '/work/n213304/learn/anime_retweet_2/count_tweet'
    df_class = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col="id")

    url_and_user_set = load_data(tweet_url_dir, tweet_user_dir)
    calculate_overall_correlation(url_and_user_set)

if __name__ == "__main__":
    main()
