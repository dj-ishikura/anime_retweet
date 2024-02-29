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
from scipy.stats import linregress

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

def calculate_overall_correlation(url_and_user_set, df_class):
    all_dfs = []
    
    weekly_tweet_user_cluster_name = {0: "上昇", 1: "下降", 2: "山型", 3: "横ばい"}

    for id, df in url_and_user_set.items():
        # 次週のユーザ数を計算
        df['next_week_user_count'] = df['tweet_users_count'].shift(-1)
        df['weekly_cluster'] = df_class.loc[id, 'weekly_tweet_user_clusters']
        df['weekly_cluster'] = df['weekly_cluster'].apply(lambda x: weekly_tweet_user_cluster_name[x])
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs)

    # 相関とp値を計算
    temp_df = pd.DataFrame({
        'this_week_user_count': combined_df['tweet_users_count'],
        'next_week_user_count': combined_df['next_week_user_count'],
        'weekly_cluster': combined_df['weekly_cluster']
    }).dropna()
    
    # temp_df = temp_df[(temp_df['this_week_user_count'] > 1000) | (temp_df['next_week_user_count'] > 1000)]

    corr, p_value = pearsonr(temp_df['this_week_user_count'], temp_df['next_week_user_count'])

    # 線形回帰分析
    X = sm.add_constant(temp_df['this_week_user_count'])  # 定数項（切片）を追加
    y = temp_df['next_week_user_count']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # 結果の出力
    print(f"週間ツイートユーザ数の相関係数: {corr:.2f}, p値: {p_value:.2e}")
    print(f"傾き（回帰係数）: {model.params['this_week_user_count']}, 切片: {model.params['const']}")

    cluster_colors = {
        '上昇': 'red',   # 上昇
        '下降': 'blue',  # 下降
        '山型': 'green', # W型(山型)
        '横ばい': 'purple' # U型(横ばい)
    }
    
    # ユニークなクラスタIDを取得
    unique_clusters = temp_df['weekly_cluster'].unique()

    # グラフ生成 (オプショナル)
    plt.figure(figsize=(10, 10))
    # クラスタごとにプロット
    for cluster in unique_clusters:
        # 散布図のプロット
        cluster_data = temp_df[temp_df['weekly_cluster'] == cluster]
        plt.scatter(cluster_data['this_week_user_count'], cluster_data['next_week_user_count'], 
                    alpha=0.5, color=cluster_colors[cluster], label=f'Cluster {cluster}')

        # 回帰直線のプロット
        slope, intercept, r_value, p_value, std_err = linregress(cluster_data['this_week_user_count'], cluster_data['next_week_user_count'])
        x = pd.np.linspace(cluster_data['this_week_user_count'].min(), cluster_data['this_week_user_count'].max(), 100)
        y = slope * x + intercept
        # plt.plot(x, y, color=cluster_colors[cluster], linewidth=2)
        print(f'クラスタ {cluster}: 傾き = {slope:.2f}, 切片 = {intercept:.2f}, 決定係数 = {r_value**2:.2f}')

    plt.legend()
    # 点線でy = xをプロット
    x_values = range(int(min(temp_df['this_week_user_count'].min(), temp_df['next_week_user_count'].min())),
                 int(max(temp_df['this_week_user_count'].max(), temp_df['next_week_user_count'].max())) + 1)
    y_values = x_values
    plt.plot(x_values, y_values, '--', color='gray')
    plt.title(f"週間ツイートユーザ数の相関\n相関係数: {corr:.2f}, p値: {p_value:.2e}")
    plt.xlabel('ある週の週間ツイートユーザ数')
    plt.ylabel('次週の週間ツイートユーザ数')
    plt.grid(True)
    plt.savefig("results/correlation_this_user_and_next_user.png")
    plt.close()

def main():
    tweet_url_dir = 'tweet_url_weekly'
    tweet_user_dir = '/work/n213304/learn/anime_retweet_2/count_tweet'
    df_class = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col="id")

    url_and_user_set = load_data(tweet_url_dir, tweet_user_dir)
    calculate_overall_correlation(url_and_user_set, df_class)

if __name__ == "__main__":
    main()
