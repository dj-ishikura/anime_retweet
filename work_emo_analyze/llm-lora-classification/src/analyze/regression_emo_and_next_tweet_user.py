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

def load_data(tweet_emo_dir, tweet_user_dir):
    # データをロードして結合する
    emo_and_user_set = {}

    for file_name in os.listdir(tweet_emo_dir):
        if file_name.endswith('.csv'):
            id = os.path.splitext(file_name)[0]

            # 感情データとユーザ数データのパスを取得
            emo_file_path = os.path.join(tweet_emo_dir, file_name)
            user_file_path = os.path.join(tweet_user_dir, id + "_1_week_tweet_counts.csv")

            # データを読み込む
            emo_df = pd.read_csv(emo_file_path)
            emo_df['positive_ratio'] = emo_df['positive'] / emo_df['tweet_count']
            emo_df['neutral_ratio'] = emo_df['neutral'] / emo_df['tweet_count']
            emo_df['negative_ratio'] = emo_df['negative'] / emo_df['tweet_count']
            user_df = pd.read_csv(user_file_path)

            # 列名を変更して重複を解消
            emo_df.rename(columns={'tweet_count': 'emo_tweet_count'}, inplace=True)
            user_df.rename(columns={'tweet_count': 'user_tweet_count'}, inplace=True)

            # 'date' 列を基にデータを結合
            merged_df = pd.merge(emo_df, user_df, on='date', how='inner')

            emo_and_user_set[id] = merged_df

    return emo_and_user_set

def perform_full_regression(emo_and_user_set):
    # 全データフレームの結合
    combined_df = pd.concat(emo_and_user_set.values())

    # 次週ユーザー数を計算
    combined_df['next_week_user'] = combined_df['user_tweet_count'].shift(-1)

    # 独立変数と従属変数を定義
    X = combined_df[['positive', 'neutral', 'negative']]  # 独立変数
    y = combined_df['next_week_user']  # 従属変数

    # 定数項（切片）を追加
    X = sm.add_constant(X)

    # OLSモデルの適用
    model = sm.OLS(y, X, missing='drop').fit()

    # 結果の表示
    print(model.summary())

def calculate_cluster_correlation(emo_and_user_set, df_class):
    all_dfs = []
    weekly_tweet_user_cluster_name = {0: "上昇", 1: "下降", 2: "U型(横ばい)", 3: "W型(山型)"}
    mean_tweet_user_cluster_name = {0: "多い", 1: "少ない", 2: "中くらい"}

    for id, df in emo_and_user_set.items():
        df['next_week_user'] = df['user_tweet_count'].shift(-1)
        df['weekly_cluster'] = df_class.loc[id, 'weekly_tweet_user_clusters']
        df['weekly_cluster'] = df['weekly_cluster'].apply(lambda x: weekly_tweet_user_cluster_name[x])
    
        df['mean_cluster'] = df_class.loc[id, 'mean_tweet_user_clusters']
        df['mean_cluster'] = df['mean_cluster'].apply(lambda x: mean_tweet_user_cluster_name[x])

        df['cluster'] = df['weekly_cluster'] + ' - ' + df['mean_cluster']
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs)

    # クラスタごとにデータを分割し、重回帰分析を実行
    cluster_groups = combined_df.groupby('weekly_cluster')
    perform_regression(cluster_groups)

    cluster_groups = combined_df.groupby('mean_cluster')
    perform_regression(cluster_groups)

    cluster_groups = combined_df.groupby('cluster')
    perform_regression(cluster_groups)

def perform_regression(cluster_groups):
    results = []

    for cluster, group in cluster_groups:
        X = group[['positive', 'neutral', 'negative']]  # 独立変数
        y = group['next_week_user']  # 従属変数
        X = sm.add_constant(X)  # 定数項（切片）を追加
        model = sm.OLS(y, X, missing='drop').fit()  # OLSモデルの適用

        # 重要な統計量を抽出
        result = {
            'Cluster': cluster,
            'R-squared': model.rsquared,
            'Adj. R-squared': model.rsquared_adj,
            'Positive Coef': model.params['positive'],
            'Neutral Coef': model.params['neutral'],
            'Negative Coef': model.params['negative'],
            'Positive P>|t|': model.pvalues['positive'],
            'Neutral P>|t|': model.pvalues['neutral'],
            'Negative P>|t|': model.pvalues['negative'],
        }
        results.append(result)

    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    print(results_df)


def main():
    tweet_emo_dir = 'tweet_emo_weekly'
    tweet_user_dir = '/work/n213304/learn/anime_retweet_2/count_tweet'
    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)
    df_class = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col="id")

    emo_and_user_set = load_data(tweet_emo_dir, tweet_user_dir)
    perform_full_regression(emo_and_user_set)
    calculate_cluster_correlation(emo_and_user_set, df_class)

if __name__ == "__main__":
    main()
