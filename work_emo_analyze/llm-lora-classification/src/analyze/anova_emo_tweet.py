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
from statsmodels.formula.api import ols
from scipy import stats

def load_data(tweet_emo_dir, tweet_user_dir):
    # データをロードして返す
    data_list = []
    for file_name in os.listdir(tweet_emo_dir):
        if file_name.endswith('.csv'):
            id = os.path.splitext(file_name)[0]
            emo_file_path = os.path.join(tweet_emo_dir, file_name)
            user_file_path = os.path.join(tweet_user_dir, id+"_1_week_tweet_counts.csv")
            data_list.append((id, emo_file_path, user_file_path))
    return data_list

def calculate_emotion_data(id, emo_file_path, user_file_path, df_class, df_title):
    df = pd.read_csv(emo_file_path)
    df_tweet_user = pd.read_csv(user_file_path)

    mean_weekly_tweet_user = df_tweet_user["tweet_users_count"].mean()

    # 各感情の合計ツイート数を計算
    total_positive = df['positive'].sum()
    total_neutral = df['neutral'].sum()
    total_negative = df['negative'].sum()

    # 全ツイート数の合計を計算
    total_tweet_counts = df['tweet_count'].sum()

    weekly_tweet_user_cluster_name = {0: "上昇", 1: "下降", 2: "山型", 3: "横ばい"}
    mean_tweet_user_cluster_name = {0: "多い", 1: "少ない", 2: "中くらい"}
    weekly_tweet_user_clusters = weekly_tweet_user_cluster_name[df_class.loc[id, 'weekly_tweet_user_clusters']]
    mean_tweet_user_clusters = mean_tweet_user_cluster_name[df_class.loc[id, 'mean_tweet_user_clusters']]

    emotion_data = {
        'id': id,
        'title': df_title.loc[id, '作品名'],
        'ネガティブ_radio': total_negative / total_tweet_counts,
        'ニュートラル_radio': total_neutral / total_tweet_counts,
        'ポジティブ_radio': total_positive / total_tweet_counts,
        'ネガティブ': total_negative,
        'ニュートラル': total_neutral,
        'ポジティブ': total_positive,
        'mean_weekly_tweet_user': mean_weekly_tweet_user,
        'weekly_tweet_user_clusters': weekly_tweet_user_clusters,
        'mean_tweet_user_clusters': mean_tweet_user_clusters,
        'clusters': weekly_tweet_user_clusters + '-' + mean_tweet_user_clusters
    }

    return emotion_data

import itertools

def perform_pairwise_ttests(df, cluster_column, value_column_list):
    # クラスタのユニークな値を取得
    clusters = df[cluster_column].unique()

    # すべてのクラスタペアの組み合わせを生成
    cluster_pairs = list(itertools.combinations(clusters, 2))

    for value_column in value_column_list:
        results = []
        for cluster1, cluster2 in cluster_pairs:
            # 各クラスタのデータを抽出
            data1 = df[df[cluster_column] == cluster1][value_column]
            data2 = df[df[cluster_column] == cluster2][value_column]

            # t検定の実行
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)  # 等分散を仮定しない

            # 結果を記録
            results.append({
                'Cluster Pair': (cluster1, cluster2),
                't-statistic': t_stat,
                'p-value': p_value
            })
        print(value_column)
        print(pd.DataFrame(results))

def perform_anova(df):
    # ANOVAを実施する関数
    for column in ['ポジティブ_radio', 'ニュートラル_radio', 'ネガティブ_radio']:
        cluster0 = df[df['weekly_tweet_user_clusters'] == 0][column]
        cluster1 = df[df['weekly_tweet_user_clusters'] == 1][column]
        cluster2 = df[df['weekly_tweet_user_clusters'] == 2][column]
        cluster3 = df[df['weekly_tweet_user_clusters'] == 3][column]

        f_stat, p_value = stats.f_oneway(cluster0, cluster1, cluster2, cluster3)
        print("column:", column)
        print("F統計量:", f_stat)
        print("p値:", p_value)

    for column in ['ポジティブ_radio', 'ニュートラル_radio', 'ネガティブ_radio']:
        cluster0 = df[df['mean_tweet_user_clusters'] == 0][column]
        cluster1 = df[df['mean_tweet_user_clusters'] == 1][column]
        cluster2 = df[df['mean_tweet_user_clusters'] == 2][column]

        f_stat, p_value = stats.f_oneway(cluster0, cluster1, cluster2, cluster3)
        print("column:", column)
        print("F統計量:", f_stat)
        print("p値:", p_value)
        

def main():
    tweet_emo_dir = 'tweet_emo_weekly'
    tweet_user_dir = '/work/n213304/learn/anime_retweet_2/count_tweet'
    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)
    df_class = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col="id")

    data_list = load_data(tweet_emo_dir, tweet_user_dir)
    emotion_data_list = [calculate_emotion_data(id, emo_path, user_path, df_class, df_title) for id, emo_path, user_path in data_list]

    df = pd.DataFrame(emotion_data_list)
    
    # ANOVAの実施
    anova_results = perform_anova(df)

    value_column_list = ['ポジティブ_radio', 'ニュートラル_radio', 'ネガティブ_radio']
    perform_pairwise_ttests(df, 'weekly_tweet_user_clusters', value_column_list)
    perform_pairwise_ttests(df, 'mean_tweet_user_clusters', value_column_list)
    # perform_pairwise_ttests(df, 'clusters', value_column_list)

if __name__ == "__main__":
    main()

