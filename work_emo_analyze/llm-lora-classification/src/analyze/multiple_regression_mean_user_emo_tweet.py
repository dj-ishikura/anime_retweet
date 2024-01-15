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
        'weekly_tweet_user_clusters': df_class.loc[id, 'weekly_tweet_user_clusters']
    }

    return emotion_data

def perform_regression_analysis(df, cluster_col):
    results = {}
    for cluster in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster]

        X = cluster_df[['ネガティブ', 'ニュートラル', 'ポジティブ']]
        X = sm.add_constant(X)
        y = cluster_df['mean_weekly_tweet_user']

        model = sm.OLS(y, X).fit()
        results[cluster] = model.summary()
    return results

def perform_regression_analysis_radio(df, cluster_col):
    results = {}
    for cluster in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster]

        X = cluster_df[['ネガティブ_radio', 'ニュートラル_radio', 'ポジティブ_radio']]
        X = sm.add_constant(X)
        y = cluster_df['mean_weekly_tweet_user']

        model = sm.OLS(y, X).fit()
        results[cluster] = model.summary()
    return results

def main():
    tweet_emo_dir = 'tweet_emo_weekly'
    tweet_user_dir = '/work/n213304/learn/anime_retweet_2/count_tweet'
    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)
    df_class = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col="id")

    data_list = load_data(tweet_emo_dir, tweet_user_dir)
    emotion_data_list = [calculate_emotion_data(id, emo_path, user_path, df_class, df_title) for id, emo_path, user_path in data_list]

    df = pd.DataFrame(emotion_data_list)
    regression_results = perform_regression_analysis(df, 'weekly_tweet_user_clusters')

    for cluster, result in regression_results.items():
        print(f"Cluster {cluster} Regression Results:\n{result}")



if __name__ == "__main__":
    main()