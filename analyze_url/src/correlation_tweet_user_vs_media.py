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
from scipy import stats

def load_data(tweet_dir, df_class):
    # データをロードして結合する
    anime_tweet_list = []

    for file_name in os.listdir(tweet_dir):
        if file_name.endswith('.jsonl'):
            id = os.path.splitext(file_name)[0]
            tweet_file_path = os.path.join(tweet_dir, file_name)
            df = pd.read_json(tweet_file_path, lines=True, dtype={'tweet_id': str})
            title = df_class.loc[id, 'title']
            mean_tweet_user_count = df_class.loc[id, 'mean_tweet_user_count']

            tweet_count = len(df)

            # mediaを含むツイートの数をカウント
            media_count = df['media'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
            
            # mediaでphotoを含むツイートの数をカウント
            photo_count = df['media'].apply(lambda x: any(item['type'] == 'photo' for item in x) if isinstance(x, list) else False).sum()

            # mediaでphotoを含むツイートの数をカウント
            video_count = df['media'].apply(lambda x: any(item['type'] == 'video' for item in x) if isinstance(x, list) else False).sum()

            # urlを含むツイートの数をカウント
            url_count = df['urls'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
            
            anime_tweet_data = {
                'id': id,
                'title': title,
                'mean_tweet_user_count': mean_tweet_user_count,
                'tweet_count': tweet_count,
                'media_count': media_count,
                'photo_count': photo_count,
                'video_count': video_count,
                'url_count': url_count
            }
            
            anime_tweet_list.append(anime_tweet_data)

    return pd.DataFrame(anime_tweet_list)

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
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs)

    # クラスタごとにデータを分割
    cluster_groups = combined_df.groupby('weekly_cluster')
    print_correlation(cluster_groups)

    cluster_groups = combined_df.groupby('mean_cluster')
    print_correlation(cluster_groups)

def print_correlation(cluster_groups):
    # 各クラスタの相関係数を計算
    for cluster, group in cluster_groups:
        corr_pos = group['positive'].corr(group['next_week_user'])
        corr_neu = group['neutral'].corr(group['next_week_user'])
        corr_neg = group['negative'].corr(group['next_week_user'])
        print(f"クラスタ {cluster} - ポジティブ: {corr_pos}, ニュートラル: {corr_neu}, ネガティブ: {corr_neg}")

def calculate_overall_correlation(emo_and_user_set):
    # 各データフレームに対して次週ユーザ数の計算を行い、それらを結合
    all_dfs = []

    for id, df in emo_and_user_set.items():
        df['next_week_user'] = df['tweet_users_count'].shift(-1)
        all_dfs.append(df)
    print(df)

    combined_df = pd.concat(all_dfs)

    # 相関を計算
    corr_pos = combined_df['positive_ratio'].corr(combined_df['next_week_user'])
    corr_neu = combined_df['neutral_ratio'].corr(combined_df['next_week_user'])
    corr_neg = combined_df['negative_ratio'].corr(combined_df['next_week_user'])

    print(f"全体の相関係数 - ポジティブ: {corr_pos}, ニュートラル: {corr_neu}, ネガティブ: {corr_neg}")

def plot_tweet_data(df, y_key):
    # 相関係数の計算
    correlation, p_value = stats.pearsonr(df['mean_tweet_user_count'], df[y_key])

    # ツイート数と指定されたキーの関係をプロット
    plt.figure(figsize=(10, 10))
    plt.scatter(df['mean_tweet_user_count'], df[y_key], alpha=0.5, s=100)
    """
    for i, txt in enumerate(df['title']):
        plt.annotate(txt, (df['mean_tweet_user_count'][i], df[y_key][i]))
    """
    plt.xlabel('平均週間ツイートユーザ数', fontsize=24)
    plt.ylabel(y_key, fontsize=24)
    plt.title(f'相関係数: {correlation:.2f}, p値: {p_value:.4f}', fontsize=24)
    plt.grid(True)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.savefig(f'results/correlation_tweet_user_vs_{y_key}.png')
    plt.close()

def main():
    tweet_dir = '/work/n213304/learn/anime_retweet_2/analyze_url/tweet_url'

    df_class = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col="id")

    df = load_data(tweet_dir, df_class)
    # df = pd.read_csv('results/anime_tweet_info.csv')
    plot_tweet_data(df, 'media_count')
    plot_tweet_data(df, 'url_count')
    plot_tweet_data(df, 'photo_count')
    plot_tweet_data(df, 'video_count')
    df.to_csv('results/anime_tweet_info.csv')
    
if __name__ == "__main__":
    main()
