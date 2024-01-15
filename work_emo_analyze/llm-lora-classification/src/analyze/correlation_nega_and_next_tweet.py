import os
import pandas as pd
import numpy as np

def calculate_correlations(tweet_emo_dir):
    correlations_list = []
    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)
    path = "/work/n213304/learn/anime_retweet_2/anime_class.csv"
    df_class = pd.read_csv(path, index_col="id")
    
    mean_tweet_user_cluster_name = {0: "多い", 1: "少ない", 2: "中くらい"}
    weekly_tweet_user_cluster_name = {0: "上昇", 1: "下降", 2: "U型(横ばい)", 3: "W型(山型)"}

    for file_name in os.listdir(tweet_emo_dir):
        if file_name.endswith('.csv'):
            id = os.path.splitext(file_name)[0]
            file_path = os.path.join(tweet_emo_dir, file_name)
            df = pd.read_csv(file_path)

            # ネガティブツイートの量と次週の全ツイートの量の相関を計算
            negative_counts = df['negative'][:-1]
            next_week_tweet_counts = df['tweet_count'][1:]
            correlation = np.corrcoef(negative_counts, next_week_tweet_counts)[0, 1]
            
            anime_id = os.path.splitext(file_name)[0]
            weekly_tweet_user_clusters = df_class.loc[id, 'weekly_tweet_user_clusters']
            mean_tweet_user_clusters = df_class.loc[id, 'mean_tweet_user_clusters']
            data = {
                'id': id,
                'title': df_title.loc[id, '作品名'],
                'negative_correlation': correlation,
                'weekly_tweet_user_clusters': weekly_tweet_user_cluster_name[weekly_tweet_user_clusters],
                'mean_tweet_user_clusters': mean_tweet_user_cluster_name[mean_tweet_user_clusters]
            }
            correlations_list.append(data)

    return correlations_list

def main():

    tweet_emo_dir = 'tweet_emo_weekly'  # このディレクトリ名は状況に応じて調整してください
    correlations_list = calculate_correlations(tweet_emo_dir)

    df_correlations = pd.DataFrame(correlations_list)
    sorted_df = df_correlations.sort_values(by=['weekly_tweet_user_clusters', 'mean_tweet_user_clusters'])

    # CSVとして出力
    sorted_df.to_csv("./src/analyze/correlation_nega_and_next_tweet.csv", index=False)

if __name__ == "__main__":
    main()
