import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

def load_data_from_directory(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.jsonl'):
            with open(os.path.join(directory_path, filename), 'r') as file:
                data_dict = json.load(file)
                data.append(data_dict)
    return pd.DataFrame(data)

def plot_scatter(merged_df):
    plt.figure()
    plt.scatter(merged_df['followers_count'], merged_df['total_tweet_count'], label='フォロワー数 vs ツイート数')
    plt.xlabel('フォロワー数')
    plt.ylabel('ツイート数')
    plt.title('フォロワー数 vs ツイート数')
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(rotation=30)
    plt.savefig('plot_user_anime_tweet_count_follower.png')

    # フォロー数とツイート数の関係を示す散布図
    plt.figure()
    plt.scatter(merged_df['following_count'], merged_df['total_tweet_count'], label='フォロー数 vs ツイート数', color='red')
    plt.xlabel('フォロー数')
    plt.ylabel('ツイート数')
    plt.title('フォロー数 vs ツイート数')
    plt.ticklabel_format(style='plain', axis='x')
    plt.savefig('plot_user_anime_tweet_count_following.png')

    plt.figure()
    plt.scatter(merged_df['follow_ratio'], merged_df['total_tweet_count'], label='フォロー比 vs ツイート数', color='red')
    plt.xlabel('フォロー比')
    plt.ylabel('ツイート数')
    plt.title('フォロー比 vs ツイート数')
    plt.ticklabel_format(style='plain', axis='x')

    plt.savefig('plot_user_anime_tweet_count_follow_ratio.png')

if __name__ == "__main__":
    tweet_file = "user_anime_tweet_count.jsonl"
    df_tweet = pd.read_json(tweet_file, lines=True)    
    df_follow = pd.read_json("tweet_user_follower_number.jsonl", lines=True)
    merged_df = pd.merge(df_tweet, df_follow, on='user_id')
    merged_df['follow_ratio'] = merged_df['followers_count'] / merged_df['following_count']
    print(merged_df[merged_df['total_tweet_count'] > 50000])
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print("mean")
    print(merged_df.mean())
    print("median")
    print(merged_df.median())
    # merged_df = merged_df[merged_df['total_tweet_count'] < 400000]
    
    plot_scatter(merged_df)
