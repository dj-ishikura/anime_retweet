import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties # 日本語対応
import japanize_matplotlib
import numpy as np

def get_info_from_csv(id):
    df = pd.read_csv('./anime_data_updated.csv', index_col=0)
    title = df.loc[id, '作品名']
    return title

def plot_distribution(jsonl_dir):
    follower_data_path = "tweet_user_follower_number.jsonl"
    anime_class_df = pd.read_csv('result/class_anime_list.csv')
    color_dict = {'miner': 'blue', 'hit': 'green', 'trend': 'red'}

    fig_following = plt.figure()
    plt.title('Following count distribution for multiple animes')
    plt.yscale('log')

    fig_follower = plt.figure()
    plt.title('Follower count distribution for multiple animes')
    plt.yscale('log')
    
    fig_follow_ratio = plt.figure()
    plt.title('Follow ratio distribution for multiple animes')
    plt.yscale('log')

    for idx, jsonl_file in enumerate(os.listdir(jsonl_dir)):
        if jsonl_file.endswith('.jsonl'):
            jsonl_path = os.path.join(jsonl_dir, jsonl_file)
            id = os.path.basename(jsonl_path).replace('.jsonl', '')
            #  or anime_class_df.loc[anime_class_df['id'] == id, 'class'].values[0] == 'miner'
            if id not in anime_class_df['id'].values:
                print(id)
                continue

            title = get_info_from_csv(id)
            user_data_df = pd.read_json(jsonl_path, lines=True)
            user_ids_set = set()
            
            for user_ids in user_data_df['user_ids']:
                user_ids_set.update(user_ids)
                
            follower_data = pd.read_json(follower_data_path, lines=True)
            merged_data = follower_data[follower_data['user_id'].isin(user_ids_set)]

            anime_class = anime_class_df.loc[anime_class_df['id'] == id, 'class'].values[0]
            color = color_dict.get(anime_class, 'black') 
            
            plt.figure(fig_following.number)
            plt.boxplot(merged_data['following_count'].dropna(), positions=[idx], labels=[title], boxprops=dict(color=color))
            
            plt.figure(fig_follower.number)
            plt.boxplot(merged_data['followers_count'].dropna(), positions=[idx], labels=[title], boxprops=dict(color=color))

            # フォロー比を計算するのだ
            follow_ratio = merged_data['followers_count'] / merged_data['following_count']
            plt.figure(fig_follow_ratio.number)
            plt.boxplot(follow_ratio.dropna(), positions=[idx], labels=[title], boxprops=dict(color=color))
            
    plt.figure(fig_following.number)
    plt.savefig('plot_anime_following_user_distribution_box.png')

    plt.figure(fig_follower.number)
    plt.savefig('plot_anime_follower_user_distribution_box.png')

    plt.figure(fig_follow_ratio.number)
    plt.savefig('plot_anime_follow_ratio_distribution_box.png')

if __name__ == "__main__":
    plot_distribution("weekly_tweet_users_list")
