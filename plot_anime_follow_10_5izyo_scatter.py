import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data_from_directory(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.jsonl'):
            with open(os.path.join(directory_path, filename), 'r') as file:
                data_dict = json.load(file)
                data.append(data_dict)
    return pd.DataFrame(data)

def plot_scatter(data_df, anime_class_df):
    color_dict = {'miner': 'blue', 'hit': 'green', 'trend': 'red'}
    
    # アニメのIDを文字列に変換して、データフレームをマージするのだ
    anime_class_df['id'] = anime_class_df['id'].astype(str)
    data_df['id'] = data_df['id'].str.replace('.jsonl', '')
    merged_df = pd.merge(data_df, anime_class_df, on='id', how='left')
    
    plt.figure()
    stats_data = []
    for anime_class, color in color_dict.items():
        class_data = merged_df[merged_df['class'] == anime_class]

        plt.subplot(2, 2, 1)
        plt.scatter(class_data['total_users'], class_data['followers_1000000_or_less'], label=anime_class, color=color)
        plt.xlabel('Total Users')
        plt.ylabel('Followers 1000000 or Less')
        
        plt.subplot(2, 2, 2)
        plt.scatter(class_data['total_users'], class_data['following_100000_or_less'], label=anime_class, color=color)
        plt.xlabel('Total Users')
        plt.ylabel('Following 100000 or Less')

        plt.subplot(2, 2, 3)
        plt.scatter(class_data['total_users'], class_data['followers_10000000_or_less'], label=anime_class, color=color)
        plt.xlabel('Total Users')
        plt.ylabel('Followers 10000000 or Less')
        
        plt.subplot(2, 2, 4)
        plt.scatter(class_data['total_users'], class_data['following_1000000_or_less'], label=anime_class, color=color)
        plt.xlabel('Total Users')
        plt.ylabel('Following 1000000 or Less')


    plt.tight_layout()
    plt.legend()
    plt.savefig('plot_anime_follow_10_5_izyo_scatter.png')
    plt.show()

if __name__ == "__main__":
    directory_path = 'anime_follow_10ika_2' # ディレクトリパスを指定するのだ
    data_df = load_data_from_directory(directory_path)
    
    anime_class_df = pd.read_csv('result/class_anime_list.csv') # クラスデータをロードするのだ
    plot_scatter(data_df, anime_class_df)
