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
                data.append(json.load(file))
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

        # クラス内の0の数を表示するのだ
        zero_counts = {
            'Class': anime_class,
            'followers_10_or_less == 0': sum(class_data['followers_10_or_less'] == 0),
            'following_10_or_less == 0': sum(class_data['following_10_or_less'] == 0),
            'followers_100_or_less == 0': sum(class_data['followers_100_or_less'] == 0),
            'following_100_or_less == 0': sum(class_data['following_100_or_less'] == 0),
        }
        stats_data.append(zero_counts)

        plt.subplot(2, 2, 1)
        plt.scatter(class_data['total_users'], class_data['followers_10_or_less'], label=anime_class, color=color)
        plt.xlabel('Total Users')
        plt.ylabel('Followers 10 or Less')
        
        # 直線を描くのだ
        # a, b = np.polyfit(class_data['total_users'], class_data['followers_10_or_less'], 1)
        # plt.plot(class_data['total_users'], a * class_data['total_users'] + b, color=color, linestyle='--')

        plt.subplot(2, 2, 2)
        plt.scatter(class_data['total_users'], class_data['following_10_or_less'], label=anime_class, color=color)
        plt.xlabel('Total Users')
        plt.ylabel('Following 10 or Less')

        plt.subplot(2, 2, 3)
        plt.scatter(class_data['total_users'], class_data['followers_100_or_less'], label=anime_class, color=color)
        plt.xlabel('Total Users')
        plt.ylabel('Followers 100 or Less')


        plt.subplot(2, 2, 4)
        plt.scatter(class_data['total_users'], class_data['following_100_or_less'], label=anime_class, color=color)
        plt.xlabel('Total Users')
        plt.ylabel('Following 100 or Less')


    plt.tight_layout()
    plt.legend()
    plt.savefig('plot_anime_follow_10ika_scatter.png')
    plt.show()

    # 統計データをデータフレームに変換するのだ
    stats_df = pd.DataFrame(stats_data)
    
    # データフレームをCSVファイルとして保存するのだ
    stats_df.to_csv('result/zero_counts_stats.csv', index=False)

    correlation_coefficient = data_df['total_users'].corr(data_df['followers_10_or_less'])
    print(correlation_coefficient)
    correlation_coefficient = data_df['total_users'].corr(data_df['following_10_or_less'])
    print(correlation_coefficient)



if __name__ == "__main__":
    directory_path = 'anime_follow_10ika' # ディレクトリパスを指定するのだ
    data_df = load_data_from_directory(directory_path)
    
    anime_class_df = pd.read_csv('result/class_anime_list.csv') # クラスデータをロードするのだ
    plot_scatter(data_df, anime_class_df)
