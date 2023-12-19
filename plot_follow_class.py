import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ファイルの読み込み
anime_class_df = pd.read_csv('result/class_anime_list.csv')
tweet_df = pd.read_csv('result/tweet_mean.csv')

dir_path = "follow_user_distribution"
# 指定ディレクトリ内のすべてのCSVファイルを読み込む

stats = []
for file_name in os.listdir(dir_path):
    if file_name.endswith('.csv'):
        filepath = os.path.join(dir_path, file_name)
        df = pd.read_csv(filepath, index_col=0)
        anime_id = filepath.split('/')[-1].replace(".csv", "")
        
        # 基本的な統計データを計算するのだ
        desc = df.describe()
        
        # 平均と中央値を取り出して、カラム名を動的に設定するのだ
        mean_values = desc.loc['mean'].tolist()
        median_values = desc.loc['50%'].tolist()
        
        stats.append([anime_id] + mean_values + median_values)

# カラム名を動的に設定するのだ
columns = ['id'] + [f'{col}_mean' for col in df.columns] + [f'{col}_median' for col in df.columns]

# 統計データをデータフレームに変換するのだ
stats_df = pd.DataFrame(stats, columns=columns)

# データフレームをマージします
df = pd.merge(anime_class_df, stats_df, on='id')
df = pd.merge(df, tweet_df, on='id')
print(df.columns)

# クラスと色の対応
color_dict = {'miner': 'blue', 'hit': 'green', 'trend': 'red'}

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column, class_column, output_file, color_dict, bin_edges):
    plt.figure(figsize=(10,6))

    for label in color_dict.keys():
        subset = df[df[class_column] == label]
        sns.histplot(data=subset, x=column, bins=bin_edges, 
                     kde=False, label=label, color=color_dict[label], alpha=0.2)

        # クラスごとの平均と標準偏差を求めます
        average = subset[column].mean()
        std_dev = subset[column].std()
        print('{} : average {}, std {}'.format(label, average, std_dev))

        # 平均と標準偏差をプロットします
        plt.axvline(average, color=color_dict[label], linestyle='--')

    # グラフの設定
    plt.title(f'Histogram of {column} by {class_column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend(title=class_column)
    plt.savefig(output_file)

def plot_scatter(df, x_column, y_column, class_column, output_file, color_dict):
    plt.figure(figsize=(10,6))

    # クラスごとに散布図をプロットします
    for label in color_dict.keys():
        subset = df[df[class_column] == label]
        
        # 散布図をプロットします
        plt.scatter(subset[x_column], subset[y_column], label=label, color=color_dict[label], alpha=0.8)
        
        # クラスごとの平均をプロットします
        x_avg = subset[x_column].mean()
        y_avg = subset[y_column].mean()
        
        # plt.axhline(y_avg, color=color_dict[label], linestyle='--')
        # plt.axvline(x_avg, color=color_dict[label], linestyle='--')
        
        # print(f'{label} class - X avg: {x_avg}, Y avg: {y_avg}')

    # グラフの設定
    plt.title(f'Scatter Plot of {y_column} versus {x_column} by {class_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title=class_column)
    plt.savefig(output_file)

# 使用例:
plot_scatter(df, 'followers_count_mean', 'following_count_mean', 'class', 'plot_follow_class_anime_scatter.png', color_dict)
plot_scatter(df, 'followers_count_mean', 'tweet_mean', 'class', 'plot_follower_tweet_class_anime_scatter.png', color_dict)
plot_scatter(df, 'following_count_mean', 'tweet_mean', 'class', 'plot_following_tweet_class_anime_scatter.png', color_dict)
plot_scatter(df, 'follow_ratio_mean', 'tweet_mean', 'class', 'plot_follow_ratio_tweet_class_anime_scatter.png', color_dict)
plot_scatter(df, 'followers_count_mean', 'user_mean', 'class', 'plot_follower_user_class_anime_scatter.png', color_dict)
plot_scatter(df, 'following_count_mean', 'user_mean', 'class', 'plot_following_user_class_anime_scatter.png', color_dict)
plot_scatter(df, 'follow_ratio_mean', 'user_mean', 'class', 'plot_follow_ratio_user_class_anime_scatter.png', color_dict)


df.to_csv("plot_follow_class.csv")
df[df['followers_count_mean'] > 30000].to_csv("plot_follow_class_.csv")