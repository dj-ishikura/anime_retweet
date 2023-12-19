import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
import numpy as np
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

def plot_scatter_merged(df_merged):
    plt.figure(figsize=(8, 8))

    # クラスと色の対応
    color_dict = {'trend': 'red', 'hit': 'green', 'miner': 'blue'}
    columns_to_plot = [['tweet_mean', 'retweet_mean'], 
                       ['user_mean_tweet', 'user_mean_retweet'], 
                       ['ratio_mean_tweet', 'ratio_mean_retweet']]
    labels = ['Mean', 'User Mean', 'Ratio Mean']
    
    for i, cols in enumerate(columns_to_plot):
        for class_name in ['miner', 'hit', 'trend']:
            subset = df_merged[df_merged['class'] == class_name]
            plt.subplot(2, 2, i+1)
            plt.scatter(subset[cols[0]], subset[cols[1]], 
            label=class_name, color=color_dict[class_name], alpha=0.4)
        plt.xlabel('Tweet')
        plt.ylabel('Retweet')
        plt.title(f'Tweet vs Retweet {labels[i]}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('result/tweet_vs_retweet_scatter.png')

def plot_scatter_merged_ratio(df_merged):
    plt.figure(figsize=(8, 8))

    # クラスと色の対応
    color_dict = {'trend': 'red', 'hit': 'green', 'miner': 'blue'}
    columns_to_plot = [['ratio_mean_tweet', 'tweet_mean'], 
                    ['ratio_mean_tweet', 'user_mean_tweet'], 
                    ['ratio_mean_retweet', 'retweet_mean'],
                    ['ratio_mean_retweet', 'user_mean_retweet']]
    
    for i, cols in enumerate(columns_to_plot):
        for class_name in ['miner', 'hit', 'trend']:
            subset = df_merged[df_merged['class'] == class_name]
            plt.subplot(2, 2, i+1)
            plt.scatter(subset[cols[0]], subset[cols[1]], 
            label=class_name, color=color_dict[class_name], alpha=0.4)
        plt.xlabel('Ratio')
        plt.ylabel(cols[0].split('_')[-1])
        plt.title(f'{cols[0]} vs {cols[1]}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('result/tweet_vs_ratio_scatter.png')

def calculate_averages(directory, mode):
    file_names = [f for f in os.listdir(directory) if f.endswith(f'1_week_{mode}_counts.csv')]
    data = []

    for file_name in file_names:
        df = pd.read_csv(os.path.join(directory, file_name))
        tweet_mean = df[f'{mode}_count'].mean()  # 列名を動的に更新するのだ
        user_mean = df[f'{mode}_users_count'].mean()  # 新しい列名を動的に更新するのだ
        ratio_mean = df[f'{mode}_ratio'].mean()  # 新しい列名を動的に更新するのだ
        data.append({'id': file_name.split('_')[0], f'{mode}_mean': tweet_mean, 'user_mean': user_mean, 'ratio_mean': ratio_mean})

    return pd.DataFrame(data)

def plot_averages(df, mode):

    # クラスと色の対応
    color_dict = {'trend': 'red', 'hit': 'green', 'miner': 'blue'}
    columns_to_plot = [f'{mode}_mean', 'user_mean', 'ratio_mean']
    titles = [f'{mode} Mean', 'User Mean', 'Ratio Mean']
    anime_class_df = pd.read_csv('result/class_anime_list.csv')
    df = pd.merge(anime_class_df, df, on='id')

    for i, col in enumerate(columns_to_plot):
        plt.figure(figsize=(10, 5))
        for class_name in ['miner', 'hit', 'trend']:
            subset = df[df['class'] == class_name]
            labels = subset['id']
            
            # 散布図の作成
            plt.subplot(1, 2, 1)
            plt.scatter(labels, subset[col], label=class_name, 
            color=color_dict[class_name], alpha=0.4)
            '''
            for j, label in enumerate(labels):
                plt.text(labels[j], df.loc[j, col], str(label))
            '''
        plt.xlabel('File ID')
        plt.ylabel('Average Value')
        plt.title(f'{titles[i]} per File')
        plt.legend()

        # ヒストグラムの作成
        plt.subplot(1, 2, 2)
        plt.hist(df[col], bins=20, alpha=0.7, edgecolor='black')
        sorted_averages = sorted(df[col])
        median = np.percentile(sorted_averages, 50) # len(averages)の80%に相当する値
        plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label=f'中央値:{median}')
        
        upper = np.percentile(sorted_averages, 90) # len(averages)の80%に相当する値
        plt.axvline(upper, color='g', linestyle='dashed', linewidth=2, label=f'90パーセントタイル:{upper:.3f}')
        # plt.axvline(500, color='b', linestyle='dashed', linewidth=2, label='しきい値')

        plt.minorticks_on()
        plt.xlabel('Average Value')
        plt.ylabel('Frequency')
        plt.title(f'{titles[i]} Histogram')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'result/{mode}_{col}.png')

        print(col)
        for i in range(0, 101, 10):
            percentile_value = np.percentile(sorted_averages, i)
            print(f"{i}% percentile: {percentile_value}")

def calculate_average_and_plot(directory, threshold, mode):
    directory = f'{directory}_{mode}'  # ディレクトリ名を動的に更新するのだ
    df_csv = calculate_averages(directory, mode)  # modeを関数に渡すのだ
    df_csv.to_csv(f'result/{mode}_mean.csv', index=False)  # CSVファイル名を動的に更新するのだ
    plot_averages(df_csv, mode)
    return df_csv

# Replace 'your_directory' with the path to your directory
mode = 'tweet'  # 'tweet' または 'retweet' を指定するのだ
# df_retweet = calculate_average_and_plot('count', 500, 'retweet')  # modeを関数に渡すのだ
df_tweet = calculate_average_and_plot('count', 500, 'tweet')  # modeを関数に渡すのだ

# データフレームのマージ
df_merged = pd.merge(df_tweet, df_retweet, on='id', suffixes=('_tweet', '_retweet'))
anime_class_df = pd.read_csv('result/class_anime_list.csv')
df_merged = pd.merge(anime_class_df, df_merged, on='id')
print(df_merged.head())
# 散布図のプロット
plot_scatter_merged(df_merged)
plot_scatter_merged_ratio(df_merged)

