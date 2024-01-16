import pandas as pd
import os
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語対応

def plot_user_tweet_interval(directory, output_png):
    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)

    # アニメごとのツイート間隔データを格納するためのリスト
    intervals_data = []
    title_list = []

    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path)

            # intervalを数値型に変換し、NaN値を除外
            df['average_interval'] = pd.to_numeric(df['average_interval'], errors='coerce').dropna()

            # アニメIDをファイル名から取得
            anime_id = os.path.splitext(file_name)[0]

            # アニメタイトルを取得
            title = df_title.loc[anime_id, '作品名'] if anime_id in df_title.index else '不明'
            title_list.append(title)

            # アニメごとのツイート間隔データをリストに追加
            intervals_data.append(df['average_interval'].tolist())

    # 箱ひげ図のプロット
    plt.figure(figsize=(6, 12))
    plt.boxplot(intervals_data, labels=[title for title in title_list], vert=False)
    plt.title('アニメごとのユーザー平均ツイート間隔')
    plt.xlabel('アニメタイトル')
    plt.ylabel('平均間隔（秒）')
    plt.xticks(rotation=90)  # x軸のラベルを90度回転

    plt.savefig(output_png, bbox_inches='tight')
    plt.close()

# ディレクトリ名と出力ファイル名を指定
user_tweet_interval_anime_dir = 'user_tweet_interval_anime'
output_png = 'results/user_tweet_interval_average_boxplot.png'

# 関数を実行
plot_user_tweet_interval(user_tweet_interval_anime_dir, output_png)
