import pandas as pd
import os
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語対応

def plot_user_tweet_interval(directory, output_png):
    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)

    anime_tweet_interval = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path)

            # 平均間隔を計算
            df['average_interval'] = pd.to_numeric(df['average_interval'], errors='coerce')
            avg_intervals = df['average_interval'].mean() / 60

            total_tweet = df['tweet_count'].sum()

            # アニメIDをファイル名から取得
            anime_id = os.path.splitext(file_name)[0]

            # アニメタイトルを取得
            title = df_title.loc[anime_id, '作品名'] if anime_id in df_title.index else '不明'

            data = {
                'id': anime_id,
                'title': title,
                'avg_intervals': avg_intervals,
                'total_tweet': total_tweet
            }
            anime_tweet_interval.append(data)

    df = pd.DataFrame(anime_tweet_interval)
    df = df.sort_values(by='total_tweet', ascending=True)
    df.dropna(subset=['avg_intervals'], inplace=True)

    fig = plt.figure()
    
    df.plot(kind='barh', x='title', y='avg_intervals', color='skyblue', width=0.5, figsize=(6, 12))
    plt.title('アニメごとのユーザー平均ツイート間隔')
    plt.xlabel('平均間隔（秒）')
    plt.ylabel('アニメタイトル')

    plt.savefig(output_png, bbox_inches='tight')
    plt.close()

# ディレクトリ名と出力ファイル名を指定
user_tweet_interval_anime_dir = 'user_tweet_interval_anime'
output_png = 'results/user_tweet_interval_average.png'

# 関数を実行
plot_user_tweet_interval(user_tweet_interval_anime_dir, output_png)
