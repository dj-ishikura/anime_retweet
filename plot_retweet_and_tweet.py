import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib
import os
import img2pdf


def plot_data(tweet_df, retweet_df, output_png, title, id):

    # データフレームを結合する
    df = pd.concat([tweet_df, retweet_df], axis=1)
    df.columns = ['Tweet Users Count', 'Retweet Users Count']

    # プロットを作成
    fig, ax = plt.subplots()

    df.plot(kind='line', marker='o', ax=ax)

    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index.strftime('%Y-%m-%d'), rotation=45, horizontalalignment='right')

    plt.title(f'{id}\n{title} : Tweet vs Retweet Users Count')
    plt.xlabel('Date')
    plt.ylabel('Users Count')
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_png)

def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('./anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']

    return title

if __name__ == "__main__":
    # ディレクトリを指定します
    tweet_directory = 'count_tweet_2022_2'
    retweet_directory = 'count_retweet_2022'

    # 出力ファイルを指定します
    output_directory = 'retweet_vs_tweet'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_files = []

    # ディレクトリ内のすべてのCSVファイルをループします
    for filename in os.listdir(tweet_directory):
        if filename.endswith('1_week_tweet_counts.csv'):
            # CSVファイルを読み込みます
            id = filename.split('_')[0]
            output_png = os.path.join(output_directory, id + '.png')
            
            df_tweet = pd.read_csv(os.path.join(tweet_directory, id+'_1_week_tweet_counts.csv'), index_col='date', parse_dates=True)
            df_retweet = pd.read_csv(os.path.join(retweet_directory, id+'_1_week_retweet_counts.csv'), index_col='date', parse_dates=True)
            
            title = get_info_from_csv(id)
            plot_data(df_tweet, df_retweet, output_png, title, id)
            image_files.append(output_png)

    output_pdf = 'retweet_vs_tweet.pdf'
    # すべての画像をPDFとして結合
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert([i for i in image_files if i.endswith(".png")]))
    