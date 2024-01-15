import json
import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob
import matplotlib.pyplot as plt
import pytz
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']
    start_date = df.loc[id, '開始日']
    start_date = datetime.strptime(start_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))
    end_date = df.loc[id, '終了日']
    end_date = datetime.strptime(end_date, "%Y年%m月%d日").replace(tzinfo=pytz.timezone('Asia/Tokyo'))
    # end_date = end_date + timedelta(days=7)

    return title, start_date, end_date

def plot_tweet_weekly(df, output_png, title, id):
    # プロットの作成と保存
    df.plot(kind='bar', stacked=True, y=['positive_ratio', 'neutral_ratio', 'negative_ratio'], color=['lightcoral', 'khaki', 'lightblue'])
    plt.title(f'{id}\n{title} : Tweet Emo Radio')
    df['month_day'] = df['date'].str.split('-').str[1] + '-' + df['date'].str.split('-').str[2]
    plt.xticks(ticks=range(len(df['month_day'])), labels=df['month_day'], rotation=45)
    plt.xlabel('放送日', fontsize=14)
    plt.ylabel('週間ツイート数', fontsize=14)
    plt.legend(labels=['ポジティブ', 'ニュートラル', 'ネガティブ'])  # 凡例のラベルを指定
    plt.tight_layout()  # ラベルが画像の外に出ないように調整
    plt.savefig(output_png)
    plt.close()

tweet_emo_dir = 'tweet_emo_weekly'
output_dir = 'tweet_emo_radio'

# 出力ディレクトリの作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ディレクトリ内のすべてのCSVファイルに対して処理を行う
for file_name in os.listdir(tweet_emo_dir):
    if file_name.endswith('.csv'):
        id = os.path.splitext(file_name)[0]
        file_path = os.path.join(tweet_emo_dir, file_name)
        df = pd.read_csv(file_path)
        title, start_date, end_date = get_info_from_csv(id)

        # 割合を計算
        df['positive_ratio'] = df['positive'] / df['tweet_count']
        df['neutral_ratio'] = df['neutral'] / df['tweet_count']
        df['negative_ratio'] = df['negative'] / df['tweet_count']

        output_path = os.path.join(output_dir, id+".png")
        plot_tweet_weekly(df, output_path, title, id)

