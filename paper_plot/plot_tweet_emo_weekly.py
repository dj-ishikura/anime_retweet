# -*- coding: utf-8 -*-
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
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42

def plot_tweet_weekly(input_csv, output_png, title):
    df = pd.read_csv(input_csv)
    df['month_day'] = df['date'].str.split('-').str[1] + '-' + df['date'].str.split('-').str[2]

    # プロットの作成と保存
    df.plot(kind='bar', stacked=True, y=['positive', 'neutral', 'negative'], 
            color=['lightcoral', 'khaki', 'lightblue'], figsize = (8,6))
    # plt.title(f'{id}\n{title} : Tweet Emo Count')
    plt.title(f'{title}', fontsize=20)
    plt.tick_params(axis='both', labelsize=12)
    plt.xticks(ticks=range(len(df['month_day'])), labels=df['month_day'], rotation=45)
    plt.xlabel('放送日', fontsize=18)
    plt.ylabel('週間ツイート数', fontsize=18)
    plt.legend(labels=['ポジティブ', 'ニュートラル', 'ネガティブ'], fontsize=16)  # 凡例のラベルを指定
    plt.tight_layout()  # ラベルが画像の外に出ないように調整
    plt.savefig(output_png)

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

if __name__ == "__main__":
    import sys

    input_csv = sys.argv[1]
    output_png = sys.argv[2]
    title = sys.argv[3]
    # title, start_date, end_date = get_info_from_csv(id)

    plot_tweet_weekly(input_csv, output_png, title)
