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

def make_3division(df):
    df.index = pd.to_datetime(df.index)  # インデックスをdatetimeに変換
    # 週数（何週目か）を計算
    df['week_number'] = (df.index - df.index.min()).days // 7 + 1

    # 全体の週数
    total_weeks = df['week_number'].max()

    # カテゴリを割り当てる関数
    def categorize_week(week):
        if week <= 4:
            return 'First'
        elif week <= total_weeks - 4:
            return 'Middle'
        else:
            return 'Last'

    # カテゴリを割り当て
    df['division'] = df['week_number'].apply(categorize_week)
    # 各クラスごとの平均値を求める
    division_means = df.groupby('division')['count'].mean()

    # 各行に対して、その行が属するクラスの平均値を代入する
    df['division_mean'] = df['division'].map(division_means)

    return df

def plot_week_tweet_user_tendency(input_file, output_png, title, id):
    df = pd.read_csv(input_file)
    if len(df) > 1:
        df.set_index('date', inplace=True)
        df['month_day'] = df.index.str.split('-').str[1] + '-' + df.index.str.split('-').str[2]
        df = make_3division(df)

        # プロットの作成
        fig, ax1 = plt.subplots()

        # 折れ線グラフの作成
        line_plot = df.plot(kind='line', x='month_day', y='count', marker='o', ax=ax1, legend=False)

        # 棒グラフの作成
        bar_plot = df.iloc[:-1].plot(kind='bar', x='month_day', y='division_mean', ax=ax1, 
        color='green', align='edge', width=1, alpha=0.4, label='区間ごとの週間ツイートユーザ数の平均値')

        # 凡例の表示（フォントサイズの変更も含む）
        ax1.legend(handles=[bar_plot.get_legend_handles_labels()[0][0]], labels=['区間ごとの週間ツイートユーザ数の平均値'], fontsize=14)

        
        # 文字サイズの調整
        ax1.set_xlabel('放送日', fontsize=14)
        ax1.set_ylabel('週間ツイートユーザ数', fontsize=14)
        plt.title(f'{title}', fontsize=16)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)

        plt.xticks(rotation=45)  # x軸のラベルを45度回転して見やすくする
        plt.tight_layout()  # ラベルが画像の外に出ないように調整
        plt.savefig(output_png)


def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('./anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']
    # end_date = end_date + timedelta(days=7)

    return title

if __name__ == "__main__":
    input_directory = 'count_tweet'
    output_directory = 'week_tweet_user_tendency'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('1_week_tweet_counts.csv'):
            id = filename.split('_')[0]
            output_png = os.path.join(output_directory, id + '.png')            
            title = get_info_from_csv(id)
            input_file = os.path.join(input_directory, filename)
            plot_week_tweet_user_tendency(input_file, output_png, title, id)

