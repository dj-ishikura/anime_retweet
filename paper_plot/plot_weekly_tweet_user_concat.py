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

def plot_weekly_tweet_user(input_csv_A, input_csv_B, output_png):
    df_A = pd.read_csv(input_csv_A)
    df_B = pd.read_csv(input_csv_B)

    # 放送週数を追加（行のインデックス＋1を使用）
    df_A['broadcast_week'] = df_A.index + 1
    df_B['broadcast_week'] = df_B.index + 1

    # プロットの作成
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(df_A['broadcast_week'], df_A['tweet_users_count'], marker='o', label='テレビアニメ作品A')
    plt.plot(df_B['broadcast_week'], df_B['tweet_users_count'], marker='o', label='テレビアニメ作品B')

    plt.tick_params(axis='both', labelsize=12)

    plt.xlabel('放送週数', fontsize=16)
    plt.ylabel('週間ツイートユーザ数', fontsize=16)

    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_png)

if __name__ == "__main__":
    import sys

    input_csv_A = "/work/n213304/learn/anime_retweet_2/count_tweet/2022-10-582_1_week_tweet_counts.csv"
    input_csv_B = "/work/n213304/learn/anime_retweet_2/count_tweet/2022-10-588_1_week_tweet_counts.csv"
    output_png = "plot/plot_weekly_tweet_user_concat.pdf"

    plot_weekly_tweet_user(input_csv_A, input_csv_B, output_png)
