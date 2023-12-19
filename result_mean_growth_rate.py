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
import numpy as np

def calculate_average_growth_rate(input_file):
    df = pd.read_csv(input_file)
    
    # 1つ目のcountを取得
    first_count = df['count'].iloc[0]
    
    # 最後のcountを取得
    last_count = df['count'].iloc[-1]
    
    # 1つ目のcountと最後のcountを比較して増加率を計算
    growth_rate = (last_count / first_count) * 100
    
    return growth_rate



def calculate_average_growth_rate_2(input_file):
    df = pd.read_csv(input_file)
    
    # 増加率を計算
    df['growth_rate'] = df['count'].pct_change() * 100

    # inf または -inf を NaN に変換
    df['growth_rate'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # 平均増加率を計算（NaNは自動的に除外される）
    avg_growth_rate = df['growth_rate'].mean()
    
    return avg_growth_rate

if __name__ == "__main__":
    input_directory = 'count_tweet'
    output_csv= './result/mean_growth_rate.csv'

    results = pd.DataFrame(columns=["id", "avg_growth_rate"])

    for filename in os.listdir(input_directory):
        if filename.endswith('1_week_tweet_counts.csv'):
            input_file = os.path.join(input_directory, filename)
            id = filename.split('_')[0]
            avg_growth_rate = calculate_average_growth_rate(input_file)

            # 結果をデータフレームに追加
            new_row = pd.DataFrame({"id": [id], "avg_growth_rate": [avg_growth_rate]})
            results = pd.concat([results, new_row], ignore_index=True)


    # 結果をCSVに保存
    results.dropna(subset=["avg_growth_rate"], inplace=True)
    results.to_csv(output_csv, index=False)