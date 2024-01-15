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

# データファイルのパス
data_file_path = '/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/data/tweet_data_randam_text.jsonl'

# データを読み込む
with open(data_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# predictionsの値ごとにカウント
predictions_count = {0: 0, 1: 0, 2: 0}
for item in data:
    predictions_count[item['predictions']] += 1

# 円グラフのプロット
labels = ['ネガティブ', 'ニュートラル', 'ポジティブ']
sizes = [predictions_count[0], predictions_count[1], predictions_count[2]]
colors = ['lightblue', 'khaki', 'lightcoral']

plt.figure(figsize=(10, 6))
plt.title("ツイート全体の感情の割合", fontsize=20)
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, fontsize=16)
plt.tight_layout()  # ラベルが画像の外に出ないように調整
plt.savefig("/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/src/analyze/emo_radio_randam.png")