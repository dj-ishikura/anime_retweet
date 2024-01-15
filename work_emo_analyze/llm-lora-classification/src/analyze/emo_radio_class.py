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

def get_title_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']

    return title

tweet_emo_dir = 'tweet_emo_weekly_kari'
path = "/work/n213304/learn/anime_retweet_2/anime_class.csv"
df_class = pd.read_csv(path, index_col="id")

df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

mean_tweet_user_cluster_name = {0: "多い", 1: "少ない", 2: "中くらい"}
weekly_tweet_user_cluster_name = {0: "上昇", 1: "下降", 2: "山型", 3: "横ばい"}

anime_tweet_count_list = []

for file_name in os.listdir(tweet_emo_dir):
    if file_name.endswith('.csv'):
        id = os.path.splitext(file_name)[0]
        file_path = os.path.join(tweet_emo_dir, file_name)
        df = pd.read_csv(file_path)
        total_positive = df['positive'].sum()
        total_neutral = df['neutral'].sum()
        total_negative = df['negative'].sum()
        total_tweet = df['tweet_count'].sum()

        # anime_class.csv からのデータを取得
        weekly_tweet_user_clusters = df_class.loc[id, 'weekly_tweet_user_clusters']
        mean_tweet_user_clusters = df_class.loc[id, 'mean_tweet_user_clusters']

        data = {
            'id': id,
            'title': df_title.loc[id, '作品名'],
            'tweet_count': total_tweet,
            'positive': total_positive,
            'neutral': total_neutral,
            'negative': total_negative,
            'weekly_tweet_user_clusters': weekly_tweet_user_cluster_name[weekly_tweet_user_clusters],
            'mean_tweet_user_clusters': mean_tweet_user_cluster_name[mean_tweet_user_clusters]
        }
        anime_tweet_count_list.append(data)

# 全てのデータを結合
combined_df = pd.DataFrame(anime_tweet_count_list)

# アニメ作品全体の感情の合計を計算
total_positive = combined_df['positive'].sum()
total_neutral = combined_df['neutral'].sum()
total_negative = combined_df['negative'].sum()

# 円グラフのプロット
fig, ax = plt.subplots(figsize=(10, 6))
ax.pie(
    [total_positive, total_neutral, total_negative],
    labels=['ポジティブ', 'ニュートラル', 'ネガティブ'],
    colors=['lightcoral', 'khaki', 'lightblue'],
    autopct='%1.1f%%'
)
ax.set_title('アニメ作品全体の感情分析結果', fontsize=20)

# フォントサイズの設定
for text in ax.texts:
    text.set_fontsize(16)

plt.tight_layout()
plt.savefig("./src/analyze/emo_radio_overall.png")
plt.close()

grouped_df = combined_df.groupby(['weekly_tweet_user_clusters', 'mean_tweet_user_clusters'])

# 各グループでポジティブ、ニュートラル、ネガティブの割合を計算
cluster_analysis = grouped_df.agg({
    'positive': 'sum',
    'neutral': 'sum',
    'negative': 'sum',
    'tweet_count': 'sum'
}).apply(lambda x: x / x['tweet_count'], axis=1)

# データフレームのインデックスをリセット（グラフのラベル用）
cluster_analysis = cluster_analysis.reset_index()
cluster_analysis['cluster'] = cluster_analysis['weekly_tweet_user_clusters'] + '-' + cluster_analysis['mean_tweet_user_clusters']


# 積み上げ棒グラフのプロット
cluster_analysis.plot(
    kind='barh', 
    stacked=True, 
    x='cluster',  # 新しい組み合わせた列を使用
    y=['positive', 'neutral', 'negative'],
    color=['lightcoral', 'khaki', 'lightblue']
)

plt.title('クラスタ毎の感情割合')
plt.xlabel('クラスタ', fontsize=16)
plt.ylabel('割合', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend(['ポジティブ', 'ニュートラル', 'ネガティブ'])
plt.tight_layout()  # ラベルが画像の外に出ないように調整
plt.savefig("./src/analyze/emo_radio_class.png")
plt.close()


grouped_df = combined_df.groupby(['weekly_tweet_user_clusters'])

# 各グループでポジティブ、ニュートラル、ネガティブの割合を計算
cluster_analysis = grouped_df.agg({
    'positive': 'sum',
    'neutral': 'sum',
    'negative': 'sum',
    'tweet_count': 'sum'
}).apply(lambda x: x / x['tweet_count'], axis=1)

# データフレームのインデックスをリセット（グラフのラベル用）
desired_order = ["上昇", "下降", "山型", "横ばい", ]
cluster_analysis = cluster_analysis.reset_index().set_index('weekly_tweet_user_clusters').loc[desired_order].reset_index()

# 積み上げ棒グラフのプロット
cluster_analysis.plot(
    kind='bar', 
    stacked=True, 
    x='weekly_tweet_user_clusters',  # 新しい組み合わせた列を使用
    y=['positive', 'neutral', 'negative'],
    color=['lightcoral', 'khaki', 'lightblue']
)

# plt.title('クラスタ毎の感情割合')
plt.xlabel('週間ツイートユーザ数の推移傾向', fontsize=16)
plt.xticks(rotation=0)
plt.ylabel('割合', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend(['ポジティブ', 'ニュートラル', 'ネガティブ'])
plt.tight_layout()  # ラベルが画像の外に出ないように調整
plt.savefig("./src/analyze/emo_radio_class_weekly.pdf")
plt.close()

grouped_df = combined_df.groupby(['mean_tweet_user_clusters'])

# 各グループでポジティブ、ニュートラル、ネガティブの割合を計算
cluster_analysis = grouped_df.agg({
    'positive': 'sum',
    'neutral': 'sum',
    'negative': 'sum',
    'tweet_count': 'sum'
}).apply(lambda x: x / x['tweet_count'], axis=1)

# データフレームのインデックスをリセット（グラフのラベル用）
desired_order = ["多い", "中くらい", "少ない"]
cluster_analysis = cluster_analysis.reset_index().set_index('mean_tweet_user_clusters').loc[desired_order].reset_index()

# 積み上げ棒グラフのプロット
cluster_analysis.plot(
    kind='bar',  # 垂直バーに変更
    stacked=True,
    x='mean_tweet_user_clusters',  # y軸に設定
    y=['positive', 'neutral', 'negative'],  # x軸に設定
    color=['lightcoral', 'khaki', 'lightblue']
)

# plt.title('クラスタ毎の感情割合')
plt.xlabel('平均週間ツイートユーザ数', fontsize=16)
plt.xticks(rotation=0)
plt.ylabel('割合', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend(['ポジティブ', 'ニュートラル', 'ネガティブ'])
plt.tight_layout()  # ラベルが画像の外に出ないように調整
plt.savefig("./src/analyze/emo_radio_class_mean.pdf")
plt.close()


combined_df.plot(
    kind='barh', 
    stacked=True, 
    x='title',  # 新しい組み合わせた列を使用
    y=['positive', 'neutral', 'negative'],
    color=['lightcoral', 'khaki', 'lightblue']
)

plt.title('アニメ毎のツイートの感情分類')
plt.xlabel('クラスタ')
plt.ylabel('ツイート数')
# plt.legend(['ポジティブ', 'ニュートラル', 'ネガティブ'])
plt.yticks(fontsize=8)  # y軸のフォントサイズを8に設定
plt.xticks(rotation=0)  # x軸のラベルの角度を0度（水平）に設定
plt.tight_layout()  # ラベルが画像の外に出ないように調整
plt.savefig("./src/analyze/emo_number_title.png")
plt.close()

grouped_df = combined_df.groupby(['title'])

# 各グループでポジティブ、ニュートラル、ネガティブの割合を計算
cluster_analysis = grouped_df.agg({
    'positive': 'sum',
    'neutral': 'sum',
    'negative': 'sum',
    'tweet_count': 'sum'
}).apply(lambda x: x / x['tweet_count'], axis=1)

cluster_analysis = cluster_analysis.reset_index()

plt.figure(figsize=(10, 15))  # グラフのサイズを横10インチ、縦15インチに設定
cluster_analysis.plot(kind='barh', stacked=True, x='title', y=['positive', 'neutral', 'negative'], color=['lightcoral', 'khaki', 'lightblue'])
plt.title('アニメ毎のツイートの感情分類')
plt.xlabel('クラスタ')
plt.ylabel('ツイート数')
# plt.legend(['ポジティブ', 'ニュートラル', 'ネガティブ'])
plt.yticks(fontsize=8)  # y軸のフォントサイズを8に設定
plt.xticks(rotation=0)  # x軸のラベルの角度を0度（水平）に設定
plt.tight_layout()  # レイアウトを調整
plt.savefig("./src/analyze/emo_ratio_title.png")
plt.close()

# クラスタリングごとの感情分析, 円グラフ, サブプロット
grouped_df = combined_df.groupby(['weekly_tweet_user_clusters', 'mean_tweet_user_clusters'])
cluster_analysis = grouped_df.agg({
    'positive': 'sum',
    'neutral': 'sum',
    'negative': 'sum',
    'tweet_count': 'sum'
}).apply(lambda x: x / x['tweet_count'], axis=1)
cluster_analysis = cluster_analysis.reset_index()

# 4×3のグリッドでサブプロットを作成
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for i in range(3):  # weekly_tweet_user_clusters のクラスタ数
    for j in range(4):  # mean_tweet_user_clusters のクラスタ数
        # クラスタに対応するデータを選択
        cluster_data = cluster_analysis[
            (cluster_analysis['mean_tweet_user_clusters'] == mean_tweet_user_cluster_name[i]) &
            (cluster_analysis['weekly_tweet_user_clusters'] == weekly_tweet_user_cluster_name[j]) 
        ]

        # NaN値を0に置き換え
        pie_data = cluster_data[['positive', 'neutral', 'negative']].sum().fillna(0)

        ax = axes[i, j]
        ax.set_title(f'{mean_tweet_user_cluster_name[i]}-{weekly_tweet_user_cluster_name[j]}', fontsize=24)

        # データの合計が0でないことを確認
        if pie_data.sum() == 0:
            ax.axis("off")
        else:
            # 選択したデータを元に円グラフを描画
            pie_slices, texts, autotexts = ax.pie(
                cluster_data[['positive', 'neutral', 'negative']].sum(), 
                labels=['ポジティブ', 'ニュートラル', 'ネガティブ'], 
                colors=['lightcoral', 'khaki', 'lightblue'], 
                autopct='%1.1f%%'
            )

            # ラベルのフォントサイズを設定
            for text in texts:
                text.set_fontsize(20)

            # パーセンテージのフォントサイズを設定
            for autotext in autotexts:
                autotext.set_fontsize(20)

# グラフ全体のタイトル
# plt.suptitle('各クラスタにおける感情分析結果', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # タイトルのためのスペースを確保
plt.savefig("./src/analyze/emo_radio_class_circle.png")
plt.close()