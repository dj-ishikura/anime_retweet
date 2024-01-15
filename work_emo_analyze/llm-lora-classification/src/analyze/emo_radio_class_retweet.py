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

def load_and_merge_data(retweet_object_file, tweet_emo_file):
    # JSON ファイルの読み込み
    with open(tweet_emo_file, 'r', encoding='utf-8') as file:
        predictions_data = json.load(file)

    # CSV ファイルの読み込み
    retweet_data = pd.read_csv(retweet_object_file)

    # tweet_id 列を文字列型に変換
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df['tweet_id'] = predictions_df['tweet_id'].astype(str)
    retweet_ids = set(predictions_df['tweet_id'])

    retweet_data['tweet_id'] = retweet_data['tweet_id'].astype(str)
    tweet_emo_ids = set(retweet_data['tweet_id'])

    # 結合されなかった tweet_id を特定
    unmerged_ids_predictions = retweet_ids - tweet_emo_ids
    unmerged_ids_retweet = tweet_emo_ids - retweet_ids

    print(f"結合されなかった tweet_id の数 (predictions): {len(unmerged_ids_predictions)}")
    print(f"結合されなかった tweet_id の数 (retweet): {len(unmerged_ids_retweet)}")

    return predictions_df.merge(retweet_data, on='tweet_id')

def calculate_retweet_percentages(merged_df):
    # 各カテゴリごとのリツイート数の合計を計算
    retweet_sums = merged_df.groupby('predictions')['retweet_count'].sum()

    # 全リツイート数の合計を計算
    total_retweets = retweet_sums.sum()

    # 各カテゴリの割合を計算
    retweet_percentages = retweet_sums / total_retweets * 100

    return retweet_percentages

def get_data():
    mean_tweet_user_cluster_name = {0: "多い", 1: "少ない", 2: "中くらい"}
    weekly_tweet_user_cluster_name = {0: "上昇", 1: "下降", 2: "U型(横ばい)", 3: "W型(山型)"}

    retweet_object_dir = "/work/n213304/learn/anime_retweet_2/retweet_count"
    tweet_emo_dir = '/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/prediction'
    path = "/work/n213304/learn/anime_retweet_2/anime_class.csv"
    df_class = pd.read_csv(path, index_col="id")

    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0) # keywords.csvはあなたのファイル名に置き換えてください

    anime_tweet_count_list = []
    for file_name in os.listdir(tweet_emo_dir):
        if file_name.endswith('.json'):
            id = os.path.splitext(file_name)[0]
            print(df_title.loc[id, '作品名'])
            tweet_emo_file = os.path.join(tweet_emo_dir, file_name)
            retweet_object_file = os.path.join(retweet_object_dir, id+'.csv')
            merged_df = load_and_merge_data(retweet_object_file, tweet_emo_file)

            retweet_sums = merged_df.groupby('predictions')['retweet_count'].sum()
            max_retweets = merged_df.groupby('predictions')['retweet_count'].max()
            avg_retweets = merged_df.groupby('predictions')['retweet_count'].mean()

            # anime_class.csv からのデータを取得
            weekly_tweet_user_clusters = df_class.loc[id, 'weekly_tweet_user_clusters']
            mean_tweet_user_clusters = df_class.loc[id, 'mean_tweet_user_clusters']

            data = {
                'id': id,
                'title': df_title.loc[id, '作品名'],
                'tweet_count': retweet_sums.sum(),
                'positive': retweet_sums.get(2, 0),  # 仮定: '2' がポジティブ
                'neutral': retweet_sums.get(1, 0),   # 仮定: '1' がニュートラル
                'negative': retweet_sums.get(0, 0),  # 仮定: '0' がネガティブ
                'positive_radio': retweet_sums.get(2, 0) / retweet_sums.sum(),
                'neutral_radio': retweet_sums.get(1, 0) / retweet_sums.sum(),
                'negative_radio': retweet_sums.get(0, 0) / retweet_sums.sum(),
                'max_retweet_positive': max_retweets.get(2, 0),
                'max_retweet_neutral': max_retweets.get(1, 0),
                'max_retweet_negative': max_retweets.get(0, 0),
                'avg_retweet_positive': avg_retweets.get(2, 0),
                'avg_retweet_neutral': avg_retweets.get(1, 0),
                'avg_retweet_negative': avg_retweets.get(0, 0),
                'weekly_tweet_user_clusters': weekly_tweet_user_cluster_name[weekly_tweet_user_clusters],
                'mean_tweet_user_clusters': mean_tweet_user_cluster_name[mean_tweet_user_clusters]
            }
            anime_tweet_count_list.append(data)

    return pd.DataFrame(anime_tweet_count_list)

def plot_emo_number_title(df):
    # データフレームをプロット
    df.plot(
        kind='barh',
        stacked=True,
        x='title',  # タイトル列をx軸に
        y=['positive', 'neutral', 'negative'],  # 各感情カテゴリの列をy軸に
        color=['lightcoral', 'khaki', 'lightblue']  # 各カテゴリの色
    )

    # タイトルと軸ラベルの設定
    plt.title('アニメ毎のツイートの感情分類')
    plt.xlabel('ツイート数')
    plt.ylabel('アニメタイトル')

    # y軸とx軸のフォントサイズの設定
    plt.yticks(fontsize=8)
    plt.xticks(rotation=0)

    # プロットのレイアウトを調整
    plt.tight_layout()

    # プロットを保存
    plt.savefig("./src/analyze/emo_radio_class_retweet/emo_number_title.png")

    # プロットを閉じる
    plt.close()

def plot_emo_radio_title(df):

    # データフレームをプロット
    df.plot(
        kind='barh',
        stacked=True,
        x='title',  # タイトル列をx軸に
        y=['positive_radio', 'neutral_radio', 'negative_radio'],  # 各感情カテゴリの列をy軸に
        color=['lightcoral', 'khaki', 'lightblue']  # 各カテゴリの色
    )

    # タイトルと軸ラベルの設定
    plt.title('アニメ毎のツイートの感情分類（割合）')
    plt.xlabel('割合 (%)')
    plt.ylabel('アニメタイトル')

    # y軸とx軸のフォントサイズの設定
    plt.yticks(fontsize=8)
    plt.xticks(rotation=0)

    # プロットのレイアウトを調整
    plt.tight_layout()

    # プロットを保存
    plt.savefig("./src/analyze/emo_radio_class_retweet/emo_ratio_title.png")

    # プロットを閉じる
    plt.close()

def plot_emo_radio_class(df):
    grouped_df = df.groupby(['weekly_tweet_user_clusters', 'mean_tweet_user_clusters'])

    # 各グループでポジティブ、ニュートラル、ネガティブの割合を計算
    cluster_analysis = grouped_df.agg({
        'positive': 'sum',
        'neutral': 'sum',
        'negative': 'sum',
        'tweet_count': 'sum'
    })

    # 割合を計算
    cluster_analysis[['positive', 'neutral', 'negative']] = (
        cluster_analysis[['positive', 'neutral', 'negative']].div(cluster_analysis['tweet_count'], axis=0) * 100
    )

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
    plt.xlabel('クラスタ')
    plt.ylabel('割合')
    plt.legend(['ポジティブ', 'ニュートラル', 'ネガティブ'])
    plt.tight_layout()  # ラベルが画像の外に出ないように調整
    plt.savefig("./src/analyze/emo_radio_class_retweet/emo_radio_class.png")
    plt.close()

def plot_emo_radio_mean(df):
    grouped_df = df.groupby(['mean_tweet_user_clusters'])

    # 各グループでポジティブ、ニュートラル、ネガティブの割合を計算
    cluster_analysis = grouped_df.agg({
        'positive': 'sum',
        'neutral': 'sum',
        'negative': 'sum',
        'tweet_count': 'sum'
    })

    # 割合を計算
    cluster_analysis[['positive', 'neutral', 'negative']] = (
        cluster_analysis[['positive', 'neutral', 'negative']].div(cluster_analysis['tweet_count'], axis=0) * 100
    )

    # データフレームのインデックスをリセット（グラフのラベル用）
    cluster_analysis = cluster_analysis.reset_index()

    # 積み上げ棒グラフのプロット
    cluster_analysis.plot(
        kind='barh', 
        stacked=True, 
        x='mean_tweet_user_clusters',
        y=['positive', 'neutral', 'negative'],
        color=['lightcoral', 'khaki', 'lightblue']
    )

    plt.title('クラスタ毎の感情割合')
    plt.xlabel('クラスタ')
    plt.ylabel('割合')
    plt.legend(['ポジティブ', 'ニュートラル', 'ネガティブ'])
    plt.tight_layout()  # ラベルが画像の外に出ないように調整
    plt.savefig("./src/analyze/emo_radio_class_retweet/emo_radio_mean.png")
    plt.close()

def plot_emo_radio_weekly(df):
    grouped_df = df.groupby(['weekly_tweet_user_clusters'])

    # 各グループでポジティブ、ニュートラル、ネガティブの割合を計算
    cluster_analysis = grouped_df.agg({
        'positive': 'sum',
        'neutral': 'sum',
        'negative': 'sum',
        'tweet_count': 'sum'
    })

    # 割合を計算
    cluster_analysis[['positive', 'neutral', 'negative']] = (
        cluster_analysis[['positive', 'neutral', 'negative']].div(cluster_analysis['tweet_count'], axis=0) * 100
    )

    # データフレームのインデックスをリセット（グラフのラベル用）
    cluster_analysis = cluster_analysis.reset_index()

    # 積み上げ棒グラフのプロット
    cluster_analysis.plot(
        kind='barh', 
        stacked=True, 
        x='weekly_tweet_user_clusters',
        y=['positive', 'neutral', 'negative'],
        color=['lightcoral', 'khaki', 'lightblue']
    )

    plt.title('クラスタ毎の感情割合')
    plt.xlabel('クラスタ')
    plt.ylabel('割合')
    plt.legend(['ポジティブ', 'ニュートラル', 'ネガティブ'])
    plt.tight_layout()  # ラベルが画像の外に出ないように調整
    plt.savefig("./src/analyze/emo_radio_class_retweet/emo_radio_weekly.png")
    plt.close()

def plot_max_retweet_emo_by_cluster(df):
    # 各アニメで最大リツイート数を持つ感情を特定
    df['max_retweet_emo'] = df[['max_retweet_positive', 'max_retweet_neutral', 'max_retweet_negative']].idxmax(axis=1)

    # クラスタごとに最大リツイート感情を集計
    grouped_df = df.groupby(['weekly_tweet_user_clusters', 'mean_tweet_user_clusters', 'max_retweet_emo']).size().unstack(fill_value=0)

    # データフレームのインデックスをリセット（グラフのラベル用）
    grouped_df = grouped_df.reset_index()
    grouped_df['cluster'] = (
        grouped_df['weekly_tweet_user_clusters'].astype(str) 
        + '-' 
        + grouped_df['mean_tweet_user_clusters'].astype(str)
    )

    # 積み上げ棒グラフのプロット
    grouped_df.plot(
        kind='barh', 
        stacked=True, 
        x='cluster',  # 新しい組み合わせた列を使用
        y=['max_retweet_positive', 'max_retweet_neutral', 'max_retweet_negative'],
        color=['lightcoral', 'khaki', 'lightblue']
    )

    plt.title('クラスタ毎の最大リツイート感情の分布')
    plt.xlabel('クラスタ')
    plt.ylabel('数')
    plt.legend(['ポジティブ', 'ニュートラル', 'ネガティブ'])
    plt.tight_layout()  # ラベルが画像の外に出ないように調整
    plt.savefig("./src/analyze/emo_radio_class_retweet/emo_max_retweet_class.png")
    plt.close()


def main():
    merged_df = get_data()

    plot_emo_number_title(merged_df)
    plot_emo_radio_title(merged_df)
    plot_emo_radio_class(merged_df)
    plot_max_retweet_emo_by_cluster(merged_df)
    plot_emo_radio_weekly(merged_df)
    plot_emo_radio_mean(merged_df)

# 以下の部分は以前と変わらず
if __name__ == "__main__":
    main()

