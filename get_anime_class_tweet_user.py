from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.barycenters import dtw_barycenter_averaging
import os
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties # 日本語対応
import japanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shutil
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import glob


# データを格納するディレクトリのパス
directory_path = 'count_tweet'

# データを格納する辞書
data_dict = {}
original_data_list = []
anime_tweet_user_list = []
anime_ids = []

# ディレクトリ内の各ファイルを処理するが、最初の10件のファイルのみを処理する
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, index_col=0)

        if len(df) == 1:
            print(f"Skipping series from {filename} due to length being 1.")
            continue  # この時系列データをスキップし、次のファイルに移動するのだ
        
        anime_tweet_user_list.append(df[["tweet_users_count"]].mean())
        original_data_list.append(df[["tweet_users_count"]].values)
        anime_id = filename.split('_')[0]
        anime_ids.append(anime_id)

data_series = to_time_series_dataset(original_data_list)

# 対数変換を適用するのだ
log_transformed_data = np.log1p(anime_tweet_user_list)  # np.log1pは0のための安定した対数変換を提供するのだ

from sklearn.preprocessing import StandardScaler

# 正規化を適用するのだ
scaler = StandardScaler()
normalized_data = scaler.fit_transform(log_transformed_data)

from sklearn.cluster import KMeans

# K-meansクラスタリングを適用するのだ
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters)  # ここではクラスタ数を3としているが、適切なクラスタ数を選択することが重要なのだ
clusters = kmeans.fit_predict(normalized_data)

df_anime = pd.read_csv('./anime_data_updated.csv', index_col=0)
title_list = []
for id in anime_ids:
    title = df_anime.loc[id, '作品名']
    title_list.append(title)

# クラスターラベルをデータフレームに追加する
cluster_df = pd.DataFrame({'id': anime_ids, 'title': title_list, 'class': clusters})
# クラスターラベルでデータフレームをソートする
sorted_cluster_df = cluster_df.sort_values(by='class')

# ソートされたデータフレームをCSVファイルに保存する
sorted_cluster_df.to_csv('anime_class_tweet_user.csv', index=False)

from tslearn.barycenters import dtw_barycenter_averaging

def plot_and_save_graphs(title_list, data_series, pred, n_clusters):
    # クラスタごとにディレクトリを作成する
    directory = 'cluster_plots_tweet_user'
    if os.path.exists(directory):
        shutil.rmtree(directory)

    # 新しいディレクトリを作成する
    os.makedirs(directory)
    
    # クラスタごとにグラフをプロットする
    for i in range(n_clusters):
        plt.figure()
        
        # 各クラスタの時系列データを収集する
        cluster_series = [data_series[idx].ravel() for idx, label in enumerate(pred) if label == i]
        # しかし、時系列データが10より少ない要素を持っている場合は除外する
        cluster_series = [data_series[idx].ravel() for idx, label in enumerate(pred) if label == i and len(data_series[idx].ravel()) >= 10]
        
        # クラスタのバリセンターを計算する
        cluster_barycenter = dtw_barycenter_averaging(cluster_series)
        
        for idx, label in enumerate(pred):
            if label == i:
                plt.plot(data_series[idx].ravel())  # 個々の時系列をプロットする
        
        # バリセンターを黒色でプロットする
        plt.plot(cluster_barycenter.ravel(), color='black', linewidth=2, label='Cluster Barycenter')
        
        plt.legend()
        plt.title(f'Cluster {i}')
        plt.savefig(f'cluster_plots_tweet_user/cluster_{i}.png')
        plt.close()  # メモリを節約するためにプロットを閉じる

        plt.figure()

    # クラスの一覧を取得するのだ
    unique_clusters = np.unique(pred)

    # 各クラスに対してプロットを行うのだ
    for unique_cluster in unique_clusters:
        # 現在のクラスに属するデータポイントを抽出するのだ
        mask = np.array(pred) == unique_cluster
        current_avg_weekly_tweet_users = np.array(anime_tweet_user_list)[mask]
        
        # データポイントをプロットするのだ
        plt.scatter(current_avg_weekly_tweet_users, 
                    [unique_cluster] * len(current_avg_weekly_tweet_users), 
                    label=f'Class {unique_cluster}')

    # 凡例を表示するのだ
    plt.legend()

    # 軸ラベルを設定するのだ
    plt.xlabel('Average Weekly Tweet Users')
    plt.ylabel('Class')

    plt.savefig(f'cluster_plots_tweet_user/cluster_.png')
    plt.close()  # メモリを節約するためにプロットを閉じる


def save_images_to_pdf(n_clusters):
    pdf_filename = 'cluster_plots_tweet_user.pdf'
    
    with PdfPages(pdf_filename) as pdf_pages:
        cluster_path = sorted(glob.glob('cluster_plots_tweet_user/*.png'))
        print(cluster_path)
        for png_file in cluster_path:
            img = plt.imread(png_file)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')  # 軸を非表示にする
            pdf_pages.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()

# グラフのプロットと保存
plot_and_save_graphs(title_list, data_series, clusters, n_clusters)

# PDFへの画像の保存
save_images_to_pdf(n_clusters)