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


# データを格納するディレクトリのパス
directory_path = 'count_tweet'

# データを格納する辞書
data_dict = {}
anime_weekly_tweet_list = []
anime_ids = []

# ディレクトリ内の各ファイルを処理するが、最初の10件のファイルのみを処理する
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, index_col=0)

        if len(df) == 1:
            print(f"Skipping series from {filename} due to length being 1.")
            continue  # この時系列データをスキップし、次のファイルに移動するのだ

        # 正規化
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(df[["tweet_users_count"]])

        # データの長さを取得
        original_length = len(normalized_data)

        # リサンプリング
        x_original = np.linspace(0, 1, original_length)  # 現在の時間点
        x_new = np.linspace(0, 1, 12)  # 新しい時間点
        interpolating_function = interp1d(x_original, normalized_data, axis=0, kind='linear', fill_value="extrapolate")
        resampled_data = interpolating_function(x_new)

        # 処理されたデータをリストに追加
        anime_id = filename.split('_')[0]
        anime_ids.append(anime_id)
        anime_weekly_tweet_list.append(resampled_data)

anime_weekly_tweet_series = to_time_series_dataset(anime_weekly_tweet_list)

# 導関数を格納するリスト
derivatives_list = []

# 各アニメについて導関数を計算するのだ
for series in anime_weekly_tweet_series:
    # numpyのgradient関数を使用して導関数を計算するのだ
    derivative = np.gradient(series, axis=0)
    derivatives_list.append(derivative)

# 導関数のデータセットを作成するのだ
derivatives_dataset = to_time_series_dataset(derivatives_list)

# 類似度行列の作成

from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# データセットの正規化（必要に応じて）
scaler = TimeSeriesScalerMeanVariance()
derivatives_dataset_scaled = scaler.fit_transform(derivatives_dataset)

# cdist_dtw関数を使用して類似度行列を計算するのだ
similarity_matrix = cdist_dtw(derivatives_dataset_scaled)

# クラスタリング

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# クラスタ数を設定するのだ。この例では3としているが、君のデータに適したクラスタ数を選ぶことが重要なのだ
n_clusters = 10

# K-meansクラスタリングを実行するのだ
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(similarity_matrix)

# シルエットスコアを計算してクラスタリングの質を評価するのだ
silhouette_avg = silhouette_score(similarity_matrix, clusters)

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
sorted_cluster_df.to_csv('anime_class_ddtw.csv', index=False)

from tslearn.barycenters import dtw_barycenter_averaging

def plot_and_save_graphs(title_list, anime_weekly_tweet_series, clusters, n_clusters):
    # クラスタごとにディレクトリを作成する
    directory = 'cluster_plots_ddtw'
    if os.path.exists(directory):
        shutil.rmtree(directory)

    # 新しいディレクトリを作成する
    os.makedirs(directory)
    
    # クラスタごとにグラフをプロットする
    for i in range(n_clusters):
        plt.figure()
        
        # 各クラスタの時系列データを収集する
        cluster_series = [anime_weekly_tweet_series[idx].ravel() for idx, label in enumerate(clusters) if label == i]
        # しかし、時系列データが10より少ない要素を持っている場合は除外する
        cluster_series = [anime_weekly_tweet_series[idx].ravel() for idx, label in enumerate(clusters) if label == i and len(anime_weekly_tweet_series[idx].ravel()) >= 10]
        
        # クラスタのバリセンターを計算する
        cluster_barycenter = dtw_barycenter_averaging(cluster_series)
        
        for idx, label in enumerate(clusters):
            if label == i:
                plt.plot(anime_weekly_tweet_series[idx].ravel())  # 個々の時系列をプロットする
        
        # バリセンターを黒色でプロットする
        plt.plot(cluster_barycenter.ravel(), color='black', linewidth=2, label='Cluster Barycenter')
        
        plt.legend()
        plt.title(f'Cluster {i}')
        plt.savefig(f'cluster_plots_ddtw/cluster_{i}.png')
        plt.close()  # メモリを節約するためにプロットを閉じる


def save_images_to_pdf(n_clusters):
    pdf_filename = 'cluster_plots_ddtw.pdf'
    
    with PdfPages(pdf_filename) as pdf_pages:
        for i in range(n_clusters):
            cluster_path = os.path.join('cluster_plots_ddtw', f'cluster_{i}.png')
            if os.path.exists(cluster_path):
                img = plt.imread(cluster_path)
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.axis('off')  # 軸を非表示にする
                plt.title(f'Cluster {i}', fontsize=16)  # クラスタ番号をタイトルとして表示する
                pdf_pages.savefig(bbox_inches='tight', pad_inches=0)
                plt.close()

# グラフのプロットと保存
plot_and_save_graphs(title_list, anime_weekly_tweet_series, clusters, n_clusters)

# PDFへの画像の保存
save_images_to_pdf(n_clusters)