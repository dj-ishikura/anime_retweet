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
        
        original_data_list.append(df[["tweet_users_count"]].values)

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
data_series = to_time_series_dataset(original_data_list)
print(f'作品数 : {len(anime_weekly_tweet_list)}')

n_clusters = 3
dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
                          n_init=5,
                          metric="euclidean",
                          verbose=False,
                          max_iter_barycenter=10,
                          random_state=0)

pred = dba_km.fit_predict(anime_weekly_tweet_series)

df_anime = pd.read_csv('./anime_data_updated.csv', index_col=0)
title_list = []
for id in anime_ids:
    title = df_anime.loc[id, '作品名']
    title_list.append(title)

# クラスターラベルをデータフレームに追加する
cluster_df = pd.DataFrame({'id': anime_ids, 'title': title_list, 'class': pred})
# クラスターラベルでデータフレームをソートする
sorted_cluster_df = cluster_df.sort_values(by='class')

# ソートされたデータフレームをCSVファイルに保存する
sorted_cluster_df.to_csv('anime_class_dtw.csv', index=False)

from tslearn.barycenters import dtw_barycenter_averaging

def plot_and_save_graphs(title_list, anime_weekly_tweet_series, pred, n_clusters):
    # クラスタごとにディレクトリを作成する
    directory = 'cluster_plots'
    if os.path.exists(directory):
        shutil.rmtree(directory)

    # 新しいディレクトリを作成する
    os.makedirs(directory)
    
    # クラスタごとにグラフをプロットする
    for i in range(n_clusters):
        plt.figure()
        
        # 各クラスタの時系列データを収集する
        cluster_series = [anime_weekly_tweet_series[idx].ravel() for idx, label in enumerate(pred) if label == i]
        # しかし、時系列データが10より少ない要素を持っている場合は除外する
        cluster_series = [anime_weekly_tweet_series[idx].ravel() for idx, label in enumerate(pred) if label == i and len(anime_weekly_tweet_series[idx].ravel()) >= 10]
        
        # クラスタのバリセンターを計算する
        cluster_barycenter = dtw_barycenter_averaging(cluster_series)
        
        for idx, label in enumerate(pred):
            if label == i:
                plt.plot(anime_weekly_tweet_series[idx].ravel())  # 個々の時系列をプロットする
        
        # バリセンターを黒色でプロットする
        plt.plot(cluster_barycenter.ravel(), color='black', linewidth=2, label='Cluster Barycenter')
        
        plt.legend()
        plt.title(f'Cluster {i}')
        plt.savefig(f'cluster_plots/cluster_{i}.png')
        plt.close()  # メモリを節約するためにプロットを閉じる

        plt.figure()
        
        # 各クラスタの時系列データを収集する
        cluster_series = [data_series[idx].ravel() for idx, label in enumerate(pred) if label == i]

        # クラスタのバリセンターを計算する
        cluster_barycenter = dtw_barycenter_averaging(cluster_series)
        
        for idx, label in enumerate(pred):
            if label == i:
                plt.plot(data_series[idx].ravel())  # 個々の時系列をプロットする
        
        # バリセンターを黒色でプロットする
        plt.plot(cluster_barycenter.ravel(), color='black', linewidth=2, label='Cluster Barycenter')
        
        plt.legend()
        plt.title(f'Cluster {i}')
        plt.savefig(f'cluster_plots/cluster_{i}_.png')
        plt.close()  # メモリを節約するためにプロットを閉じる


def save_images_to_pdf(n_clusters):
    pdf_filename = 'cluster_plots.pdf'
    
    with PdfPages(pdf_filename) as pdf_pages:
        cluster_path = sorted(glob.glob('cluster_plots/*.png'))
        print(cluster_path)
        for png_file in cluster_path:
            img = plt.imread(png_file)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')  # 軸を非表示にする
            pdf_pages.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()

# グラフのプロットと保存
plot_and_save_graphs(title_list, anime_weekly_tweet_series, pred, n_clusters)

# PDFへの画像の保存
save_images_to_pdf(n_clusters)