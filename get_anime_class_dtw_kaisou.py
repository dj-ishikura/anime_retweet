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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import fastcluster
from scipy.cluster.hierarchy import dendrogram, fcluster
from tslearn.metrics import cdist_dtw


# データを格納するディレクトリのパス
directory_path = 'count_tweet'

# データを格納する辞書
data_dict = {}
original_data_list = []
anime_weekly_tweet_list = []
mean_anime_weekly_tweet_list = []
anime_ids = []

# ディレクトリ内の各ファイルを処理するが、最初の10件のファイルのみを処理する
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, index_col=0)

        if len(df) == 1:
            print(f"Skipping series from {filename} due to length being 1.")
            continue  # この時系列データをスキップし、次のファイルに移動するのだ
        else len(df) < 11:
            print(f"Skipping series from {filename} due to length being 11 under.")
            continue  # この時系列データをスキップし、次のファイルに移動するのだ
        
        original_data_list.append(df[["tweet_users_count"]].values)
        mean_anime_weekly_tweet_list.append(df[["tweet_users_count"]].mean().values)

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

# 平均値のリストから中央値を計算
# データを DataFrame に変換する
df_mean = pd.DataFrame(mean_anime_weekly_tweet_list, columns=['value'])
# Min-Max正規化を行う
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df_mean)
# 正規化されたデータに対して階層的クラスタリングを行う
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
tweet_user_class = agg_clustering.fit_predict(normalized_data)
tweet_user_class = [t + 1 for t in tweet_user_class]

# Plot the normalized data and color it by cluster labels
plt.figure(figsize=(10, 6))
plt.scatter(range(len(normalized_data)), normalized_data, c=tweet_user_class, cmap='rainbow')
plt.title('Clustering Results with 2 clusters')
plt.xlabel('Index')
plt.ylabel('Normalized Value')
plt.savefig("plot_mean_anime_weekly_tweet_class_normalized.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(mean_anime_weekly_tweet_list)), mean_anime_weekly_tweet_list, c=tweet_user_class, cmap='rainbow')
plt.title('Clustering Results with 2 clusters')
plt.xlabel('Index')
plt.ylabel('Value')
plt.savefig("plot_mean_anime_weekly_tweet_class.png")
plt.close()

# Plot the hierarchical clustering dendrogram
linked = linkage(normalized_data, method=agg_clustering.linkage)  # using 'ward' method
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', labels=tweet_user_class, distance_sort='descending', show_leaf_counts=True)
plt.savefig("plot_mean_anime_weekly_tweet_class_kaisou.png")
plt.close()


# DTW距離行列を計算するのだ
distance_matrix = cdist_dtw(anime_weekly_tweet_series)
# fastclusterを使用して階層的クラスタリングを実行するのだ
linkage_matrix = fastcluster.linkage_vector(distance_matrix, method='ward')
# クラスタリングの結果を取得するのだ
n_clusters = 3
pred = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
# pred[:] = 1

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, orientation='top', labels=pred, distance_sort='descending', show_leaf_counts=True)
plt.savefig("plot_anime_weekly_tweet_class_kaisou.png")
plt.close()

df_anime = pd.read_csv('./anime_data_updated.csv', index_col=0)
title_list = []
for id in anime_ids:
    title = df_anime.loc[id, '作品名']
    title_list.append(title)

# クラスターラベルをデータフレームに追加する
cluster_df = pd.DataFrame({'id': anime_ids, 'title': title_list, 'class': pred, 'tweet_user_class': tweet_user_class})
# クラスターラベルでデータフレームをソートする
sorted_cluster_df = cluster_df.sort_values(by=['class', 'tweet_user_class'])

# ソートされたデータフレームをCSVファイルに保存する
sorted_cluster_df.to_csv('anime_class_dtw_kaisou.csv', index=False)

from tslearn.barycenters import dtw_barycenter_averaging

def plot_and_save_graphs(title_list, anime_weekly_tweet_series, sorted_cluster_df, class_label, tweet_user_label, directory):  
    plt.figure()
    
    # 指定されたラベルに一致する時系列データを収集する
    matching_series = [anime_weekly_tweet_series[idx].ravel() 
                       for idx, row in sorted_cluster_df.iterrows() 
                       if row['class'] == class_label and row['tweet_user_class'] == tweet_user_label]

    print(f'{class_label}_{tweet_user_label} : {len(matching_series)}')
    
    # しかし、時系列データが10より少ない要素を持っている場合は除外する
    matching_series = [series for series in matching_series if len(series) >= 10]
    
    # クラスタのバリセンターを計算する
    cluster_barycenter = dtw_barycenter_averaging(matching_series)
    
    for idx, row in sorted_cluster_df.iterrows():
        if row['class'] == class_label and row['tweet_user_class'] == tweet_user_label:
            plt.plot(anime_weekly_tweet_series[idx].ravel())  # 個々の時系列をプロットする
    
    # バリセンターを黒色でプロットする
    plt.plot(cluster_barycenter.ravel(), color='black', linewidth=2, label='Cluster Barycenter')
    
    plt.legend()
    plt.title(f'Class {class_label}, Tweet User Class {tweet_user_label}')
    plt.savefig(f'{directory}/cluster_{class_label}_{tweet_user_label}_.png')
    plt.close()  # メモリを節約するためにプロットを閉じる

    plt.figure()
    
    # 指定されたラベルに一致する時系列データを収集する
    matching_series = [original_data_list[idx].ravel() 
                       for idx, row in sorted_cluster_df.iterrows() 
                       if row['class'] == class_label and row['tweet_user_class'] == tweet_user_label]
    
    # しかし、時系列データが10より少ない要素を持っている場合は除外する
    matching_series = [series for series in matching_series if len(series) >= 10]
    
    # クラスタのバリセンターを計算する
    cluster_barycenter = dtw_barycenter_averaging(matching_series)
    
    for idx, row in sorted_cluster_df.iterrows():
        if row['class'] == class_label and row['tweet_user_class'] == tweet_user_label:
            plt.plot(original_data_list[idx].ravel())  # 個々の時系列をプロットする
    
    # バリセンターを黒色でプロットする
    plt.plot(cluster_barycenter.ravel(), color='black', linewidth=2, label='Cluster Barycenter')
    
    plt.legend()
    plt.title(f'Class {class_label}, Tweet User Class {tweet_user_label}')
    plt.savefig(f'{directory}/cluster_{class_label}_{tweet_user_label}.png')
    plt.close()  # メモリを節約するためにプロットを閉じる


def save_images_to_pdf(n_clusters):
    pdf_filename = 'cluster_plots_dtw_kaisou.pdf'
    
    with PdfPages(pdf_filename) as pdf_pages:
        cluster_path = sorted(glob.glob('cluster_plots_dtw_kaisou/*.png'))
        print(cluster_path)
        for png_file in cluster_path:
            img = plt.imread(png_file)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')  # 軸を非表示にする
            pdf_pages.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()

# グラフのプロットと保存
directory = 'cluster_plots_dtw_kaisou'
if os.path.exists(directory):
    shutil.rmtree(directory)
os.makedirs(directory)
for class_label in sorted_cluster_df['class'].unique():
    for tweet_user_label in sorted_cluster_df['tweet_user_class'].unique():
        plot_and_save_graphs(title_list, anime_weekly_tweet_series, sorted_cluster_df, class_label, tweet_user_label, directory)

# PDFへの画像の保存
save_images_to_pdf(n_clusters)