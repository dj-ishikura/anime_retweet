import os
import glob
import shutil
import numpy as np
import pandas as pd
import fastcluster
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import cdist_dtw
from matplotlib.backends.backend_pdf import PdfPages
from tslearn.barycenters import dtw_barycenter_averaging
import matplotlib.gridspec as gridspec

def load_and_preprocess_data(directory_path):
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

    return anime_weekly_tweet_series, original_data_list, anime_ids, normalized_data

def get_result_cluster(anime_ids, title_list, dtw_class, tweet_user_class):
    # クラスターラベルをデータフレームに追加する
    cluster_df = pd.DataFrame({'id': anime_ids, 'title': title_list, 'dtw_class': dtw_class, 'tweet_user_class': tweet_user_class})
    # クラスターラベルでデータフレームをソートする
    sorted_cluster_df = cluster_df.sort_values(by=['dtw_class', 'tweet_user_class'])

    # ソートされたデータフレームをCSVファイルに保存する
    sorted_cluster_df.to_csv('anime_class_dtw_kaisou.csv', index=False)
    return sorted_cluster_df


def get_title(anime_ids):
    df_anime = pd.read_csv('./anime_data_updated.csv', index_col=0)
    title_list = []
    for id in anime_ids:
        title = df_anime.loc[id, '作品名']
        title_list.append(title)
    return title_list

def cluster_by_mean_value(mean_anime_weekly_tweet_list, tweet_clusters):
    # 平均値のリストから中央値を計算
    # データを DataFrame に変換する
    df_mean = pd.DataFrame(mean_anime_weekly_tweet_list, columns=['value'])
    # Min-Max正規化を行う
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df_mean)
    # 正規化されたデータに対して階層的クラスタリングを行う
    agg_clustering = AgglomerativeClustering(n_clusters=tweet_clusters, linkage='ward')
    tweet_user_class = agg_clustering.fit_predict(normalized_data)
    tweet_user_class = [t + 1 for t in tweet_user_class]

    plt.tick_params(axis='both', labelsize=14)  # 14は目盛りの数字の大きさ
    plt.figure(figsize=(9, 9))
    plt.scatter(range(len(normalized_data)), normalized_data, c=tweet_user_class, cmap='rainbow')
    plt.title('Clustering Results with 2 clusters')
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')
    plt.savefig("plot_mean_anime_weekly_tweet_class_normalized.png")
    plt.close()

    plt.figure(figsize=(9, 9))
    plt.scatter(range(len(mean_anime_weekly_tweet_list)), mean_anime_weekly_tweet_list, c=tweet_user_class, cmap='rainbow')
    plt.title('Clustering Results with 2 clusters')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.savefig(f'plot_mean_anime_weekly_tweet_class_{tweet_clusters}.png')
    plt.close()

    # Plot the hierarchical clustering dendrogram
    linked = linkage(normalized_data, method=agg_clustering.linkage)  # using 'ward' method
    plt.figure(figsize=(9, 9))
    dendrogram(linked, orientation='top', labels=tweet_user_class, distance_sort='descending', show_leaf_counts=True)
    plt.savefig("plot_mean_anime_weekly_tweet_class_kaisou.png")
    plt.close()

    return tweet_user_class, agg_clustering

def cluster_by_dtw(anime_weekly_tweet_series, dtw_clusters):
    # DTW距離行列を計算するのだ
    distance_matrix = cdist_dtw(anime_weekly_tweet_series)
    # fastclusterを使用して階層的クラスタリングを実行するのだ
    linkage_matrix = fastcluster.linkage_vector(distance_matrix, method='ward')
    # クラスタリングの結果を取得するのだ
    dtw_class = fcluster(linkage_matrix, dtw_clusters, criterion='maxclust')

    plt.figure(figsize=(9, 9))
    dendrogram(linkage_matrix, orientation='top', labels=dtw_class, distance_sort='descending', show_leaf_counts=True)
    plt.savefig("plot_anime_weekly_tweet_class_kaisou.png")
    plt.close()

    return dtw_class

def plot_and_save_graphs(title_list, anime_weekly_tweet_series, sorted_cluster_df, dtw_class_label, tweet_user_label, directory):
    plt.figure()
    
    # 指定されたラベルに一致する時系列データを収集する
    matching_series = [anime_weekly_tweet_series[idx].ravel() 
                       for idx, row in sorted_cluster_df.iterrows() 
                       if row['dtw_class'] == dtw_class_label and row['tweet_user_class'] == tweet_user_label]

    print(f'{dtw_class_label}_{tweet_user_label} : {len(matching_series)}')
    
    # しかし、時系列データが10より少ない要素を持っている場合は除外する
    # matching_series = [series for series in matching_series if len(series) >= 10]
    
    # クラスタのバリセンターを計算する
    if len(matching_series) > 0:
        cluster_barycenter = dtw_barycenter_averaging(matching_series)
        # バリセンターを黒色でプロットする
        plt.plot(cluster_barycenter.ravel(), color='black', linewidth=2, label='Cluster Barycenter')
    
    for idx, row in sorted_cluster_df.iterrows():
        if row['dtw_class'] == dtw_class_label and row['tweet_user_class'] == tweet_user_label:
            plt.plot(anime_weekly_tweet_series[idx].ravel())  # 個々の時系列をプロットする
    
    plt.legend()
    plt.title(f'DTW Class {dtw_class_label}, Tweet User Class {tweet_user_label}')
    plt.savefig(f'{directory}/cluster_{dtw_class_label}_{tweet_user_label}_normalized.png')
    plt.close()  # メモリを節約するためにプロットを閉じる

    plt.figure()
    
    # 指定されたラベルに一致する時系列データを収集する
    matching_series = [original_data_list[idx].ravel() 
                       for idx, row in sorted_cluster_df.iterrows() 
                       if row['dtw_class'] == dtw_class_label and row['tweet_user_class'] == tweet_user_label]
    
    # しかし、時系列データが10より少ない要素を持っている場合は除外する
    # matching_series = [series for series in matching_series if len(series) >= 10]
    
    # クラスタのバリセンターを計算する
    if len(matching_series) > 0:
        cluster_barycenter = dtw_barycenter_averaging(matching_series)
        # バリセンターを黒色でプロットする
        plt.plot(cluster_barycenter.ravel(), color='black', linewidth=2, label='Cluster Barycenter')
    
    for idx, row in sorted_cluster_df.iterrows():
        if row['dtw_class'] == dtw_class_label and row['tweet_user_class'] == tweet_user_label:
            plt.plot(original_data_list[idx].ravel())  # 個々の時系列をプロットする
    
    plt.legend()
    plt.title(f'DTW Class {dtw_class_label}, Tweet User Class {tweet_user_label}')
    plt.savefig(f'{directory}/cluster_{dtw_class_label}_{tweet_user_label}_.png')
    plt.close()  # メモリを節約するためにプロットを閉じる


def save_images_to_pdf(directory):
    pdf_filename = 'cluster_plots_dtw_kaisou.pdf'
    
    with PdfPages(pdf_filename) as pdf_pages:
        cluster_path = sorted(glob.glob(f'{directory}/*.png'))
        print(cluster_path)
        for png_file in cluster_path:
            img = plt.imread(png_file)
            plt.figure(figsize=(9, 9))
            plt.imshow(img)
            plt.axis('off')  # 軸を非表示にする
            pdf_pages.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()

def plot_masu(dtw_clusters, tweet_clusters):

    def plot_masu_(filetype):
        # スケーリングしたfigsizeを使ってプロット
        
        fig = plt.figure(figsize=(tweet_clusters*3, dtw_clusters*3 - 1))
        gs = gridspec.GridSpec(dtw_clusters, tweet_clusters, figure=fig, wspace=0.05, hspace=0.05)

        for dtw_class_label in range(1, dtw_clusters + 1):
            for tweet_user_label in range(1, tweet_clusters + 1):
                filename = f'cluster_{dtw_class_label}_{tweet_user_label}_{filetype}.png'
                filepath = os.path.join(directory, filename)
                
                img = plt.imread(filepath)
                
                ax = fig.add_subplot(gs[dtw_class_label - 1, tweet_user_label - 1])
                ax.tick_params(axis='x', labelsize=18)  # 14は目盛りの数字の大きさ
                ax.tick_params(axis='y', labelsize=18)  # 14は目盛りの数字の大きさ
                ax.imshow(img)
                ax.axis('off')

        plt.tight_layout()  # 余白の調整
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        output_filepath = f'result/plot_anime_class_{dtw_clusters}_{tweet_clusters}_{filetype}.png'
        fig.savefig(output_filepath)
        plt.close(fig)
    
    plot_masu_('')
    plot_masu_('normalized')

if __name__ == '__main__':
    directory_path = 'count_tweet'
    dtw_clusters = 3
    tweet_clusters = 3
    for dtw_clusters in range(3, 6):
        for tweet_clusters in range(3, 6):
            anime_weekly_tweet_series, original_data_list, anime_ids, normalized_data = load_and_preprocess_data(directory_path)
            mean_anime_weekly_tweet_list = [x.mean() for x in original_data_list]
            tweet_user_class, agg_clustering = cluster_by_mean_value(mean_anime_weekly_tweet_list, tweet_clusters)
            dtw_class = cluster_by_dtw(anime_weekly_tweet_series, dtw_clusters)
            title_list = get_title(anime_ids)
            sorted_cluster_df = get_result_cluster(anime_ids, title_list, dtw_class, tweet_user_class)
            
            # グラフのプロットと保存
            directory = 'cluster_plots'
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            
            for dtw_class_label in sorted_cluster_df['dtw_class'].unique():
                for tweet_user_label in sorted_cluster_df['tweet_user_class'].unique():
                    plot_and_save_graphs(title_list, anime_weekly_tweet_series, sorted_cluster_df, dtw_class_label, tweet_user_label, directory)

            # PDFへの画像の保存
            save_images_to_pdf(directory)
            plot_masu(dtw_clusters, tweet_clusters)