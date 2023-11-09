from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
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
from matplotlib import gridspec

def get_data(directory_path):
    anime_tweet_data_dict = {}
    anime_weekly_tweet_list = []
    mean_anime_weekly_tweet_list = []
    anime_ids = []

    # 指定されたディレクトリ内のCSVファイルを読み込む
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path, index_col=0)

            # データの長さが1の場合はスキップ
            if len(df) < 11:
                continue

            anime_id = filename.split('_')[0]
            anime_ids.append(anime_id)
            anime_weekly_tweet_list.append(df["tweet_users_count"].tolist())
            mean_anime_weekly_tweet_list.append(df["tweet_users_count"].mean())

    anime_tweet_data_dict['anime_ids'] = anime_ids
    anime_tweet_data_dict['anime_weekly_tweet_list'] = anime_weekly_tweet_list
    anime_tweet_data_dict['mean_anime_weekly_tweet_list'] = mean_anime_weekly_tweet_list

    return anime_tweet_data_dict

def cluster_by_mean_tweet_users(anime_tweet_data_dict, mean_tweet_user_class):
    mean_values = anime_tweet_data_dict['mean_anime_weekly_tweet_list']

    # 階層的クラスタリングを実行する
    clustering = AgglomerativeClustering(n_clusters=mean_tweet_user_class, linkage='ward')
    labels = clustering.fit_predict(np.array(mean_values).reshape(-1, 1))

    # クラスタリングの結果を辞書に追加する
    anime_tweet_data_dict['mean_tweet_user_clusters'] = labels

    return anime_tweet_data_dict

def cluster_by_weekly_tweet_users(anime_tweet_data_dict, weekly_tweet_user_class):
    # 週ごとのツイート数のリストを取得
    weekly_tweet_data = anime_tweet_data_dict['anime_weekly_tweet_list']
    
    # リサンプリングされた時系列データのリストを作成
    resampled_and_normalized_data = []

    for series in weekly_tweet_data:
        # 12週間にリサンプリング
        x_original = np.linspace(0, 1, len(series))
        x_resampled = np.linspace(0, 1, 12)
        resampled_series = np.interp(x_resampled, x_original, series)

        # 正規化
        normalized_series = MinMaxScaler().fit_transform(resampled_series.reshape(-1, 1)).ravel()
        resampled_and_normalized_data.append(normalized_series)

    # スケーリングされたデータを時系列データセットに変換
    scaled_weekly_tweet_data = TimeSeriesScalerMeanVariance().fit_transform(np.array(resampled_and_normalized_data).reshape(len(resampled_and_normalized_data), -1, 1))

    # DTWを使った時系列クラスタリング
    model = TimeSeriesKMeans(n_clusters=weekly_tweet_user_class, metric="dtw", verbose=True, max_iter=10)
    labels = model.fit_predict(scaled_weekly_tweet_data)

    # クラスタリングの結果を辞書に追加
    anime_tweet_data_dict['weekly_tweet_user_clusters'] = labels

    return anime_tweet_data_dict

def save_clustering_results_to_csv(anime_tweet_data_dict, filename):

    # アニメタイトルの取得
    df_anime = pd.read_csv('./anime_data_updated.csv', index_col=0)
    title_list = []
    for id in anime_tweet_data_dict['anime_ids']:
        title = df_anime.loc[id, '作品名']
        title_list.append(title)

    # アニメのID、週ごとのツイートユーザ数のクラスタラベル、平均ツイートユーザ数のクラスタラベルを含むDataFrameを作成
    results_df = pd.DataFrame({
        'id': anime_tweet_data_dict['anime_ids'],
        'title': title_list,
        'weekly_tweet_user_clusters': anime_tweet_data_dict['weekly_tweet_user_clusters'],
        'mean_tweet_user_clusters': anime_tweet_data_dict['mean_tweet_user_clusters']
    })
    
    # CSVファイルに保存
    results_df.to_csv(filename, index=False)

def plot_and_save_all_clusters_scaled(anime_tweet_data_dict, output_file):
    # 週ごとのツイート数のリストを取得
    weekly_tweet_data = anime_tweet_data_dict['anime_weekly_tweet_list']

    # 正規化された時系列データのリストを作成
    normalized_data = []

    # MinMaxScalerのインスタンスを作成
    scaler = MinMaxScaler()

    # 全ての時系列データを正規化
    for series in weekly_tweet_data:
        # 12週間にリサンプリング（もし必要なら）
        x_original = np.linspace(0, 1, len(series))
        x_resampled = np.linspace(0, 1, 12)
        resampled_series = np.interp(x_resampled, x_original, series)

        # 正規化
        normalized_series = scaler.fit_transform(resampled_series.reshape(-1, 1)).ravel()
        normalized_data.append(normalized_series)

    # 正規化されたデータをディクショナリに追加
    anime_tweet_data_dict['scaled_weekly_tweet_data'] = normalized_data

    num_weekly_clusters = len(set(anime_tweet_data_dict['weekly_tweet_user_clusters']))
    num_mean_clusters = len(set(anime_tweet_data_dict['mean_tweet_user_clusters']))

    fig, axes = plt.subplots(num_weekly_clusters, num_mean_clusters, figsize=(15, 8), sharex=False, sharey=False)

    for i, w_cluster in enumerate(sorted(set(anime_tweet_data_dict['weekly_tweet_user_clusters']))):
        for j, m_cluster in enumerate(sorted(set(anime_tweet_data_dict['mean_tweet_user_clusters']))):
            ax = axes[i][j] if num_weekly_clusters > 1 and num_mean_clusters > 1 else axes[max(i, j)]
            indices = [index for index, (w_c, m_c) in enumerate(zip(anime_tweet_data_dict['weekly_tweet_user_clusters'], anime_tweet_data_dict['mean_tweet_user_clusters'])) if w_c == w_cluster and m_c == m_cluster]
            
            # サブプロットにデータがある場合のみプロット
            if indices:
                for idx in indices:
                    data = anime_tweet_data_dict['scaled_weekly_tweet_data'][idx]
                    ax.plot(data)
                    
                # 目盛りラベルの設定
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.tick_params(axis='both', which='minor', labelsize=8)
                
                # 軸の範囲を設定
                data_lengths = [len(anime_tweet_data_dict['scaled_weekly_tweet_data'][idx]) for idx in indices]
                max_length = max(data_lengths)
                ax.set_xlim([0, max_length - 1])
                
                # y軸の範囲を個別に設定
                all_y_values = [y for idx in indices for y in anime_tweet_data_dict['scaled_weekly_tweet_data'][idx]]
                ax.set_ylim([min(all_y_values), max(all_y_values)])
            
            # タイトルとラベルの設定
            ax.set_title(f'Cluster W{w_cluster} - M{m_cluster}')
            ax.set_xlabel('Week')
            ax.set_ylabel('Tweet Users Count')

    # グラフ全体の設定
    plt.tight_layout()
    plt.suptitle('Weekly Tweet User Clusters vs Mean Tweet User Clusters', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def plot_and_save_all_clusters(anime_tweet_data_dict, output_file):
    num_weekly_clusters = len(set(anime_tweet_data_dict['weekly_tweet_user_clusters']))
    num_mean_clusters = len(set(anime_tweet_data_dict['mean_tweet_user_clusters']))

    fig, axes = plt.subplots(num_weekly_clusters, num_mean_clusters, figsize=(15, 8), sharex=False, sharey=False)

    for i, w_cluster in enumerate(sorted(set(anime_tweet_data_dict['weekly_tweet_user_clusters']))):
        for j, m_cluster in enumerate(sorted(set(anime_tweet_data_dict['mean_tweet_user_clusters']))):
            ax = axes[i][j] if num_weekly_clusters > 1 and num_mean_clusters > 1 else axes[max(i, j)]
            indices = [index for index, (w_c, m_c) in enumerate(zip(anime_tweet_data_dict['weekly_tweet_user_clusters'], anime_tweet_data_dict['mean_tweet_user_clusters'])) if w_c == w_cluster and m_c == m_cluster]
            
            # サブプロットにデータがある場合のみプロット
            if indices:
                for idx in indices:
                    data = anime_tweet_data_dict['anime_weekly_tweet_list'][idx]
                    ax.plot(data)
                    
                # 目盛りラベルの設定
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.tick_params(axis='both', which='minor', labelsize=8)
                
                # 軸の範囲を設定
                data_lengths = [len(anime_tweet_data_dict['anime_weekly_tweet_list'][idx]) for idx in indices]
                max_length = max(data_lengths)
                ax.set_xlim([0, max_length - 1])
                
                # y軸の範囲を個別に設定
                all_y_values = [y for idx in indices for y in anime_tweet_data_dict['anime_weekly_tweet_list'][idx]]
                ax.set_ylim([min(all_y_values), max(all_y_values)])
            
            # タイトルとラベルの設定
            ax.set_title(f'Cluster W{w_cluster} - M{m_cluster}')
            ax.set_xlabel('Week')
            ax.set_ylabel('Tweet Users Count')

    # グラフ全体の設定
    plt.tight_layout()
    plt.suptitle('Weekly Tweet User Clusters vs Mean Tweet User Clusters', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def main():
    mean_tweet_user_class = 3
    weekly_tweet_user_class = 3
    directory_path = 'count_tweet'
    anime_tweet_data_dict = get_data(directory_path)
    anime_tweet_data_dict = cluster_by_mean_tweet_users(anime_tweet_data_dict, mean_tweet_user_class)
    anime_tweet_data_dict = cluster_by_weekly_tweet_users(anime_tweet_data_dict, weekly_tweet_user_class)

    save_clustering_results_to_csv(anime_tweet_data_dict, 'anime_class.csv')
    plot_and_save_all_clusters(anime_tweet_data_dict, f'plot_anime_class_w{weekly_tweet_user_class}_m{mean_tweet_user_class}.png')
    plot_and_save_all_clusters_scaled(anime_tweet_data_dict, f'plot_anime_class_scaled_w{weekly_tweet_user_class}_m{mean_tweet_user_class}.png')

if __name__ == "__main__":
    main()
