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
import warnings
from sklearn.exceptions import ConvergenceWarning
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.barycenters import dtw_barycenter_averaging

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

def filter_top_percentage_data(anime_tweet_data_dict, top_percentage):
    # 平均ツイート数の上位パーセンテージに相当する値を計算
    threshold = np.percentile(anime_tweet_data_dict['mean_anime_weekly_tweet_list'], 100 - top_percentage)

    # 閾値以上のツイート数を持つアニメのみをフィルタリング
    top_indices = [i for i, mean_value in enumerate(anime_tweet_data_dict['mean_anime_weekly_tweet_list']) if mean_value >= threshold]

    # 辞書を更新して、上位パーセンテージのデータのみを含むようにする
    anime_tweet_data_dict['anime_ids'] = [anime_tweet_data_dict['anime_ids'][i] for i in top_indices]
    anime_tweet_data_dict['anime_weekly_tweet_list'] = [anime_tweet_data_dict['anime_weekly_tweet_list'][i] for i in top_indices]
    anime_tweet_data_dict['mean_anime_weekly_tweet_list'] = [anime_tweet_data_dict['mean_anime_weekly_tweet_list'][i] for i in top_indices]

    return anime_tweet_data_dict

from sklearn.cluster import KMeans

def cluster_time_series(anime_tweet_data_dict, n_clusters, resample_length=12):
    def differencing(series):
        # 1時点前のデータとの差分を取る
        diff_series = np.diff(series, n=1)
        return diff_series

    # リサンプリングされた時系列データのリストを作成
    resampled_data = []

    for series in anime_tweet_data_dict['anime_weekly_tweet_list']:
        # リサンプリング（各時系列を同じ長さに揃える）
        x_original = np.linspace(0, 1, len(series))
        x_resampled = np.linspace(0, 1, resample_length)
        resampled_series = np.interp(x_resampled, x_original, series)
        resampled_data.append(resampled_series)

    # 差分変換を適用
    diff_data = np.array([differencing(series) for series in resampled_data])

    # クラスタリングモデルを作成し、フィットさせる
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(diff_data)

    # クラスタリングの結果を辞書に追加
    anime_tweet_data_dict['weekly_tweet_user_clusters'] = labels

    return anime_tweet_data_dict

def cluster_time_series_kshape(anime_tweet_data_dict, n_clusters, resample_length=12):
    # リサンプリングして時系列データのリストを作成
    resampled_data = []
    for series in anime_tweet_data_dict['anime_weekly_tweet_list']:
        # 12週間にリサンプリング
        x_original = np.linspace(0, 1, len(series))
        x_resampled = np.linspace(0, 1, resample_length)
        resampled_series = np.interp(x_resampled, x_original, series)


        # 正規化
        normalized_series = MinMaxScaler().fit_transform(resampled_series.reshape(-1, 1)).ravel()
        resampled_data.append(normalized_series)

    # すべての時系列を同じ長さに揃えたデータセットを作成
    resampled_data = np.array(resampled_data)

    # 時系列データをスケーリング
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # z-normalize the time series
    scaled_data = scaler.fit_transform(resampled_data)

    # kshapeクラスタリングモデルを作成し、フィットさせる
    ks = KShape(n_clusters=n_clusters, n_init=1, random_state=42)
    labels = ks.fit_predict(scaled_data)

    # クラスタリングの結果を辞書に追加
    anime_tweet_data_dict['weekly_tweet_user_clusters'] = labels

    return anime_tweet_data_dict

def cluster_time_series_with_features(anime_tweet_data_dict, n_clusters, resample_length=12):
    # リサンプリングして時系列データのリストを作成
    resampled_data = []
    for series in anime_tweet_data_dict['anime_weekly_tweet_list']:
        # 12週間にリサンプリング
        x_original = np.linspace(0, 1, len(series))
        x_resampled = np.linspace(0, 1, resample_length)
        resampled_series = np.interp(x_resampled, x_original, series)
        resampled_data.append(resampled_series)
    
    # DataFrame形式で時系列データを整形
    time_series_df = pd.DataFrame({
        'id': np.repeat(np.arange(len(resampled_data)), resample_length),
        'time': np.tile(np.arange(resample_length), len(resampled_data)),
        'value': np.concatenate(resampled_data)
    })

    # 特徴抽出（警告が出る可能性があるため、警告を無視）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # EfficientFCParameters() を使用して効率的な特徴量設定を取得
        extraction_settings = EfficientFCParameters()
        extracted_features = extract_features(time_series_df, column_id='id', column_sort='time',
                                              default_fc_parameters=extraction_settings)


    # NaNを含む列を削除
    extracted_features.dropna(axis=1, inplace=True)

    # K-Meansクラスタリングを実行
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(extracted_features)

    # クラスタリングの結果を辞書に追加
    anime_tweet_data_dict['weekly_tweet_user_clusters'] = labels

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

    # MinMaxScalerのインスタンスを作成（全てのデータに対して1つのインスタンスを使用）
    scaler = MinMaxScaler()

    for series in weekly_tweet_data:
        # 12週間にリサンプリング
        x_original = np.linspace(0, 1, len(series))
        x_resampled = np.linspace(0, 1, 12)
        resampled_series = np.interp(x_resampled, x_original, series)

        # 正規化（0から1の範囲）
        normalized_series = scaler.fit_transform(resampled_series.reshape(-1, 1)).ravel()
        resampled_and_normalized_data.append(normalized_series)

    # 正規化されたデータをディクショナリに追加
    anime_tweet_data_dict['scaled_weekly_tweet_data'] = np.array(resampled_and_normalized_data)

    # DTWを使った時系列クラスタリング
    model = TimeSeriesKMeans(n_clusters=weekly_tweet_user_class, metric="dtw", verbose=True, max_iter=10)
    labels = model.fit_predict(anime_tweet_data_dict['scaled_weekly_tweet_data'])

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

def plot_and_save_all_clusters_scaled(anime_tweet_data_dict, output_file, weekly_tweet_user_class, mean_tweet_user_class):
    scaled_data = anime_tweet_data_dict['scaled_weekly_tweet_data']

    # グラフを描画
    fig, axes = plt.subplots(weekly_tweet_user_class, mean_tweet_user_class, figsize=(15, 8), sharex=True, sharey=True)

    # 各クラスタに属する時系列をプロット
    for i in range(weekly_tweet_user_class):
        for j in range(mean_tweet_user_class):
            ax = axes[i, j] if weekly_tweet_user_class > 1 and mean_tweet_user_class > 1 else axes[i]
            
            # 指定したクラスタに属する時系列データのインデックスを取得
            indices = [idx for idx, label in enumerate(anime_tweet_data_dict['weekly_tweet_user_clusters']) if label == i]

            # クラスタ内の時系列データのリストを取得
            cluster_series = [scaled_data[idx] for idx in indices if anime_tweet_data_dict['mean_tweet_user_clusters'][idx] == j]
            
            # 各時系列データをプロット
            for idx in indices:
                if anime_tweet_data_dict['mean_tweet_user_clusters'][idx] == j:
                    ax.plot(scaled_data[idx].ravel(),alpha=0.8, label=f'Series {idx}')

            # クラスタのバリセンターを計算
            if cluster_series:  # クラスタにデータが存在する場合のみ計算
                barycenter = dtw_barycenter_averaging(cluster_series)
                ax.plot(barycenter.ravel(), "k-", linewidth=2, alpha=1, label='Barycenter')
            
            # タイトルとラベルの設定
            ax.set_title(f'Weekly Cluster {i} - Mean Cluster {j}')
            ax.set_xlabel('Week')
            ax.set_ylabel('Normalized Tweet Users Count')

    # グラフ全体の設定
    plt.tight_layout()
    plt.suptitle('Weekly and Mean Tweet User Clusters with Barycenters', fontsize=16)
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
    anime_tweet_data_dict = filter_top_percentage_data(anime_tweet_data_dict, 20)

    anime_tweet_data_dict = cluster_by_mean_tweet_users(anime_tweet_data_dict, mean_tweet_user_class)
    anime_tweet_data_dict = cluster_by_weekly_tweet_users(anime_tweet_data_dict, weekly_tweet_user_class)

    save_clustering_results_to_csv(anime_tweet_data_dict, 'anime_class.csv')
    plot_and_save_all_clusters(anime_tweet_data_dict, f'plot_anime_class_w{weekly_tweet_user_class}_m{mean_tweet_user_class}.png')
    plot_and_save_all_clusters_scaled(anime_tweet_data_dict, f'plot_anime_class_scaled_w{weekly_tweet_user_class}_m{mean_tweet_user_class}.png'
    ,weekly_tweet_user_class,mean_tweet_user_class)

if __name__ == "__main__":
    main()
