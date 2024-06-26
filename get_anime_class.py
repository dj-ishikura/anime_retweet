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
from sklearn.cluster import KMeans
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42

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
            if (len(df) < 11) or (len(df) > 13):
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
    print(f'閾値：{threshold}')

    # フィルタリング前のアニメ作品の平均週間ツイートユーザ数の分布
    plot_tweet_mean_hist(anime_tweet_data_dict)

    # 閾値以上のツイート数を持つアニメのみをフィルタリング
    top_indices = [i for i, mean_value in enumerate(anime_tweet_data_dict['mean_anime_weekly_tweet_list']) if mean_value >= threshold]

    # 辞書を更新して、上位パーセンテージのデータのみを含むようにする
    anime_tweet_data_dict['anime_ids'] = [anime_tweet_data_dict['anime_ids'][i] for i in top_indices]
    anime_tweet_data_dict['anime_weekly_tweet_list'] = [anime_tweet_data_dict['anime_weekly_tweet_list'][i] for i in top_indices]
    anime_tweet_data_dict['mean_anime_weekly_tweet_list'] = [anime_tweet_data_dict['mean_anime_weekly_tweet_list'][i] for i in top_indices]

    return anime_tweet_data_dict

def plot_tweet_mean_hist(anime_tweet_data_dict):
    averages = anime_tweet_data_dict['mean_anime_weekly_tweet_list']
    plt.figure()
    bins = range(0, 1500, 100)
    plt.hist(averages, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    sorted_averages = sorted(averages)
    median = np.percentile(sorted_averages, 50) # len(averages)の80%に相当する値
    plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label=f'中央値:{median:.1f}')
    
    upper = np.percentile(sorted_averages, 90) # len(averages)の80%に相当する値
    plt.axvline(upper, color='g', linestyle='dashed', linewidth=2, label=f'90パーセントタイル:{upper:.1f}')

    plt.minorticks_on()
    plt.tick_params(axis='both', labelsize=12)
    # Set title and labels
    # plt.title('平均週間ツイートユーザの分布')
    plt.xlabel('平均週間ツイートユーザ数', fontsize=16)
    plt.ylabel('作品数', fontsize=16)
    plt.legend(fontsize=14)
    plt.savefig('plot_tweet_mean_hist_11_13_week_anime.pdf')
    plt.savefig('plot_tweet_mean_hist_11_13_week_anime.png')

def plot_cluster_by_mean_tweet_users(anime_tweet_data_dict):
    # プロットのためにデータとラベルを取得
    mean_values = np.array(anime_tweet_data_dict['mean_anime_weekly_tweet_list'])
    labels = np.array(anime_tweet_data_dict['mean_tweet_user_clusters'])

    # 散布図をプロット
    label_to_text = ["多いクラスタ", "少ないクラスタ", "中くらいクラスタ"]
    plt.figure(figsize=(6.4, 4.8))

    # 表示順序を手動で指定
    for cluster_label in [0, 2, 1]:  # "多い" -> "中くらい" -> "少ない" の順
        plt.scatter(
            x=np.arange(len(mean_values))[labels == cluster_label],
            y=np.array(mean_values)[labels == cluster_label],
            label=label_to_text[cluster_label]
        )
        
    # クラスタ間の線を引くための値を計算
    # "多いクラスタ"と"中くらいクラスタ"の接続点
    high_to_mid_y = (np.min(mean_values[labels == 0]) + np.max(mean_values[labels == 2])) / 2
    # "中くらいクラスタ"と"少ないクラスタ"の接続点
    mid_to_low_y = (np.min(mean_values[labels == 2]) + np.max(mean_values[labels == 1])) / 2

    # x軸の範囲を取得
    x_range = np.arange(len(mean_values))

    # クラスタ間の接続点に線を引く
    plt.plot(x_range, [high_to_mid_y] * len(x_range), 'k--')
    plt.plot(x_range, [mid_to_low_y] * len(x_range), 'k--')

    plt.tick_params(axis='both', labelsize=12)
    plt.xlabel('テレビアニメ作品ID', fontsize=16)
    plt.ylabel('平均週間ツイートユーザ数', fontsize=16)
    plt.legend(fontsize=14, loc='best')
    plt.savefig("plot_anime_class_mean_tweet_users.pdf", bbox_inches='tight')
    plt.savefig("plot_anime_class_mean_tweet_users.png", bbox_inches='tight')


def save_cluster_boundaries(anime_tweet_data_dict):
    cluster_labels = anime_tweet_data_dict['mean_tweet_user_clusters']
    mean_values = anime_tweet_data_dict['mean_anime_weekly_tweet_list']

    # クラスタごとの境界値を計算する
    cluster_boundaries = {}
    for cluster in range(max(cluster_labels) + 1):
        cluster_data = [mean_values[i] for i in range(len(mean_values)) if cluster_labels[i] == cluster]
        cluster_boundaries[cluster] = [min(cluster_data), max(cluster_data)]

    # CSVファイルに保存する
    df = pd.DataFrame.from_dict(cluster_boundaries, orient='index', columns=['min', 'max'])
    df.index.name = 'mean_tweet_user_clusters'
    df.to_csv("anime_class_mean_tweet_users_boudaries.csv")

def cluster_by_mean_tweet_users(anime_tweet_data_dict, mean_tweet_user_class):
    mean_values = anime_tweet_data_dict['mean_anime_weekly_tweet_list']

    # 階層的クラスタリングを実行する
    clustering = AgglomerativeClustering(n_clusters=mean_tweet_user_class, linkage='ward')
    labels = clustering.fit_predict(np.array(mean_values).reshape(-1, 1))

    # クラスタリングの結果を辞書に追加する
    anime_tweet_data_dict['mean_tweet_user_clusters'] = labels

    plot_cluster_by_mean_tweet_users(anime_tweet_data_dict)
    save_cluster_boundaries(anime_tweet_data_dict)

    return anime_tweet_data_dict

def plot_cluster_by_weekly_tweet_users(anime_tweet_data_dict):
    scaled_data = anime_tweet_data_dict['scaled_weekly_tweet_data']
    weekly_clusters = anime_tweet_data_dict['weekly_tweet_user_clusters']

    # 正規化前のプロット
    num_weekly_clusters = len(set(anime_tweet_data_dict['weekly_tweet_user_clusters']))
    label = ["上昇クラスタ", "下降クラスタ", "山型クラスタ", "横ばいクラスタ"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    for i in range(num_weekly_clusters):
        indices = [idx for idx, label in enumerate(weekly_clusters) if label == i]
        cluster_series = [scaled_data[idx] for idx in indices]

        ax = axes[i // 2, i % 2]  # この行を変更

        # サブプロットにデータがある場合のみプロット
        if indices:
            for idx in indices:
                data = anime_tweet_data_dict['anime_weekly_tweet_list'][idx]
                ax.plot(range(1, len(data) + 1), data)
                    
                # 目盛りラベルの設定
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.tick_params(axis='both', which='minor', labelsize=12)
                
                # 軸の範囲を設定
                data_lengths = [len(anime_tweet_data_dict['anime_weekly_tweet_list'][idx]) for idx in indices]
                max_length = max(data_lengths)
                ax.set_xlim([1, max_length])
                
                # y軸の範囲を個別に設定
                all_y_values = [y for idx in indices for y in anime_tweet_data_dict['anime_weekly_tweet_list'][idx]]
                ax.set_ylim([min(all_y_values), max(all_y_values)])
            
            # タイトルとラベルの設定
            ax.set_title(f'{label[i]}', fontsize=20) 
            ax.set_xlabel('放送週', fontsize=18)
            ax.set_ylabel('週間ツイートユーザ数', fontsize=18)

    # グラフ全体の設定
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("plot_anime_class_weekly_tweet_users.pdf", bbox_inches='tight')
    plt.savefig("plot_anime_class_weekly_tweet_users.png", bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    for i in range(num_weekly_clusters):
        indices = [idx for idx, label in enumerate(weekly_clusters) if label == i]
        cluster_series = [scaled_data[idx] for idx in indices]

        ax = axes[i // 2, i % 2]  # この行を変更

        # サブプロットにデータがある場合のみプロット
        if indices:
            for idx in indices:
                data = anime_tweet_data_dict['scaled_weekly_tweet_data'][idx]
                ax.plot(range(1, len(data) + 1), data)
                    
                # 目盛りラベルの設定
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.tick_params(axis='both', which='minor', labelsize=12)
                
                # 軸の範囲を設定
                data_lengths = [len(anime_tweet_data_dict['scaled_weekly_tweet_data'][idx]) for idx in indices]
                max_length = max(data_lengths)
                ax.set_xlim([1, max_length])
                
                # y軸の範囲を個別に設定
                all_y_values = [y for idx in indices for y in anime_tweet_data_dict['scaled_weekly_tweet_data'][idx]]
                ax.set_ylim([min(all_y_values), max(all_y_values)])
            
            # タイトルとラベルの設定
            ax.set_title(f'{label[i]}', fontsize=18) 
            ax.set_xlabel('放送週', fontsize=16)
            ax.set_ylabel('週間ツイートユーザ数', fontsize=16)


    # グラフ全体の設定
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("plot_anime_class_weekly_tweet_users_scaled.png", bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    for i in range(num_weekly_clusters):
        indices = [idx for idx, label in enumerate(weekly_clusters) if label == i]
        cluster_series = [scaled_data[idx] for idx in indices]

        ax = axes[i // 2, i % 2]  # この行を変更

        # サブプロットにデータがある場合のみプロット
        if indices:
            for idx in indices:
                data = anime_tweet_data_dict['transformed_weekly_tweet_data'][idx]
                ax.plot(range(1, len(data) + 1), data)
                    
                # 目盛りラベルの設定
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.tick_params(axis='both', which='minor', labelsize=12)
                
                # 軸の範囲を設定
                data_lengths = [len(anime_tweet_data_dict['transformed_weekly_tweet_data'][idx]) for idx in indices]
                max_length = max(data_lengths)
                ax.set_xlim([1, max_length])
                
                # y軸の範囲を個別に設定
                all_y_values = [y for idx in indices for y in anime_tweet_data_dict['transformed_weekly_tweet_data'][idx]]
                ax.set_ylim([min(all_y_values), max(all_y_values)])
            
            # タイトルとラベルの設定
            ax.set_title(f'{label[i]}', fontsize=18) 
            ax.set_xlabel('X', fontsize=16)
            ax.set_ylabel('週間ツイートユーザ数', fontsize=16)

    # グラフ全体の設定
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("plot_anime_class_weekly_tweet_users_transformed.png", bbox_inches='tight')
    plt.close()

def cluster_by_weekly_tweet_users(anime_tweet_data_dict, weekly_tweet_user_class):
    weekly_tweet_data = anime_tweet_data_dict['anime_weekly_tweet_list']
    resampled_and_normalized_data = []
    resampled_data_for_clustering = []

    scaler = MinMaxScaler()
    for series in weekly_tweet_data:
        series = series[:-1]
        x_original = np.linspace(0, 1, len(series))
        x_resampled = np.linspace(0, 1, 11)
        resampled_series = np.interp(x_resampled, x_original, series)

        normalized_series = scaler.fit_transform(resampled_series.reshape(-1, 1)).ravel()
        # normalized_series = scaler.fit_transform(x_original.reshape(-1, 1)).ravel()
        resampled_and_normalized_data.append(normalized_series)
        # filtered_series = np.delete(normalized_series, [0, -1])
        # filtered_series = np.delete(normalized_series, [])


    """
    for series in weekly_tweet_data:
        filtered_series = np.delete(series, [0, len(series)-1])
        
        x_original = np.linspace(0, 1, len(filtered_series))
        x_resampled = np.linspace(0, 1, 10)
        resampled_series = np.interp(x_resampled, x_original, filtered_series)

        normalized_series = scaler.fit_transform(resampled_series.reshape(-1, 1)).ravel()
        resampled_and_normalized_data.append(normalized_series)

        # クラスタリング用のデータから1週目と11, 12週目を除外
        resampled_data_for_clustering.append(normalized_series)
    """

    anime_tweet_data_dict['scaled_weekly_tweet_data'] = np.array(resampled_and_normalized_data)

    # k-means法でクラスタリング
    # model = AgglomerativeClustering(n_clusters=weekly_tweet_user_class, linkage='average')
    model = TimeSeriesKMeans(n_clusters=weekly_tweet_user_class, metric="dtw", verbose=True, max_iter=10, random_state=42)
    labels = model.fit_predict(anime_tweet_data_dict['scaled_weekly_tweet_data'])
    cluster_centers = model.cluster_centers_
    
    # formatted_data = to_time_series_dataset(resampled_data_for_clustering)
    # labels = model.fit_predict(formatted_data)

    anime_tweet_data_dict['weekly_tweet_user_clusters'] = labels
    anime_tweet_data_dict['transformed_weekly_tweet_data'] = cluster_centers
    
    print(f'scaled_weekly_tweet_data \n{anime_tweet_data_dict["scaled_weekly_tweet_data"]}')
    print(f'transformed_weekly_tweet_data \n{anime_tweet_data_dict["transformed_weekly_tweet_data"]}')

    plot_cluster_by_weekly_tweet_users(anime_tweet_data_dict)
    return anime_tweet_data_dict

def plot_cluster_variability(anime_tweet_data_dict, output_file):
    num_weekly_clusters = len(set(anime_tweet_data_dict['weekly_tweet_user_clusters']))

    fig, axes = plt.subplots(num_weekly_clusters, 1, figsize=(10, num_weekly_clusters * 4), sharex=True)

    all_variability_avgs = []

    for i, cluster in enumerate(sorted(set(anime_tweet_data_dict['weekly_tweet_user_clusters']))):
        indices = [index for index, w_c in enumerate(anime_tweet_data_dict['weekly_tweet_user_clusters']) if w_c == cluster]
        cluster_data = np.array([anime_tweet_data_dict['scaled_weekly_tweet_data'][idx] for idx in indices])

        variability = np.std(cluster_data, axis=0)
        avg_variability = np.mean(variability)
        all_variability_avgs.append(avg_variability)

        ax = axes[i] if num_weekly_clusters > 1 else axes
        ax.plot(range(1, len(variability) + 1), variability, label='Standard Deviation')
        ax.axhline(y=avg_variability, color='r', linestyle='-', label='Average Std Dev')
        ax.axvline(x=5, color='g', linestyle='--')  # 例として第5週を示す縦線
        
        ax.set_title(f'Cluster {cluster}')
        ax.set_xlabel('Week (Starting from Week 1)')
        ax.set_ylabel('Standard Deviation of Tweet Users Count')
        ax.set_ylim([0, 0.4])  # y軸の範囲を0から0.4に固定
        ax.legend()

    plt.tight_layout()
    plt.suptitle('Variability in Weekly Tweet User Clusters with Average', fontsize=16)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

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
        'mean_tweet_user_count': anime_tweet_data_dict['mean_anime_weekly_tweet_list'],
        'weekly_tweet_user_clusters': anime_tweet_data_dict['weekly_tweet_user_clusters'],
        'mean_tweet_user_clusters': anime_tweet_data_dict['mean_tweet_user_clusters']
    })
    
    # CSVファイルに保存
    results_df.to_csv(filename, index=False)

def plot_and_save_all_clusters_scaled(anime_tweet_data_dict, output_file, weekly_tweet_user_class, mean_tweet_user_class):
    scaled_data = anime_tweet_data_dict['scaled_weekly_tweet_data']
    weekly_clusters = anime_tweet_data_dict['weekly_tweet_user_clusters']

    # グラフを描画
    fig, axes = plt.subplots(weekly_tweet_user_class, mean_tweet_user_class, figsize=(15, 8), sharex=True, sharey=True)

    # weekly_tweet_user_clustersごとにサブプロットを作成
    for i in range(weekly_tweet_user_class):
        # 指定したweekly_clusterに属する時系列データのインデックスを取得
        indices = [idx for idx, label in enumerate(weekly_clusters) if label == i]
        cluster_series = [scaled_data[idx] for idx in indices]

        # クラスタの重心を計算
        # barycenter = TimeSeriesKMeans(n_clusters=1).fit(cluster_series).cluster_centers_[0]

        for j in range(mean_tweet_user_class):
            ax = axes[i, j] if weekly_tweet_user_class > 1 and mean_tweet_user_class > 1 else axes[i]

            # 各mean_tweet_user_clusterに属する時系列データをプロット
            for idx in indices:
                if anime_tweet_data_dict['mean_tweet_user_clusters'][idx] == j:
                    # x軸の値を1から始めるように変更
                    ax.plot(range(1, len(scaled_data[idx]) + 1), scaled_data[idx].ravel(), alpha=0.8, label=f'Series {idx}')

            # ax.plot(range(1, len(barycenter) + 1), barycenter.ravel(), "k-", linewidth=2, alpha=1, label='Barycenter')

            
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

    w_label = ["上昇", "下降", "山型", "横ばい"]
    m_label = ["多い", "少ない", "中くらい"]

    fig, axes = plt.subplots(num_weekly_clusters, num_mean_clusters, figsize=(15, 8), sharex=False, sharey=False)

    for i, w_cluster in enumerate(sorted(set(anime_tweet_data_dict['weekly_tweet_user_clusters']))):
        for j, m_cluster in enumerate(sorted(set(anime_tweet_data_dict['mean_tweet_user_clusters']))):
            ax = axes[i][j] if num_weekly_clusters > 1 and num_mean_clusters > 1 else axes[max(i, j)]
            indices = [index for index, (w_c, m_c) in enumerate(zip(anime_tweet_data_dict['weekly_tweet_user_clusters'], anime_tweet_data_dict['mean_tweet_user_clusters'])) if w_c == w_cluster and m_c == m_cluster]
            
            # サブプロットにデータがある場合のみプロット
            if indices:
                for idx in indices:
                    data = anime_tweet_data_dict['anime_weekly_tweet_list'][idx]
                    ax.plot(range(1, len(data) + 1), data)
                    
                # 目盛りラベルの設定
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.tick_params(axis='both', which='minor', labelsize=8)
                
                # 軸の範囲を設定
                data_lengths = [len(anime_tweet_data_dict['anime_weekly_tweet_list'][idx]) for idx in indices]
                max_length = max(data_lengths)
                ax.set_xlim([1, max_length])
                
                # y軸の範囲を個別に設定
                all_y_values = [y for idx in indices for y in anime_tweet_data_dict['anime_weekly_tweet_list'][idx]]
                ax.set_ylim([min(all_y_values), max(all_y_values)])
            
            # タイトルとラベルの設定
            ax.set_title(f'{w_label[w_cluster]} - {m_label[m_cluster]}', fontsize=18)
            ax.set_xlabel('放送週', fontsize=12)
            ax.set_ylabel('週間ツイートユーザ数', fontsize=12)

    # グラフ全体の設定
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def main():
    mean_tweet_user_class = 3
    weekly_tweet_user_class = 4
    directory_path = 'count_tweet'
    anime_tweet_data_dict = get_data(directory_path)
    print(f'放送週が11-13週のアニメ数 : {len(anime_tweet_data_dict["anime_weekly_tweet_list"])}')
    # 上位数%を取得
    rank_percent = 10
    anime_tweet_data_dict = filter_top_percentage_data(anime_tweet_data_dict, rank_percent)
    print(f'上位 {rank_percent} % のアニメ数 : {len(anime_tweet_data_dict["anime_weekly_tweet_list"])}')

    anime_tweet_data_dict = cluster_by_mean_tweet_users(anime_tweet_data_dict, mean_tweet_user_class)
    anime_tweet_data_dict = cluster_by_weekly_tweet_users(anime_tweet_data_dict, weekly_tweet_user_class)

    save_clustering_results_to_csv(anime_tweet_data_dict, 'anime_class.csv')
    plot_and_save_all_clusters(anime_tweet_data_dict, f'plot_anime_class_w{weekly_tweet_user_class}_m{mean_tweet_user_class}.png')
    plot_and_save_all_clusters_scaled(anime_tweet_data_dict, f'plot_anime_class_scaled_w{weekly_tweet_user_class}_m{mean_tweet_user_class}.png'
    ,weekly_tweet_user_class,mean_tweet_user_class)

if __name__ == "__main__":
    main()
