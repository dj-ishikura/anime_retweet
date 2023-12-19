from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform

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

def evaluate_clustering(n_clusters_range, X):
    sse = []  # Sum of Squared Errors
    
    for n_clusters in n_clusters_range:
        dba_km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", verbose=False)
        pred = dba_km.fit_predict(X)
        
        # SSEを計算する
        sse.append(dba_km.inertia_)
    
    return sse

def plot_evaluation(n_clusters_range, sse, save_path):
    plt.figure(figsize=(10, 5))
    
    plt.plot(n_clusters_range, sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    
    plt.tight_layout()
    plt.savefig(save_path)  # 画像を保存するのだ
    plt.close()  # メモリを節約するためにプロットを閉じるのだ


# クラスタ数の範囲を指定する
n_clusters_range = range(1, 11)

# クラスタリングの評価を行う
sse = evaluate_clustering(n_clusters_range, anime_weekly_tweet_series)

# 評価結果をプロットし、画像として保存する
save_path = os.path.join('plots', 'evaluation_plot.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ディレクトリを作成するのだ
plot_evaluation(n_clusters_range, sse, save_path)
