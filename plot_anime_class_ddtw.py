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

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# WCSSを格納するリストを初期化するのだ
wcss = []

# 1から10までのクラスタ数でループを実行するのだ
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(similarity_matrix)
    wcss.append(kmeans.inertia_)

# エルボー法のプロットを作成するのだ
plt.plot(range(1, 11), wcss)
plt.title('エルボー法')
plt.xlabel('クラスタ数')
plt.ylabel('WCSS')  # クラスタ内の平方和
plt.savefig("plot_anime_class_ddtw.png")
