import pandas as pd
import os
from multiprocessing import Pool
from itertools import combinations
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties # 日本語対応
import japanize_matplotlib
import ddtw

# データを格納するディレクトリのパス
directory_path = 'count_tweet'

# データを格納する辞書
data_dict = {}

# ディレクトリ内の各ファイルを処理するが、最初の10件のファイルのみを処理する
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        anime_id = filename.split('_')[0]
        file_path = os.path.join(directory_path, filename)
        data_dict[anime_id] = pd.read_csv(file_path, index_col=0)

print(len(data_dict))
all_data = pd.concat([data[['tweet_users_count']] for data in data_dict.values()])
scaler = MinMaxScaler().fit(all_data)

def calculate_distance(pair):
    name1, data1, name2, data2 = pair
    # スケーラを使用してdata1とdata2のtweet_users_count列を正規化する
    # 正規化されたデータを使用して距離を計算する
    distance, _ = fastdtw(data1[['tweet_users_count']], data2[['tweet_users_count']], dist=euclidean)
    return ((name1, name2), distance)

def calculate_distance_ddtw(pair):
    name1, data1, name2, data2 = pair
    
    data1_values = data1[['tweet_users_count']].values.reshape(-1, 1)
    data2_values = data2[['tweet_users_count']].values.reshape(-1, 1)
    
    # スケーラを使用してdata1とdata2のtweet_users_count列を正規化する
    # data1_scaled = scaler.fit_transform(data1_values).reshape(-1)
    # data2_scaled = scaler.fit_transform(data2_values).reshape(-1)

    try:
        # 正規化されたデータを使用して距離を計算する
        _, _, distance = ddtw.DDTW(data1_values, data2_values)
        
    except AssertionError as e:
        # print(f"AssertionError: {e}, pair: {name1, name2}")  # この行をコメントアウトまたは削除するのだ
        distance = None  # または、他の適切なデフォルト値を設定する
    return ((name1, name2), distance)
    

# すべての組み合わせを生成する
pairs = [(name1, data1, name2, data2) for (name1, data1), (name2, data2) in combinations(data_dict.items(), 2)]

# プロセスプールを作成し、並列に距離を計算する

with Pool() as pool:
    results = pool.map(calculate_distance_ddtw, pairs)

# 無効な値を持つペアを除外する
similarity_dict = {pair: distance for pair, distance in results if distance is not None}

# 有効なペアからアニメのIDのリストを作成する
anime_ids = set()
for pair in similarity_dict.keys():
    anime_ids.update(pair)

# アニメ作品の数を取得する
num_anime = len(anime_ids)

# 有効なアニメIDを辞書にマッピングしてインデックスを取得する
anime_id_to_index = {anime_id: index for index, anime_id in enumerate(sorted(anime_ids))}
print(anime_id_to_index)

# 類似度行列を初期化する (0で埋める)
similarity_matrix = np.zeros((num_anime, num_anime))

# 類似度辞書から類似度行列を満たす
for (id1, id2), distance in similarity_dict.items():
    i, j = anime_id_to_index[id1], anime_id_to_index[id2]
    similarity_matrix[i, j] = distance
    similarity_matrix[j, i] = distance  # 類似度行列は対称であることを保証する

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

print("similarity_matrix")
print(similarity_matrix)
print(similarity_matrix.shape)
print(np.allclose(similarity_matrix, similarity_matrix.T))


# squareform関数で正方行列をcondensed形式に変換するのだ
condensed_matrix = squareform(similarity_matrix, checks=False)
print("condensed_matrix")
print(condensed_matrix)

linked = linkage(condensed_matrix, 'single')

# デンドログラムを描画する
dendrogram(linked, orientation='top')
plt.savefig("plot_anime_class_ddtw.png")  # 少し修正して、plgではなくpltを使用するのだ

from scipy.cluster.hierarchy import fcluster

# 階層的クラスタリングの結果からクラスターラベルを取得する
# ここでは、t=1.0 という閾値を使用しているが、適切な閾値に調整することができるのだ
# cluster_labels = fcluster(linked, t=4, criterion='maxclust')
cluster_labels = fcluster(linked, t=2000, criterion='distance')

df_anime = pd.read_csv('./anime_data_updated.csv', index_col=0)
title_list = []
for id in anime_ids:
    title = df_anime.loc[id, '作品名']
    title_list.append(title)

# クラスターラベルをデータフレームに追加する
cluster_df = pd.DataFrame({'id': list(anime_ids), 'title': title_list, 'Cluster': cluster_labels})
# クラスターラベルでデータフレームをソートする
sorted_cluster_df = cluster_df.sort_values(by='Cluster')

# ソートされたデータフレームをCSVファイルに保存する
sorted_cluster_df.to_csv('anime_class_ddtw.csv', index=False)

def save_cluster_graphs_to_pdf(cluster_df):
    # クラスタリング結果から各クラスタのアニメIDを取得する
    clusters = cluster_df.groupby('Cluster')['id'].apply(list)

    # 出力ディレクトリを作成する (存在しない場合)
    output_dir = 'cluster_pdfs'
    os.makedirs(output_dir, exist_ok=True)

    for cluster_num, anime_ids in clusters.items():
        # 各クラスタごとにPDFファイルを作成する
        pdf_filename = os.path.join(output_dir, f'cluster_{cluster_num}.pdf')
        with PdfPages(pdf_filename) as pdf:
            for anime_id in anime_ids:
                # グラフのPNGファイル名を作成する
                graph_filename = f"count_tweet/{anime_id}_1_week_tweet_counts.png"
                
                # グラフのPNGファイルを読み込む
                img = plt.imread(graph_filename)
                
                # グラフを表示する
                plt.imshow(img)
                
                # グラフをPDFファイルに保存する
                pdf.savefig()  
                plt.close()  # 現在の図をクローズする

def save_cluster_graphs_to_png(cluster_df):
    # クラスタリング結果から各クラスタのアニメIDを取得する
    clusters = cluster_df.groupby('Cluster')['id'].apply(list)

    # 出力ディレクトリを作成する (存在しない場合)
    output_dir = 'cluster_pngs_ddtw'
    os.makedirs(output_dir, exist_ok=True)

    for cluster_num, anime_ids in clusters.items():
        # 新しいフィギュアを作成する
        fig, ax = plt.subplots(figsize=(10, 8))

        for anime_id in anime_ids:
            # データを取得する
            data = data_dict[anime_id]
            # グラフをプロットする
            # x軸に番号を付ける (日付の代わりに)
            ax.plot(range(len(data)), data['tweet_users_count'], label=cluster_df[cluster_df['id'] == anime_id]['title'].values[0])

        # 凡例を表示する
        ax.legend()

        # グラフのタイトルを設定する
        ax.set_title(f'Cluster {cluster_num}')

        # グラフをPNGファイルに保存する
        png_filename = os.path.join(output_dir, f'cluster_{cluster_num}.png')
        plt.savefig(png_filename)
        plt.close(fig)  # 現在の図をクローズする

# この関数を呼び出してPNGファイルを作成する
save_cluster_graphs_to_png(sorted_cluster_df)


# save_cluster_graphs_to_pdf(cluster_df)