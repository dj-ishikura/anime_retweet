import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# グラフのディレクトリを指定する
directory = 'cluster_plots_dtw_kaisou'

# 9x9のグリッドを作成する
fig = plt.figure(figsize=(15, 12))  # figsizeを調整する
gs = gridspec.GridSpec(9, 9, figure=fig, wspace=0.05, hspace=0.05)  # wspaceとhspaceを設定する


# 各マスにグラフを配置する
for class_label in range(1, 4):  # Class labels: 1, 2, 3
    for tweet_user_label in range(1, 4):  # Tweet User Class labels: 1, 2, 3
        # ファイル名を作成する
        filename = f'cluster_{class_label}_{tweet_user_label}_.png'
        filepath = os.path.join(directory, filename)
        
        # グラフを読み込む
        img = plt.imread(filepath)
        
        # グラフの位置を計算する
        row = (class_label - 1) * 3
        col = (tweet_user_label - 1) * 3
        
        # グラフをグリッドに配置する
        ax = fig.add_subplot(gs[row:row+3, col:col+3])
        ax.imshow(img)
        ax.axis('off')  # 軸を非表示にする

# 画像を保存する
output_filepath = 'result/plot_anime_class_dtw_kaisou_.png'
fig.savefig(output_filepath)
plt.close(fig)  # メモリを節約するためにプロットを閉じる
