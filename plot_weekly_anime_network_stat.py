import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

def get_info_from_csv(id):
    # CSVファイルを読み込みます
    df = pd.read_csv('./anime_data_updated.csv', index_col=0)

    # numberをインデックスとしてキーワードを取得します
    title = df.loc[id, '作品名']

    return title

def plot_weeekly_network_stat(csv_file_path, output_png):
    # CSVファイルを読み込むのだ
    data = pd.read_csv(csv_file_path, index_col=0)
    id = os.path.basename(csv_file_path).replace('.csv', '')
    title = get_info_from_csv(id)

    # グラフのレイアウトを設定するのだ
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    fig.suptitle(f'{title}, {id}')

    # 各統計情報を異なるサブプロットに描画するのだ
    for i, column in enumerate(data.columns):
        ax = axes[i // 3, i % 3]
        ax.plot(data.index, data[column], label=column, marker='o')
        ax.set_title(column)
        ax.set_xlabel('Week')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()

    # グラフのレイアウトを調整するのだ
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # グラフを画像として保存するのだ
    plt.savefig(output_png)

    # グラフを表示するのだ（必要に応じて）
    # plt.show()

if __name__ == "__main__":
    plot_weeekly_network_stat(sys.argv[1], sys.argv[2])