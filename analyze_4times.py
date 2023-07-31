import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import japanize_matplotlib

def analyze_data(file_path):
    # csvファイルを読み込む
    df = pd.read_csv(file_path)

    # 基本的な統計量を出力する
    print(df.describe())

    # カラムごとに処理
    for column in ['growth_rate_1st_to_2nd', 'growth_rate_2nd_to_3rd']:
        # データの最小値と最大値を取得
        min_val = df[column].min()
        max_val = df[column].max()

        # ヒストグラムのビンの数を計算（例：範囲全体を10刻みで分割）
        bins = range(int(min_val), int(max_val) + 10, 10)

        # データの平均値を計算する
        avg = df[column].mean()
        
        # 値が0以上のデータの平均値を計算する
        avg_over_zero = df[df[column] >= 0][column].mean()

        # ヒストグラムを作成
        plt.figure()
        df[column].hist(bins=bins, alpha=0.7, rwidth=0.85)

        # 平均値をプロット
        plt.axvline(avg, color='blue', linestyle='dashed', linewidth=1, label=f'mean \n{avg}')
        
        # 値が0以上の平均値をプロット
        plt.axvline(avg_over_zero, color='red', linestyle='dashed', linewidth=1, label=f'mean (values >= 0)\n{avg_over_zero}')
        if column == "growth_rate_1st_to_2nd":
            column = "初期から中期"
        elif column == "growth_rate_2nd_to_3rd":
            column = "中期から末期"

        plt.title(f'Histogram of {column}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'growth_rate_histogram_{column}.png')

if __name__ == "__main__":
    analyze_data('count_tweet_4times_growth_rates_and_counts.csv')
