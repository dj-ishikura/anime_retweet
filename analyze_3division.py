import pandas as pd
import matplotlib.pyplot as plt

def analyze_data(file_path):
    # csvファイルを読み込む
    df = pd.read_csv(file_path)

    # 基本的な統計量を出力する
    print(df.describe())

    # ヒストグラムを作成する
    plt.figure()
    bins = range(-100, 130, 10)


    df[['growth_rate_1st_to_2nd', 'growth_rate_2nd_to_3rd']].hist(bins=bins, alpha=0.7, rwidth=0.85)
    plt.tight_layout()
    plt.savefig('growth_rate_histogram.png')

    # ボックスプロットを作成する
    plt.figure()
    df[['growth_rate_1st_to_2nd', 'growth_rate_2nd_to_3rd']].boxplot()
    plt.tight_layout()
    plt.savefig('growth_rate_boxplot.png')

if __name__ == "__main__":
    analyze_data('count_tweet_3division_growth_rates_and_counts.csv')