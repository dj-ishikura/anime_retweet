import pandas as pd
import json
import sys
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

# 空のDataFrameを作成
def get_data(file):
    df_count = pd.DataFrame(columns=['date', 'count'])

    # JSONLファイルを一行ずつ読み込む
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            date = data['date']
            user_ids = data['user_ids']
            
            # user_idsの長さ（つまり、ユーザ数）をカウント
            count = len(user_ids)
            
            # カウントと日付をDataFrameに追加
            df_count = df_count.append({'date': date, 'count': count}, ignore_index=True)

    # 結果を確認
    print(df_count)
    return df_count

def get_info_from_csv(id):
    df = pd.read_csv('./anime_data_updated.csv', index_col=0)
    title = df.loc[id, '作品名']

    return title

if __name__ == "__main__":
    # ファイルの取得
    followers_file = sys.argv[1]
    users_file = sys.argv[2]
    output_png = sys.argv[3]
    output_csv = sys.argv[4]
    id = sys.argv[5]
    title = get_info_from_csv(id)

    # dfにする
    df_followers = get_data(followers_file)
    df_users = get_data(users_file)
    
    # P_t と A_t のデータをDataFrameから取得
    # df_users から P_t_plus_1 を取得（次の週のツイートユーザ数）
    P_t_plus_1 = df_users['count'][1:].to_numpy()

    # df_followers から A_t を取得（前の週のツイートに接触したユーザ数）
    A_t = df_followers['count'][:-1].to_numpy()

    A_t = np.array(A_t, dtype=np.int64)  # または dtype=np.float64
    P_t_plus_1 = np.array(P_t_plus_1, dtype=np.int64)  # または dtype=np.float64

    # データ点の数
    n = len(A_t)

    # lstsq関数が扱いやすい形にデータを変換
    A = np.vstack([A_t, np.ones(n)]).T

    # 最小二乗法で係数を求める
    alpha_beta, residuals, rank, singular_values = np.linalg.lstsq(A, P_t_plus_1, rcond=None)

    # alphaとbeta
    alpha, beta = alpha_beta

    # 残差（誤差）の和
    if len(residuals) > 0:  # フィットが良い場合、この値は空になることがある
        residual_sum = residuals[0]
    else:
        residual_sum = 0

    # 決定係数 R^2 の計算
    predicted_P_t_plus_1 = alpha * A_t + beta  # モデルによるP_{t+1}の予測値
    ss_res = np.sum((P_t_plus_1 - predicted_P_t_plus_1) ** 2)  # 残差平方和
    ss_tot = np.sum((P_t_plus_1 - np.mean(P_t_plus_1)) ** 2)  # 全平方和
    r2 = 1 - (ss_res / ss_tot)

    # 実データをプロット（折れ線グラフ）
    plt.plot(df_users['date'], df_users['count'], marker='o', label='Actual data')

    # 線形回帰の結果に基づいて予測される P_t_plus_1 を計算
    predicted_P_t_plus_1 = beta + alpha * A_t

    # 線形回帰の結果をプロット（折れ線グラフ）
    plt.plot(df_users['date'][1:], predicted_P_t_plus_1, marker='x', linestyle='--', label='Fitted line', color='red')

    # グラフの各種設定
    plt.title(f'P_t={alpha:.2f} * A_t + {beta:.2f}, residuals={r2:.2f}\n{title}, {id}', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Number of Tweeting Users')
    plt.legend()

    # グラフを表示
    plt.savefig(output_png)

    # CSVに出力
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # ヘッダー行
        csvwriter.writerow(['id', 'alpha', 'beta', 'r2', 'rank', 'singular_values'])
        
        # データ行
        csvwriter.writerow([id, alpha, beta, r2, rank, singular_values])