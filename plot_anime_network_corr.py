import pandas as pd
from scipy.stats import pearsonr
import json
import sys

def calculate_correlations(csv_file_path, csv_output_path):
    # CSVファイルを読み込むのだ
    data = pd.read_csv(csv_file_path)
    
    # 週iと週i-1のデータを取得するためにデータフレームをずらすのだ
    data_shifted = data.shift(1)
    
    # 相関係数を計算するのだ
    correlations = []
    parameters = data.columns[2:]
    for param in parameters:
        for target in ["Number of vertices", "Number of edges"]:
            correlation, p_value = pearsonr(data[target][1:], data_shifted[param][1:])
            correlations.append({
                "target": target,
                "param": param,
                "correlation": correlation,
                "p_value": p_value
            })

    # CSVファイルに保存するのだ
    correlations_df = pd.DataFrame(correlations)
    correlations_df.to_csv(csv_output_path, index=False)

# CSVファイルとJSON出力パスを指定して関数を呼び出すのだ
calculate_correlations(sys.argv[1], sys.argv[2])

# CSVファイルのパスを指定して関数を呼び出すのだ
# calculate_correlations('weekly_anime_network/2022-10-582/2022-10-582.csv')
