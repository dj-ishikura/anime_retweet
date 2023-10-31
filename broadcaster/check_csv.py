import pandas as pd
import os

data_dir = 'data/'
problematic_files = []

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file)
        try:
            # ファイルを読み込む
            pd.read_csv(file_path)
        except pd.errors.ParserError:
            # エラーが発生したファイル名をリストに追加する
            problematic_files.append(file)

# 問題のあるファイル名を出力する
if problematic_files:
    print("問題のあるCSVファイルは以下の通りなのだ：")
    for file in problematic_files:
        print(file)
else:
    print("問題のあるCSVファイルは見つからなかったのだ！")
