import pandas as pd
import os

data_dir = 'data/'
problematic_files = []

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 1行目のカンマの数を取得する
            header_comma_count = lines[0].count(',')
            
            for line in lines[1:]:
                # それ以降の行のカンマの数と比較する
                if line.count(',') != header_comma_count:
                    problematic_files.append(file)
                    break

problematic_files.sort()
# 問題のあるファイル名を出力する
if problematic_files:
    print("問題のあるCSVファイルは以下の通りなのだ：")
    for file in problematic_files:
        print(file)
else:
    print("問題のあるCSVファイルは見つからなかったのだ！")
