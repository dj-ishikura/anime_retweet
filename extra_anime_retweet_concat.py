import pandas as pd
import os
import shutil

# anime_class.csv を読み込む
anime_class_df = pd.read_csv('anime_class.csv')

# 出力ディレクトリを確認し、存在しない場合は作成する
output_dir = 'extra_anime_retweet_concat'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# anime_retweet_concat ディレクトリ内のファイルを探し、見つかったらコピーする
for anime_id in anime_class_df['id']:
    source_file_path = f'anime_retweet_concat/{anime_id}.csv'
    dest_file_path = f'{output_dir}/{anime_id}.csv'
    
    if os.path.exists(source_file_path):
        shutil.copy(source_file_path, dest_file_path)
        print(f'ファイルがコピーされました: {source_file_path} -> {dest_file_path}')
    else:
        print(f'ファイルが見つかりません: {source_file_path}')
