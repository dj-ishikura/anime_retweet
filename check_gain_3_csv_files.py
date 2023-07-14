import pandas as pd
import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
import numpy as np

# 出力ファイル名
output_file_name = 'gain_3_count_tweet'

def split_dataframe(df, n):
    # データフレームの長さを取得
    length = len(df)

    # 分割するインデックスを計算
    split_indices = [round(i*length/n) for i in range(1, n)]

    # データフレームを分割
    split_dfs = np.split(df, split_indices)

    return split_dfs

# 画像ファイルのパスをリストに入れます
def create_pdf(image_files):
    output_pdf = output_file_name+'.pdf'
    c = canvas.Canvas(output_pdf, pagesize=landscape(letter))
    for image_file in image_files:
        img = ImageReader(image_file)
        iw, ih = img.getSize()
        width, height = landscape(letter)  # Use landscape page size
        aspect = ih / float(iw)
        c.drawImage(image_file, 0, 0, width=width, height=(width * aspect))
        c.showPage()
    c.save()


# ディレクトリを指定します
directory = 'count_tweet_2022'

# 出力ファイルを指定します
output_file = output_file_name+'.txt'

image_files = []

# 出力ファイルが存在する場合は削除します
if os.path.exists(output_file):
    os.remove(output_file)

# ディレクトリ内のすべてのCSVファイルをループします
for filename in os.listdir(directory):
    if filename.endswith('1_week_tweet_counts.csv'):
        # CSVファイルを読み込みます
        df = pd.read_csv(os.path.join(directory, filename))

        # データフレームを3分割
        split_dfs = split_dataframe(df, 3)
        average_counts = [split_df['count'].mean() for split_df in split_dfs]

        # 最初の行と最後の行を比較する
        if average_counts == sorted(average_counts):
            # 画像ファイル
            image_files.append(f'{directory}/{filename[:-4]}.png')
            # 全ての行が前の行よりも大きい場合、ファイル名を出力ファイルに書き込みます
            with open(output_file, 'a') as f:
                f.write(filename + '\n')

# 出力ファイルをソートします
with open(output_file, 'r') as f:
    lines = f.readlines()

# 先頭の数字を基準にソートします
lines.sort(key=lambda line: line.split('_')[0])

image_files = sorted(image_files)
create_pdf(image_files)
with open(output_file, 'w') as f:
    f.writelines(lines)
