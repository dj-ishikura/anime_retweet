import pandas as pd
import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader

# 出力ファイル名
output_file_name = 'gain_count_tweet'

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
    if filename.endswith('.csv'):
        # CSVファイルを読み込みます
        df = pd.read_csv(os.path.join(directory, filename))
        # データフレームの行数が2より大きい場合、最後の行を削除します
        '''
        if len(df) > 2:
            df = df.iloc[:-1]
        '''
        # 'count'列の値が前の値よりも大きいかどうかをチェックします
        if all(df['count'].diff().dropna() > 0):
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
