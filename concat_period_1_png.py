import pandas as pd
import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader

# 出力ファイル名
output_file_name = 'count_retweet_2022'

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
directory = 'count_retweet_2022'

# 出力ファイルを指定します
output_file = output_file_name+'.txt'

image_files = []

# 出力ファイルが存在する場合は削除します
if os.path.exists(output_file):
    os.remove(output_file)

# ディレクトリ内のすべてのCSVファイルをループします
for filename in os.listdir(directory):
    if filename.endswith('1_week_retweet_counts.png'):
        image_files.append(f'{directory}/{filename[:-4]}.png')

image_files = sorted(image_files)
create_pdf(image_files)
