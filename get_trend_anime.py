import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
import numpy as np
from PIL import Image
import img2pdf

def combine_images(path1, path2, output_path):
    # 画像を開く
    image1 = Image.open(path1)
    image2 = Image.open(path2)

    # 画像のサイズを取得 (as they need to be the same)
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 新しい画像のサイズを計算
    new_width = width1 + width2
    new_height = max(height1, height2)

    # 新しい画像を作成
    new_image = Image.new('RGB', (new_width, new_height))

    # 既存の画像を新しい画像にペースト
    new_image.paste(image1, (0,0))
    new_image.paste(image2, (width1,0))

    # 新しい画像を保存
    new_image.save(output_path)

def create_pdf(path_list1, path_list2, output_pdf_path):
    combined_images_paths = []

    # 各ペアの画像を結合します
    for i, (path1, path2) in enumerate(zip(path_list1, path_list2)):
        output_path = f'combined_{i}.png'
        combine_images(path1, path2, output_path)
        combined_images_paths.append(output_path)

    # すべての画像をPDFとして結合します
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert([path for path in combined_images_paths]))

    # 一時的に作成した結合画像を削除します
    for path in combined_images_paths:
        os.remove(path)

def get_hit_anime(input_files, threshold):
    hit_anime_list = []

    for file in input_files:
        df = pd.read_csv(file)
        if df['count'].mean() > threshold:
            id = file.split('/')[1].split('_')[0]
            hit_anime_list.append(id)

    return hit_anime_list

def get_trend_anime(input_files, threshold):
    
    trend_anime_list = []

    for file in input_files:
        df = pd.read_csv(file, index_col='date', parse_dates=True)
        if df['count'].mean() > threshold:
            df1, df2, df3 = np.array_split(df, 3)
            avg1 = df1['count'].mean()
            avg2 = df2['count'].mean()
            avg3 = df3['count'].mean()
            growth_rates = [(avg2 - avg1) / avg1 * 100, (avg3 - avg2) / avg2 * 100]
            if growth_rates[0] > 0 and growth_rates[1] > 0:
                id = file.split('/')[1].split('_')[0]
                trend_anime_list.append(id)
    
    return trend_anime_list

# Replace 'your_directory' with the path to your directory
directory = 'count_tweet_2022'
threshold = 500
input_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('1_week_tweet_counts.csv')]

hit_anime_list = get_hit_anime(input_files, threshold)
trend_anime_list = get_trend_anime(input_files, threshold)

# まずは初期のリストをセットに変換します
trend_anime_set = set(trend_anime_list)
hit_anime_set = set(hit_anime_list)

# AND 演算
trend_anime_set = trend_anime_set & hit_anime_set

# XOR 演算
hit_anime_set = trend_anime_set ^ hit_anime_set

# 必要に応じてリストに戻す
trend_anime_list = list(trend_anime_set)
hit_anime_list = list(hit_anime_set)

print(trend_anime_list)

image_files_1 = [os.path.join(directory, f+'_1_week_tweet_counts.png') for f in trend_anime_list]
image_files_2 = [os.path.join('plot_3division_count_tweet', f+'.png') for f in trend_anime_list]
create_pdf(image_files_1, image_files_2, 'trend_anime_list.pdf')

image_files_1 = [os.path.join(directory, f+'_1_week_tweet_counts.png') for f in hit_anime_list]
image_files_2 = [os.path.join('plot_3division_count_tweet', f+'.png') for f in hit_anime_list]
create_pdf(image_files_1, image_files_2, 'hit_anime_list.pdf')

# trend_anime_listをテキストファイルに出力
with open('trend_anime_list.txt', 'w') as f:
    for item in trend_anime_list:
        f.write("%s\n" % item)

# hit_anime_listをテキストファイルに出力
with open('hit_anime_list.txt', 'w') as f:
    for item in hit_anime_list:
        f.write("%s\n" % item)
