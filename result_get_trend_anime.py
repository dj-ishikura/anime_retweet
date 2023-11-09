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
        if df['tweet_users_count'].mean() > threshold:
            id = file.split('/')[1].split('_')[0]
            hit_anime_list.append(id)

    return hit_anime_list

def get_trend_anime(input_files, threshold):
    
    trend_anime_list = []

    for file in input_files:
        df = pd.read_csv(file, index_col='date', parse_dates=True)
        if df['tweet_users_count'].mean() > threshold:
            df1, df2, df3 = np.array_split(df, 3)
            avg1 = df1['tweet_users_count'].mean()
            avg2 = df2['tweet_users_count'].mean()
            avg3 = df3['tweet_users_count'].mean()
            growth_rates = [(avg2 - avg1) / avg1 * 100, (avg3 - avg2) / avg2 * 100]
            if growth_rates[0] > 0 and growth_rates[1] > 0:
                id = file.split('/')[1].split('_')[0]
                trend_anime_list.append(id)
    
    return trend_anime_list

def get_trend_anime_2(input_files, threshold):
    
    trend_anime_list = []

    for file in input_files:
        df = pd.read_csv(file, index_col='date', parse_dates=True)
        if df['tweet_users_count'].mean() > threshold:
            df1, df2, df3 = split_df(df)
            avg1 = df1['tweet_users_count'].mean()
            avg2 = df2['tweet_users_count'].mean()
            avg3 = df3['tweet_users_count'].mean()
            growth_rates = [(avg2 - avg1) / avg1 * 100, (avg3 - avg2) / avg2 * 100]
            if growth_rates[0] > 0 and growth_rates[1] > 0:
            # if growth_rates[0] > 10: 
                id = file.split('/')[1].split('_')[0]
                trend_anime_list.append(id)
    
    return trend_anime_list


def split_df(df):
    # 週数（何週目か）を計算
    df['week_number'] = (df.index - df.index.min()).days // 7 + 1

    # 全体の週数
    total_weeks = df['week_number'].max()

    # カテゴリを割り当てる関数
    def categorize_week(week):
        if week <= 4:
            return 'First 4 weeks'
        elif week <= total_weeks - 4:
            return 'Middle weeks'
        else:
            return 'Last 4 weeks'

    # カテゴリを割り当て
    df['category'] = df['week_number'].apply(categorize_week)

    # データフレームを3つに分ける
    df_first_4_weeks = df[df['category'] == 'First 4 weeks']
    df_middle_weeks = df[df['category'] == 'Middle weeks']
    df_last_4_weeks = df[df['category'] == 'Last 4 weeks']
    return df_first_4_weeks, df_middle_weeks, df_last_4_weeks


# Replace 'your_directory' with the path to your directory
directory = 'count_tweet'
threshold = 500
input_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('1_week_tweet_counts.csv')]

anime_list = [file.split('/')[1].split('_')[0] for file in input_files]
hit_anime_list = get_hit_anime(input_files, threshold)
trend_anime_list = get_trend_anime_2(input_files, threshold)

# まずは初期のリストをセットに変換します
trend_anime_set = set(trend_anime_list)
hit_anime_set = set(hit_anime_list)
anime_set = set(anime_list)


# AND 演算
trend_anime_set = trend_anime_set & hit_anime_set

# XOR 演算
hit_anime_set = trend_anime_set ^ hit_anime_set
miner_anime_set = anime_set ^ trend_anime_set
miner_anime_set = miner_anime_set ^ hit_anime_set

# 必要に応じてリストに戻す
trend_anime_list = list(trend_anime_set)
hit_anime_list = list(hit_anime_set)

miner_anime_list = sorted(list(miner_anime_set))
print(f'anime_list: {len(anime_list)}')
print(f'trend_anime_list: {len(trend_anime_list)}')
print(f'hit_anime_list: {len(hit_anime_list)}')
print(f'miner_anime_list: {len(miner_anime_list)}')

# output_dir = 'plot_3division_tweet_count_tweet'
output_dir = 'plot_4times_count_tweet'
output_result_dir = './result/'

image_files_1 = [os.path.join(directory, f+'_1_week_tweet_counts.png') for f in trend_anime_list]
image_files_2 = [os.path.join(output_dir, f+'.png') for f in trend_anime_list]
create_pdf(image_files_1, image_files_2, output_result_dir+'trend_anime_list.pdf')

image_files_1 = [os.path.join(directory, f+'_1_week_tweet_counts.png') for f in hit_anime_list]
image_files_2 = [os.path.join(output_dir, f+'.png') for f in hit_anime_list]
create_pdf(image_files_1, image_files_2, output_result_dir+'hit_anime_list.pdf')

image_files_1 = [os.path.join(directory, f+'_1_week_tweet_counts.png') for f in miner_anime_list]
image_files_2 = [os.path.join(output_dir, f+'.png') for f in miner_anime_list]
create_pdf(image_files_1, image_files_2, output_result_dir+'miner_anime_list.pdf')

# CSVファイルを読み込みます
df = pd.read_csv('./anime_data_updated.csv', index_col=0) 

# trend_anime_listをテキストファイルに出力
with open(output_result_dir+'trend_anime_list.txt', 'w') as f:
    for item in trend_anime_list:
        # Get the title corresponding to the item ID
        title = df.loc[item, '作品名']
        f.write(f"{item}: {title}\n") 

# hit_anime_listをテキストファイルに出力
with open(output_result_dir+'hit_anime_list.txt', 'w') as f:
    for item in hit_anime_list:
        # Get the title corresponding to the item ID
        title = df.loc[item, '作品名']
        f.write(f"{item}: {title}\n") 

# hit_anime_listをテキストファイルに出力
with open(output_result_dir+'miner_anime_list.txt', 'w') as f:
    for item in miner_anime_list:
        # Get the title corresponding to the item ID
        title = df.loc[item, '作品名']
        f.write(f"{item}: {title}\n") 

# ラベル付きリストの作成: (label, anime_id) 形式のタプルのリストを作ります
class_anime_list = [(anime_id, label) for label, anime_list in zip(["trend", "hit", "miner"], [trend_anime_list, hit_anime_list, miner_anime_list]) for anime_id in anime_list]
# データフレームを作成
df_output = pd.DataFrame(class_anime_list, columns=['id', 'class'])
# アニメ名の取得
df_output['title'] = df_output['id'].apply(lambda x: df.loc[x, '作品名'])
# CSVファイルに出力
df_output.to_csv(output_result_dir + 'class_anime_list.csv', index=False)

