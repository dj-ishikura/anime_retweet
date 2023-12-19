import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# CSVファイルからクラスタ情報を読み取る
cluster_df = pd.read_csv('anime_class_dtw_kaisou.csv')

# ディレクトリパス
directory_path = 'count_tweet'

# 出力PDFsを保存するディレクトリ
output_directory = 'output_pdfs_dtw_kaisou'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# クラスタごとにPNGファイルを集める
for cluster_number in cluster_df['class'].unique():
    for tweet_cluster_number in cluster_df['tweet_user_class'].unique():
        cluster_items = cluster_df[cluster_df['class'] == cluster_number]
        cluster_items = cluster_items[cluster_items['tweet_user_class'] == tweet_cluster_number]
        pdf_filename = f'{output_directory}/cluster_{cluster_number}_{tweet_cluster_number}.pdf'
        
        with PdfPages(pdf_filename) as pdf_pages:
            for _, row in cluster_items.iterrows():
                png_file = f'{directory_path}/{row["id"]}_1_week_tweet_counts.png'
                if os.path.exists(png_file):
                    img = plt.imread(png_file)
                    plt.figure(figsize=(8, 8))
                    plt.imshow(img)
                    pdf_pages.savefig(bbox_inches='tight', pad_inches=0)
                    plt.close()
        
        print(f'クラスタ{cluster_number}のPDFが作成されたのだ: {pdf_filename}')
