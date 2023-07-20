import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
import numpy as np

def create_pdf(image_files, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=landscape(letter))
    for image_file in image_files:
        img = ImageReader(image_file)
        iw, ih = img.getSize()
        width, height = landscape(letter)  # Use landscape page size
        aspect = ih / float(iw)
        c.drawImage(image_file, 0, 0, width=width, height=(width * aspect))
        c.showPage()
    c.save()

def calculate_average(directory, threshold):
    file_names = [f for f in os.listdir(directory) if f.endswith('1_week_tweet_counts.csv')]
    hit_anime_list = []

    for file_name in file_names:
        df = pd.read_csv(os.path.join(directory, file_name))
        if df['count'].mean() > threshold:
            id = file_name.split('_')[0]
            hit_anime_list.append(id)

    return hit_anime_list

# Replace 'your_directory' with the path to your directory
directory = 'count_tweet_2022'
threshold = 500
hit_anime_list = calculate_average(directory, threshold)

image_files = [os.path.join(directory, f+'_1_week_tweet_counts.png') for f in hit_anime_list]
create_pdf(image_files, f'tweet_over_{threshold}.pdf')
image_files = [os.path.join('plot_3division_count_tweet', f+'.png') for f in hit_anime_list]
create_pdf(image_files, f'tweet_over_{threshold}_3division.pdf')

for h in hit_anime_list:
    print(h)