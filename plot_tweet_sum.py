import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
import numpy as np

def create_pdf(image_files, directory):
    output_pdf = 'tweet_sum_over_2600.pdf'
    c = canvas.Canvas(output_pdf, pagesize=landscape(letter))
    for image_file in image_files:
        image_file = directory+'/'+image_file[:-4]+'.png'
        img = ImageReader(image_file)
        iw, ih = img.getSize()
        width, height = landscape(letter)  # Use landscape page size
        aspect = ih / float(iw)
        c.drawImage(image_file, 0, 0, width=width, height=(width * aspect))
        c.showPage()
    c.save()

def calculate_total_and_plot(directory):
    file_names = [f for f in os.listdir(directory) if f.endswith('1_week_tweet_counts.csv')]
    totals = []

    for file_name in file_names:
        df = pd.read_csv(os.path.join(directory, file_name))
        total = df['count'].sum()
        totals.append(total)

    labels = [int(f.split('-')[2].split('_')[0]) for f in file_names]
    plt.scatter(labels, totals)
    image_files = []
    for i, label in enumerate(labels):
        plt.text(labels[i], totals[i], str(label))
        if totals[i] > 2600:
            image_files.append(file_names[i])

    create_pdf(image_files, directory)
    plt.xlabel('File')
    plt.ylabel('total Count')
    plt.title('total Count per File')
    plt.savefig('tweet_sum.png')

    plt.figure()
    print(np.amax(totals))
    bins = range(0, 65000, 1000)
    plt.hist(totals, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    mean = np.mean(total)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'mean:{mean}')
    plt.minorticks_on()

    sorted_totals = sorted(totals)
    upper = np.percentile(sorted_totals, 90) # len(sorted_totals)の90%に相当する値
    plt.axvline(upper, color='g', linestyle='dashed', linewidth=1, label=f'90 percentile:{upper}')

    # Set title and labels
    plt.title('Histogram of total Tweet Counts')
    plt.xlabel('total Tweet Counts')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('tweet_sum_hist.png')

    # ボックスプロットを作成
    plt.figure()
    plt.boxplot(totals, vert=False)
    plt.title('Boxplot of data')
    plt.savefig('tweet_sum_boxplot.png')


# Replace 'your_directory' with the path to your directory
calculate_total_and_plot('count_tweet_2022')
