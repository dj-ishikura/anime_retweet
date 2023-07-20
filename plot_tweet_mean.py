import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
import numpy as np

def create_pdf(image_files, directory, threshold):
    output_pdf = f'tweet_over_{threshold}.pdf'
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

def calculate_average_and_plot(directory, threshold):
    file_names = [f for f in os.listdir(directory) if f.endswith('1_week_tweet_counts.csv')]
    averages = []

    for file_name in file_names:
        df = pd.read_csv(os.path.join(directory, file_name))
        average = df['count'].mean()
        averages.append(average)

    labels = [int(f.split('-')[2].split('_')[0]) for f in file_names]
    plt.scatter(labels, averages)
    image_files = []
    for i, label in enumerate(labels):
        plt.text(labels[i], averages[i], str(label))
        if averages[i] > threshold:
            image_files.append(file_names[i])

    create_pdf(image_files, directory, threshold)
    plt.xlabel('File')
    plt.ylabel('Average Count')
    plt.title('Average Count per File')
    plt.savefig('tweet_mean.png')

    plt.figure()
    bins = range(0, 5000, 100)
    plt.hist(averages, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    mean = np.mean(average)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'mean:{mean}')
    
    sorted_averages = sorted(averages)
    upper = np.percentile(sorted_averages, 90) # len(averages)の80%に相当する値
    plt.axvline(upper, color='g', linestyle='dashed', linewidth=1, label=f'90 percentile:{upper}')

    plt.minorticks_on()
    # Set title and labels
    plt.title('Histogram of Average Tweet Counts')
    plt.xlabel('Average Tweet Counts')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('tweet_mean_hist.png')

    # ボックスプロットを作成
    plt.figure()
    plt.boxplot(averages, vert=False)
    plt.title('Boxplot of data')
    plt.savefig('tweet_mean_boxplot.png')


# Replace 'your_directory' with the path to your directory
calculate_average_and_plot('count_tweet_2022', 400)
