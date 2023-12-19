import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
import numpy as np
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

def create_pdf(image_files, directory, threshold):
    output_pdf = f'result/tweet_over_{threshold}.pdf'
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

    id_list = [i.split('_')[0] for i in file_names]
    df_csv = pd.DataFrame({'id':id_list, 'mean': averages})
    df_csv.to_csv('result/tweet_mean.csv', index=False)

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
    sorted_averages = sorted(averages)
    median = np.percentile(sorted_averages, 50) # len(averages)の80%に相当する値
    plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label=f'中央値:{median}')
    
    
    upper = np.percentile(sorted_averages, 90) # len(averages)の80%に相当する値
    plt.axvline(upper, color='g', linestyle='dashed', linewidth=2, label=f'90パーセントタイル:{upper:.3f}')
    # plt.axvline(500, color='b', linestyle='dashed', linewidth=2, label='しきい値')

    plt.minorticks_on()
    # Set title and labels
    # plt.title('平均週間ツイートユーザの分布')
    plt.xlabel('平均週間ツイートユーザ数', fontsize=14)
    plt.ylabel('作品数', fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig('result/tweet_mean_hist.png')



# Replace 'your_directory' with the path to your directory
calculate_average_and_plot('count_tweet', 500)
