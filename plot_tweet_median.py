import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader

def create_pdf(image_files, directory):
    output_pdf = 'tweet_meadian_over_900.pdf'
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

def calculate_madian_and_plot(directory):
    file_names = [f for f in os.listdir(directory) if f.endswith('1_week_tweet_counts.csv')]
    meadians = []

    for file_name in file_names:
        df = pd.read_csv(os.path.join(directory, file_name))
        meadian = df['count'].median()
        meadians.append(meadian)

    labels = [int(f.split('-')[2].split('_')[0]) for f in file_names]
    plt.scatter(labels, meadians)
    image_files = []
    for i, label in enumerate(labels):
        plt.text(labels[i], meadians[i], str(label))
        if meadians[i] > 900:
            image_files.append(file_names[i])

    create_pdf(image_files, directory)
    plt.xlabel('File')
    plt.ylabel('meadian Count')
    plt.title('meadian Count per File')
    plt.savefig('tweet_median.png')

# Replace 'your_directory' with the path to your directory
calculate_madian_and_plot('count_tweet_2022')
