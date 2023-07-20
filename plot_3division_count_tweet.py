import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties 
import japanize_matplotlib
import os
import img2pdf
import numpy as np

def plot_3division(input_file, output_file, title, id):
    df = pd.read_csv(input_file, index_col='date', parse_dates=True)
    df1, df2, df3 = np.array_split(df, 3)
    avg1 = df1['count'].mean()
    avg2 = df2['count'].mean()
    avg3 = df3['count'].mean()
    growth_rates = [(avg2 - avg1) / avg1 * 100, (avg3 - avg2) / avg2 * 100]
    data_counts = [avg1, avg2, avg3]

    plt.subplot(1, 2, 1)
    colors = ['skyblue' if x > 0 else 'darkblue' for x in growth_rates]
    plt.bar(['1st to 2nd', '2nd to 3rd'], growth_rates, color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Growth Rate (%)')
    plt.title('Growth Rate per Division')

    plt.subplot(1, 2, 2)
    plt.plot(['1st third', '2nd third', '3rd third'], data_counts, marker='o')
    plt.ylabel('Data Counts')
    plt.title('Data Counts per Division')

    plt.suptitle(f'{id}\n{title}', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig(output_file)
    plt.close()

    return growth_rates, data_counts

def get_info_from_csv(id):
    df = pd.read_csv('./anime_data_updated.csv', index_col=0)
    title = df.loc[id, '作品名']

    return title

if __name__ == "__main__":
    tweet_directory = 'count_tweet_2022'
    output_directory = 'plot_3division_count_tweet'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_files = []
    results = []

    for filename in os.listdir(tweet_directory):
        if filename.endswith('1_week_tweet_counts.csv'):
            id = filename.split('_')[0]
            output_png = os.path.join(output_directory, id + '.png')            
            title = get_info_from_csv(id)
            input_file = os.path.join(tweet_directory, filename)
            growth_rates, data_counts = plot_3division(input_file, output_png, title, id)
            image_files.append(output_png)
            results.append([id, title] + growth_rates + data_counts)

    output_pdf = 'plot_3division.pdf'
    image_files = sorted(image_files)
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert([i for i in image_files if i.endswith(".png")]))

    # Output results to CSV
    df_results = pd.DataFrame(results, columns=['id', 'title', 'growth_rate_1st_to_2nd', 'growth_rate_2nd_to_3rd', 'count_1st', 'count_2nd', 'count_3rd'])
    df_results.to_csv('count_tweet_3division_growth_rates_and_counts.csv', index=False)
