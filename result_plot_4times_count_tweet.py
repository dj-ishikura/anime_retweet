import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties 
import japanize_matplotlib
import os
import img2pdf
import numpy as np

def split_df(df):
    # 週数（何週目か）を計算
    df['week_number'] = (df.index - df.index.min()).days // 7 + 1

    # 全体の週数
    total_weeks = df['week_number'].max()

    # カテゴリを割り当てる関数
    def categorize_week(week):
        if week <= 4:
            return 'First'
        elif week <= total_weeks - 4:
            return 'Middle'
        else:
            return 'Last'

    # カテゴリを割り当て
    df['category'] = df['week_number'].apply(categorize_week)

    # データフレームを3つに分ける
    df_first_4_weeks = df[df['category'] == 'First']
    df_middle_weeks = df[df['category'] == 'Middle']
    df_last_4_weeks = df[df['category'] == 'Last']
    return df_first_4_weeks, df_middle_weeks, df_last_4_weeks

def plot_4times(input_file, output_file, title, id):
    df = pd.read_csv(input_file, index_col='date', parse_dates=True)
    df1, df2, df3 = split_df(df)
    avg1 = df1['count'].mean()
    avg2 = df2['count'].mean()
    avg3 = df3['count'].mean()
    growth_rates = [(avg2 - avg1) / avg1 * 100, (avg3 - avg2) / avg2 * 100]
    data_counts = [avg1, avg2, avg3]

    plt.subplot(1, 2, 1)
    colors = ['skyblue' if x > 0 else 'darkblue' for x in growth_rates]
    plt.bar(['序盤 to 中盤', '中盤 to 終盤'], growth_rates, color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Growth Rate (%)')
    plt.title('Growth Rate per Division')

    plt.subplot(1, 2, 2)
    plt.plot(['序盤', '中盤', '終盤'], data_counts, marker='o')
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
    tweet_directory = 'count_tweet'
    output_directory = 'plot_4times_count_tweet'
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
            growth_rates, data_counts = plot_4times(input_file, output_png, title, id)
            image_files.append(output_png)
            average_growth_rate = np.mean(growth_rates)
            results.append([id, title, average_growth_rate] + growth_rates + data_counts)

    output_pdf = 'result/plot_4times.pdf'
    image_files = sorted(image_files)
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert([i for i in image_files if i.endswith(".png")]))

    # Output results to CSV
    df_results = pd.DataFrame(results, columns=['id', 'title', 'average_growth_rate', 'growth_rate_1st_to_2nd', 'growth_rate_2nd_to_3rd', 'count_1st', 'count_2nd', 'count_3rd'])
    df_results.to_csv('result/count_tweet_4times_growth_rates_and_counts.csv', index=False)
