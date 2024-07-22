import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import japanize_matplotlib

plt.rcParams['font.family'] = 'IPAexGothic'

def get_anime_info(anime_info, id):
    title = anime_info.loc[id, 'title']
    weekly_tweet_user_clusters = anime_info.loc[id, 'weekly_tweet_user_clusters']
    mean_tweet_user_clusters = anime_info.loc[id, 'mean_tweet_user_clusters']
    
    w_labels = ["上昇", "下降", "山型", "横ばい"]
    m_labels = ["多い", "中くらい", "低い"]

    return title, w_labels[weekly_tweet_user_clusters], m_labels[mean_tweet_user_clusters]

def plot_weekly_ratios(input_dir, anime_info, output_pdf):
    all_data = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, filename))
            id = filename.split('.')[0]
            title, w_label, m_label = get_anime_info(anime_info, id)
            
            # 週ごとの比率を計算
            weekly_ratios = df['keyword_count'] / df['user_count']
            
            min_ratio = weekly_ratios.min()
            max_ratio = weekly_ratios.max()
            
            all_data.append({
                'title': title,
                'min_ratio': min_ratio,
                'max_ratio': max_ratio,
                'ratio_range': max_ratio - min_ratio,
                'std_dev': np.std(weekly_ratios),
                'w_label': w_label,
                'mean_users': df['user_count'].mean()
            })

    # 比率の範囲（max - min）でソート
    all_data.sort(key=lambda x: x['ratio_range'], reverse=False)

    with PdfPages(output_pdf) as pdf:
        plt.figure(figsize=(16, len(all_data) * 0.4))
        plt.title('各アニメの週ごとのキーワードを含むユーザの比率（最小値-最大値）')
        plt.xlabel('キーワードを含むユーザの比率')
        plt.yticks(range(len(all_data)), [d['title'] for d in all_data])
        
        colors = {'上昇': 'red', '下降': 'blue', '山型': 'green', '横ばい': 'orange'}
        
        for i, data in enumerate(all_data):
            color = colors[data['w_label']]
            plt.plot([data['min_ratio'], data['max_ratio']], [i, i], color=color, linewidth=2, alpha=0.7)
            plt.scatter([data['min_ratio'], data['max_ratio']], [i, i], color=color, s=50, alpha=0.7)
            plt.text(1.01, i, f'幅 = {data["ratio_range"]:.4f} ({data["w_label"]})', verticalalignment='center')
        
        plt.xlim(0, 1)
        plt.tight_layout()
        
        # Remove duplicate labels
        handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors.values()]
        plt.legend(handles, colors.keys(), title='週間クラスター', loc='center left', bbox_to_anchor=(1.15, 0.5))
        
        plt.subplots_adjust(right=0.85)  # Make room for legend
        pdf.savefig(bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    input_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/plot_target_word_count_anime'
    anime_info = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col=0)
    
    # plotディレクトリを作成（存在しない場合）
    output_dir = 'plot'
    os.makedirs(output_dir, exist_ok=True)
    
    output_pdf = os.path.join(output_dir, 'anime_weekly_keyword_ratio_minmax_analysis.pdf')
    
    plot_weekly_ratios(input_directory, anime_info, output_pdf)