import json
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['font.family'] = 'IPAexGothic'

def get_anime_info(anime_info, id):
    title = anime_info.loc[id, 'title']
    weekly_tweet_user_clusters = anime_info.loc[id, 'weekly_tweet_user_clusters']
    mean_tweet_user_clusters = anime_info.loc[id, 'mean_tweet_user_clusters']
    
    w_labels = ["上昇", "下降", "山型", "横ばい"]
    m_labels = ["多い", "中くらい", "低い"]

    return title, w_labels[weekly_tweet_user_clusters], m_labels[mean_tweet_user_clusters]

def process_file(file_path):
    unique_users = set()
    unique_words = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data.get('user_id')
            if user_id not in unique_users:
                nouns = data.get('extracted_nouns', [])
                unique_words.update(nouns)
                unique_users.add(user_id)
    
    return len(unique_users), len(unique_words)

def collect_data(input_dir, anime_info):
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            id = filename.split('.')[0]
            title, w_label, m_label = get_anime_info(anime_info, id)
            
            file_path = os.path.join(input_dir, filename)
            user_count, word_count = process_file(file_path)
            
            results.append({
                'title': title,
                'user_count': user_count,
                'word_count': word_count,
                'w_label': w_label,
                'm_label': m_label
            })
    return pd.DataFrame(results)

def calculate_correlation(df):
    correlation, p_value = stats.pearsonr(df['user_count'], df['word_count'])
    return correlation, p_value

def create_plot(df, correlation, output_plot):
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['user_count'], df['word_count'])
    ax.set_xlabel('ユーザー数')
    ax.set_ylabel('単語（名詞）の種類数')
    ax.set_title(f'ユーザー数と単語の種類数の関係\n相関係数: {correlation:.2f}')
    
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower right", title="平均ユーザー数")
    ax.add_artist(legend1)
    
    z = np.polyfit(df['user_count'], df['word_count'], 1)
    p = np.poly1d(z)
    ax.plot(df['user_count'], p(df['user_count']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close(fig)

def main(input_dir, output_plot, anime_info):
    df = collect_data(input_dir, anime_info)
    
    correlation, p_value = calculate_correlation(df)
    
    create_plot(df, correlation, output_plot)
    
    print(f"相関係数: {correlation:.2f}")
    print(f"p値: {p_value:.4f}")
    
    df.to_csv(output_plot.replace('.png', '_details.csv'), index=False, encoding='utf-8')

if __name__ == '__main__':
    input_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/tweet_user_profile_noun_concat'
    output_plot = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/correlation_noun_vs_user_count_noun_concat.png'
    anime_info = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col=0)
    main(input_directory, output_plot, anime_info)