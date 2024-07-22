import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.rcParams['font.family'] = 'IPAexGothic'

def get_anime_info(anime_info, id):
    title = anime_info.loc[id, 'title']
    weekly_tweet_user_clusters = anime_info.loc[id, 'weekly_tweet_user_clusters']
    mean_tweet_user_clusters = anime_info.loc[id, 'mean_tweet_user_clusters']
    
    w_labels = ["上昇", "下降", "山型", "横ばい"]
    m_labels = ["多い", "中くらい", "低い"]

    return title, w_labels[weekly_tweet_user_clusters], m_labels[mean_tweet_user_clusters]

def analyze_weekly_correlation(input_dir, anime_info):
    results = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, filename))
            
            id = filename.split('.')[0]
            title, w_label, m_label = get_anime_info(anime_info, id)
            
            # 最終回のユーザ数を調整
            # df = df.iloc[:-1]
            
            # キーワード使用率を計算
            df['keyword_ratio'] = df['keyword_count'] / df['user_count']
            
            # 週ごとの相関を計算
            correlation, p_value = stats.pearsonr(df['user_count'], df['keyword_ratio'])
            
            results.append({
                'title': title,
                'correlation': correlation,
                'p_value': p_value,
                'w_label': w_label,
                'keyword_growth_rate': (df['keyword_count'].iloc[-1] - df['keyword_count'].iloc[0]) / df['keyword_count'].iloc[0]
            })
    
    return pd.DataFrame(results)

def plot_correlation_vs_growth(df):
    plt.figure(figsize=(14, 10))
    
    colors = {'上昇': 'red', '下降': 'blue', '山型': 'green', '横ばい': 'orange'}
    
    for w_label in colors.keys():
        df_subset = df[df['w_label'] == w_label]
        scatter = plt.scatter(df_subset['keyword_growth_rate'], df_subset['correlation'], 
                              c=colors[w_label], label=w_label,
                              s=100, alpha=0.7)
    
    plt.xlabel('ユーザ数の増加率', fontsize=18)
    plt.ylabel('ユーザ数と「アニメ」キーワード使用率の相関係数', fontsize=18)
    plt.title('ユーザ数増加率とキーワード使用率相関の関係', fontsize=18)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Y軸の範囲を-1.0から1.0に設定
    plt.ylim(-1.0, 1.0)
    
    for i, row in df.iterrows():
        plt.annotate(row['title'][:4], (row['keyword_growth_rate'], row['correlation']), fontsize=16)
    
    plt.legend(title='週間クラスター', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('plot/correlation_vs_growth_keyword_rate_colored_incloud_last.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    input_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/plot_target_word_count_anime'
    
    anime_info = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col=0)
    
    results_df = analyze_weekly_correlation(input_directory, anime_info)
    plot_correlation_vs_growth(results_df)
    
    # 結果を表示
    print(results_df.sort_values('correlation'))