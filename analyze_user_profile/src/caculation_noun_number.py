import json
import os
import pandas as pd
from collections import Counter, defaultdict

def get_anime_info(anime_info, id):
    title = anime_info.loc[id, 'title']
    weekly_tweet_user_clusters = anime_info.loc[id, 'weekly_tweet_user_clusters']
    mean_tweet_user_clusters = anime_info.loc[id, 'mean_tweet_user_clusters']
    
    w_labels = ["上昇", "下降", "山型", "横ばい"]
    m_labels = ["多い", "中くらい", "低い"]

    return title, w_labels[weekly_tweet_user_clusters], m_labels[mean_tweet_user_clusters]

def process_all_files(input_dir):
    all_words = set()
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            anime_id = filename.split('.')[0]
            file_path = os.path.join(input_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)

                    nouns = data.get('extracted_nouns', [])
                    
                    all_words.update(nouns)
    
    return len(all_words)

def main(input_dir, output_file, anime_info):
    unique_words_count = process_all_files(input_dir)
    
    print(f"Total unique words across all anime: {unique_words_count}")

if __name__ == '__main__':
    input_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/tweet_user_profile_wakachi'
    output_file = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/results/word_count_summary.csv'
    anime_info = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col=0)
    main(input_directory, output_file, anime_info)