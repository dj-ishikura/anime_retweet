import json
import os
import math
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def calculate_entropy(word_counts):
    total = sum(word_counts.values())
    return -sum((count / total) * math.log2(count / total) for count in word_counts.values())

def normalized_entropy(word_counts):
    entropy = calculate_entropy(word_counts)
    max_entropy = math.log2(len(word_counts))
    # return entropy / max_entropy if max_entropy > 0 else 0
    return entropy

def process_file(file_path):
    word_counts = Counter()
    user_set = set()
    user_profiles = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data.get('user_id')
            nouns = data.get('extracted_nouns', [])
            
            if user_id not in user_profiles:
                user_profiles[user_id] = set(nouns)
            else:
                user_profiles[user_id].update(nouns)
    
    for user_id, nouns in user_profiles.items():
        word_counts.update(nouns)
        user_set.add(user_id)
    
    return word_counts, len(user_set)

def analyze_files(input_dir):
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(input_dir, filename)
            word_counts, user_count = process_file(file_path)
            
            norm_entropy = normalized_entropy(word_counts)
            unique_words = len(word_counts)
            total_words = sum(word_counts.values())
            
            results.append({
                'filename': filename,
                'normalized_entropy': norm_entropy,
                'unique_words': unique_words,
                'total_words': total_words,
                'user_count': user_count
            })
    
    return results

def plot_entropy_vs_users(results):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='user_count', y='normalized_entropy')
    plt.title('Normalized Entropy vs User Count')
    plt.xlabel('Number of Users')
    plt.ylabel('Normalized Entropy')
    
    # 回帰直線を追加
    sns.regplot(data=df, x='user_count', y='normalized_entropy', scatter=False, color='red')
    
    # 相関係数を計算
    correlation = df['user_count'].corr(df['normalized_entropy'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/entropy_vs_users.png')
    plt.close()

def main(input_dir):
    results = analyze_files(input_dir)
    
    # 結果をCSVファイルに保存
    df = pd.DataFrame(results)
    df.to_csv('/work/n213304/learn/anime_retweet_2/analyze_user_entropy/data/entropy_results.csv', index=False)
    
    # エントロピーとユーザー数の関係をプロット
    plot_entropy_vs_users(results)
    
    # 結果を表示
    for result in sorted(results, key=lambda x: x['normalized_entropy'], reverse=True):
        print(f"File: {result['filename']}")
        print(f"  Normalized Entropy: {result['normalized_entropy']:.4f}")
        print(f"  Unique words: {result['unique_words']}")
        print(f"  Total words: {result['total_words']}")
        print(f"  User count: {result['user_count']}")
        print()

if __name__ == '__main__':
    input_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/tweet_user_profile_wakachi'
    main(input_directory)