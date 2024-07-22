import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib
import csv
import numpy as np

def get_user_anime_data(directory):
    user_anime_data = defaultdict(lambda: {'tweet': set(), 'profile': False})
    
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            anime_id = os.path.splitext(filename)[0]
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        user_id = data['user_id']
                        user_anime_data[user_id]['tweet'].add(anime_id)
                        
                        # プロフィールに「アニメ」が含まれているかチェック
                        if 'アニメ' in data.get('profile', ''):
                            user_anime_data[user_id]['profile'] = True
                    except json.JSONDecodeError as e:
                        print(f"Error in file: {filename}")
                        print(f"Error message: {str(e)}")
                        print(f"Problematic line: {line}")
                        print("---")
                        break
    
    return user_anime_data

def plot_cumulative_distribution(user_anime_data, output_png):
    counts_anime = [len(data['tweet']) for data in user_anime_data.values() if data['profile']]
    counts_non_anime = [len(data['tweet']) for data in user_anime_data.values() if not data['profile']]
    
    total_users = len(user_anime_data)
    anime_users = len(counts_anime)
    non_anime_users = len(counts_non_anime)
    max_anime_count = max(max(counts_anime), max(counts_non_anime))
    avg_anime_count = sum(len(data['tweet']) for data in user_anime_data.values()) / total_users
    
    plt.figure(figsize=(12, 8))
    
    for counts, label, color in [(counts_anime, 'プロフィールにアニメあり', 'blue'),
                                 (counts_non_anime, 'プロフィールにアニメなし', 'red')]:
        hist, bin_edges = np.histogram(counts, bins=range(1, max_anime_count + 2))
        cumulative = np.cumsum(hist[::-1])[::-1] / len(counts)
        plt.plot(bin_edges[:-1], cumulative * 100, marker='o', markersize=3, label=label, color=color)
    
    plt.xlabel('ツイートしたアニメ作品数', fontsize=16)
    plt.ylabel('n作品以上ツイートしたユーザの割合 (%)', fontsize=16)
    plt.yscale('log')
    plt.xlim(left=1)
    plt.ylim(bottom=0.01, top=100)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.xticks([1, 10, 20, 50, 100, 200, 500], rotation=45)
    plt.legend(fontsize=12)
    
    for n in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        if n <= max_anime_count:
            for counts, color in [(counts_anime, 'blue'), (counts_non_anime, 'red')]:
                hist, bin_edges = np.histogram(counts, bins=range(1, max_anime_count + 2))
                cumulative = np.cumsum(hist[::-1])[::-1] / len(counts)
                idx = np.searchsorted(bin_edges, n)
                plt.annotate(f'{n}: {cumulative[idx]*100:.2f}%', 
                             xy=(n, cumulative[idx]*100), 
                             xytext=(5, 5), 
                             textcoords='offset points',
                             fontsize=10,
                             color=color,
                             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    stats_text = f'総ユーザー数: {total_users:,}\n'
    stats_text += f'アニメプロフィールユーザー数: {anime_users:,}\n'
    stats_text += f'非アニメプロフィールユーザー数: {non_anime_users:,}\n'
    stats_text += f'最大ツイートアニメ数: {max_anime_count}\n'
    stats_text += f'平均ツイートアニメ数: {avg_anime_count:.2f}'
    
    plt.text(0.95, 0.95, stats_text,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"累積分布グラフを{output_png}に保存しました。")

    return total_users, anime_users, non_anime_users, max_anime_count, avg_anime_count

def main():
    directory = 'tweet_user_profile_all_works'
    output_png_cumulative = 'plot/plot_count_works_cumulative_all_works_anime.png'
    output_csv = 'data/count_works_cumulative_all_works_anime.csv'
    
    user_anime_data = get_user_anime_data(directory)
    plot_cumulative_distribution(user_anime_data, output_png_cumulative)
    save_to_csv(user_anime_data, output_csv)

if __name__ == "__main__":
    main()