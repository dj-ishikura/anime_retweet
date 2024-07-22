# plot_user_tweet_count_for_anime.py

import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib
import csv

def get_user_anime_data(directory):
    user_anime_data = defaultdict(set)
    
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            anime_id = os.path.splitext(filename)[0]
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        user_id = data['user_id']
                        user_anime_data[user_id].add(anime_id)
                    except json.JSONDecodeError as e:
                        print(f"Error in file: {filename}")
                        print(f"Error message: {str(e)}")
                        print(f"Problematic line: {line}")
                        print("---")
                        break
    
    return user_anime_data

def plot_histograms(user_anime_data, output_png_normal, output_png_log):
    counts = [len(anime_set) for anime_set in user_anime_data.values()]
    
    total_users = len(user_anime_data)
    max_anime_count = max(counts)
    avg_anime_count = sum(counts) / total_users
    
    stats_text = f'総ユーザー数: {total_users:,}\n'
    stats_text += f'最大ツイートアニメ数: {max_anime_count}\n'
    stats_text += f'平均ツイートアニメ数: {avg_anime_count:.2f}'
    
    # 通常のヒストグラム
    plt.figure(figsize=(12, 8))
    plt.hist(counts, bins=range(1, max_anime_count + 2), align='left', rwidth=0.8)
    plt.title('ユーザがツイートしたアニメ作品数の分布')
    plt.xlabel('ツイートしたアニメ作品数')
    plt.ylabel('ユーザ数')
    plt.xticks(range(0, max_anime_count + 1, 10), rotation=45)
    plt.grid(axis='y', alpha=0.75)
    
    plt.text(0.95, 0.95, stats_text,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(output_png_normal, dpi=300)
    plt.close()
    print(f"通常のヒストグラムを{output_png_normal}に保存しました。")
    
    # ログスケールのヒストグラム
    plt.figure(figsize=(12, 8))
    plt.hist(counts, bins=range(1, max_anime_count + 2), align='left', rwidth=0.8)
    plt.title('ユーザがツイートしたアニメ作品数の分布 (ログスケール)')
    plt.xlabel('ツイートしたアニメ作品数')
    plt.ylabel('ユーザ数 (ログスケール)')
    plt.xticks(range(0, max_anime_count + 1, 10), rotation=45)
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    
    plt.text(0.95, 0.95, stats_text,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(output_png_log, dpi=300)
    plt.close()
    print(f"ログスケールのヒストグラムを{output_png_log}に保存しました。")

    return total_users, max_anime_count, avg_anime_count

def save_to_csv(user_anime_data, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id', 'anime_count', 'anime_list'])
        for user_id, anime_set in user_anime_data.items():
            writer.writerow([user_id, len(anime_set), ','.join(sorted(anime_set))])
    print(f"データを{output_csv}に保存しました。")

def main():
    directory = 'tweet_user_profile_all_works'
    output_png_normal = 'plot/plot_count_works_by_tweet_user_histogram_all_works.png'
    output_png_log = 'plot/plot_count_works_by_tweet_user_histogram_log_all_works.png'
    output_csv = 'data/count_works_by_tweet_user_all_works.csv'
    
    user_anime_data = get_user_anime_data(directory)
    total_users, max_anime_count, avg_anime_count = plot_histograms(user_anime_data, output_png_normal, output_png_log)
    save_to_csv(user_anime_data, output_csv)
    
    print(f"総ユーザー数: {total_users:,}")
    print(f"最大ツイートアニメ数: {max_anime_count}")
    print(f"平均ツイートアニメ数: {avg_anime_count:.2f}")

if __name__ == "__main__":
    main()