import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib
import csv
import numpy as np

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

def plot_cumulative_distribution(user_anime_data, output_png):
    counts = [len(anime_set) for anime_set in user_anime_data.values()]
    
    total_users = len(user_anime_data)
    max_anime_count = max(counts)
    avg_anime_count = sum(counts) / total_users
    
    # 累積分布を計算（修正版）
    hist, bin_edges = np.histogram(counts, bins=range(1, max_anime_count + 2))
    cumulative = np.cumsum(hist[::-1])[::-1] / total_users
    
    plt.figure(figsize=(12, 8))
    plt.plot(bin_edges[:-1], cumulative * 100, marker='o', markersize=3)  # パーセンテージに変換
    # plt.title('ツイートしたアニメ作品数の累積分布')
    plt.xlabel('ツイートしたアニメ作品数', fontsize=16)
    plt.ylabel('n作品以上ツイートしたユーザの割合 (%)', fontsize=16)
    plt.yscale('log')
    plt.xlim(left=1)  # x軸の最小値を1に設定
    plt.ylim(bottom=0.01, top=100)  # y軸の範囲を0.01%から100%に設定
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.xticks([1, 10, 20, 50, 100, 200, 500], rotation=45)
    
    # 特定の点にラベルを付ける
    for n in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        if n <= max_anime_count:
            idx = np.searchsorted(bin_edges, n)
            plt.annotate(f'{n}: {cumulative[idx]*100:.2f}%', 
                         xy=(n, cumulative[idx]*100), 
                         xytext=(5, 5), 
                         textcoords='offset points',
                         fontsize=16,
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    stats_text = f'総ユーザー数: {total_users:,}\n'
    stats_text += f'最大ツイートアニメ数: {max_anime_count}\n'
    stats_text += f'平均ツイートアニメ数: {avg_anime_count:.2f}'
    
    plt.text(0.95, 0.95, stats_text,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"累積分布グラフを{output_png}に保存しました。")

    return total_users, max_anime_count, avg_anime_count
def save_to_csv(user_anime_data, output_csv):
    # 既存の関数をそのまま使用
    ...

def main():
    directory = 'tweet_user_profile_all_works'
    output_png_cumulative = 'plot/plot_count_works_cumulative_all_works.png'
    output_csv = 'data/count_works_cumulative_all_works.csv'
    
    user_anime_data = get_user_anime_data(directory)
    total_users, max_anime_count, avg_anime_count = plot_cumulative_distribution(user_anime_data, output_png_cumulative)
    save_to_csv(user_anime_data, output_csv)
    
    print(f"総ユーザー数: {total_users:,}")
    print(f"最大ツイートアニメ数: {max_anime_count}")
    print(f"平均ツイートアニメ数: {avg_anime_count:.2f}")

if __name__ == "__main__":
    main()