import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib
import csv

def get_user_anime_data(directory):
    user_anime_data = defaultdict(lambda: {'tweet': set(), 'profile': False})
    
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            anime_id = os.path.splitext(filename)[0]
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    user_id = data['user_id']
                    user_anime_data[user_id]['tweet'].add(anime_id)
                    
                    # プロフィールにアニメという文字列があるかチェック
                    profile = data.get('profile', '')
                    if 'アニメ' in profile:
                        user_anime_data[user_id]['profile'] = True
    
    return user_anime_data

def plot_histograms(user_anime_data, output_png_normal, output_png_log):
    counts_profile = [len(data['tweet']) for data in user_anime_data.values() if data['profile']]
    counts_no_profile = [len(data['tweet']) for data in user_anime_data.values() if not data['profile']]
    
    total_users = len(user_anime_data)
    profile_users = sum(1 for data in user_anime_data.values() if data['profile'])
    no_profile_users = total_users - profile_users
    max_anime_count = max(max(counts_profile, default=0), max(counts_no_profile, default=0))
    avg_anime_count = sum(len(data['tweet']) for data in user_anime_data.values()) / total_users
    
    stats_text = f'総ユーザー数: {total_users:,}\n'
    stats_text += f'プロフィールにアニメを含むユーザー数: {profile_users:,}\n'
    stats_text += f'プロフィールにアニメを含むユーザーの割合: {profile_users/total_users:,}\n'
    stats_text += f'プロフィールにアニメを含まないユーザー数: {no_profile_users:,}\n'
    stats_text += f'プロフィールにアニメを含まないユーザーの割合: {no_profile_users/total_users:,}\n'
    stats_text += f'最大ツイートアニメ数: {max_anime_count}\n'
    stats_text += f'平均ツイートアニメ数: {avg_anime_count:.2f}'
    
    for scale, output_png in [('normal', output_png_normal), ('log', output_png_log)]:
        plt.figure(figsize=(12, 8))
        plt.hist([counts_profile, counts_no_profile], bins=range(1, max_anime_count + 2), align='left', rwidth=0.8, label=['プロフィールにアニメあり', 'プロフィールにアニメなし'])
        plt.title(f'ユーザがツイートしたアニメ作品数の分布 {"(ログスケール)" if scale == "log" else ""}')
        plt.xlabel('ツイートしたアニメ作品数')
        plt.ylabel(f'ユーザ数 {"(ログスケール)" if scale == "log" else ""}')
        plt.xticks(range(0, max_anime_count + 1, 10), rotation=45)
        if scale == 'log':
            plt.yscale('log')
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        
        plt.text(0.95, 0.5, stats_text,
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"{'ログスケール' if scale == 'log' else '通常'}のヒストグラムを{output_png}に保存しました。")

    return stats_text

import numpy as np

def analyze_tweet_counts(user_anime_data):
    anime_profile_counts = [len(data['tweet']) for data in user_anime_data.values() if data['profile']]
    non_anime_profile_counts = [len(data['tweet']) for data in user_anime_data.values() if not data['profile']]
    all_counts = anime_profile_counts + non_anime_profile_counts
    
    total_users = len(user_anime_data)
    anime_profile_users = len(anime_profile_counts)
    non_anime_profile_users = len(non_anime_profile_counts)
    
    anime_profile_2plus = sum(1 for count in anime_profile_counts if count >= 2)
    non_anime_profile_2plus = sum(1 for count in non_anime_profile_counts if count >= 2)
    
    anime_profile_ratio = anime_profile_2plus / anime_profile_users if anime_profile_users > 0 else 0
    non_anime_profile_ratio = non_anime_profile_2plus / non_anime_profile_users if non_anime_profile_users > 0 else 0
    total_ratio = (anime_profile_2plus + non_anime_profile_2plus) / total_users
    
    # 平均、最大、中央値の計算
    avg_anime_count = np.mean(all_counts)
    max_anime_count = max(all_counts)
    median_anime_count = np.median(all_counts)
    
    avg_anime_profile = np.mean(anime_profile_counts)
    avg_non_anime_profile = np.mean(non_anime_profile_counts)
    median_anime_profile = np.median(anime_profile_counts)
    median_non_anime_profile = np.median(non_anime_profile_counts)
    max_anime_profile = max(anime_profile_counts)
    max_non_anime_profile = max(non_anime_profile_counts)
    
    stats_text = f'総ユーザー数: {total_users:,}\n'
    stats_text += f'プロフィールにアニメを含むユーザー数: {anime_profile_users:,}\n'
    stats_text += f'プロフィールにアニメを含まないユーザー数: {non_anime_profile_users:,}\n'
    stats_text += f'アニメ含むプロフィールで2作品以上ツイートしたユーザー数: {anime_profile_2plus:,}\n'
    stats_text += f'アニメ含むプロフィールで2作品以上ツイートしたユーザーの割合: {anime_profile_ratio:.2%}\n'
    stats_text += f'アニメ含まないプロフィールで2作品以上ツイートしたユーザー数: {non_anime_profile_2plus:,}\n'
    stats_text += f'アニメ含まないプロフィールで2作品以上ツイートしたユーザーの割合: {non_anime_profile_ratio:.2%}\n'
    stats_text += f'2作品以上ツイートしたユーザーの割合: {total_ratio:.2%}\n'
    stats_text += f'平均ツイートアニメ数 (全体): {avg_anime_count:.2f}\n'
    stats_text += f'平均ツイートアニメ数 (アニメ含むプロフィール): {avg_anime_profile:.2f}\n'
    stats_text += f'平均ツイートアニメ数 (アニメ含まないプロフィール): {avg_non_anime_profile:.2f}\n'
    stats_text += f'最大ツイートアニメ数: {max_anime_count}\n'
    stats_text += f'最大ツイートアニメ数 (アニメ含むプロフィール):: {max_anime_profile}\n'
    stats_text += f'最大ツイートアニメ数 (アニメ含まないプロフィール):: {max_non_anime_profile}\n'
    stats_text += f'中央値ツイートアニメ数 (全体): {median_anime_count:.2f}\n'
    stats_text += f'中央値ツイートアニメ数 (アニメ含むプロフィール): {median_anime_profile:.2f}\n'
    stats_text += f'中央値ツイートアニメ数 (アニメ含まないプロフィール): {median_non_anime_profile:.2f}'
    
    return stats_text

def save_to_csv(user_anime_data, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id', 'anime_count', 'profile_anime', 'anime_list'])
        for user_id, data in user_anime_data.items():
            writer.writerow([user_id, len(data['tweet']), 'Yes' if data['profile'] else 'No', ','.join(sorted(data['tweet']))])
    print(f"データを{output_csv}に保存しました。")

def main():
    directory = 'tweet_user_profile_all_works'
    output_png_normal = 'plot/plot_count_works_by_tweet_user_core_histogram.png'
    output_png_log = 'plot/plot_count_works_by_tweet_user_core_histogram_log.png'
    output_csv = 'data/count_works_by_tweet_user_core.csv'
    
    user_anime_data = get_user_anime_data(directory)
    stats_text = analyze_tweet_counts(user_anime_data)
    print(stats_text)
    
    # 他の関数呼び出しはそのまま保持
    stats_text = plot_histograms(user_anime_data, output_png_normal, output_png_log)
    save_to_csv(user_anime_data, output_csv)
    print(stats_text)

if __name__ == "__main__":
    main()
