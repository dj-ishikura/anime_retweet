import csv
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from scipy import stats

def load_user_anime_data(csv_file):
    anime_profile_counts = []
    non_anime_profile_counts = []
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            anime_count = int(row['anime_count'])
            if row['profile_anime'] == 'Yes':
                anime_profile_counts.append(anime_count)
            else:
                non_anime_profile_counts.append(anime_count)
    
    return anime_profile_counts, non_anime_profile_counts

def analyze_data(anime_profile_counts, non_anime_profile_counts):
    total_users = len(anime_profile_counts) + len(non_anime_profile_counts)
    anime_profile_users = len(anime_profile_counts)
    non_anime_profile_users = len(non_anime_profile_counts)

    avg_anime_profile = np.mean(anime_profile_counts)
    avg_non_anime_profile = np.mean(non_anime_profile_counts)
    median_anime_profile = np.median(anime_profile_counts)
    median_non_anime_profile = np.median(non_anime_profile_counts)

    t_statistic, p_value = stats.ttest_ind(anime_profile_counts, non_anime_profile_counts)

    return {
        'total_users': total_users,
        'anime_profile_users': anime_profile_users,
        'non_anime_profile_users': non_anime_profile_users,
        'avg_anime_profile': avg_anime_profile,
        'avg_non_anime_profile': avg_non_anime_profile,
        'median_anime_profile': median_anime_profile,
        'median_non_anime_profile': median_non_anime_profile,
        't_statistic': t_statistic,
        'p_value': p_value,
        'anime_profile_counts': anime_profile_counts,
        'non_anime_profile_counts': non_anime_profile_counts
    }

def plot_results(results, output_file):
    plt.figure(figsize=(12, 6))
    
    # 箱ひげ図
    plt.subplot(121)
    plt.boxplot([results['anime_profile_counts'], results['non_anime_profile_counts']], 
                labels=['アニメあり', 'アニメなし'])
    plt.title('プロフィールのアニメ有無による投稿アニメ作品数の分布')
    plt.ylabel('投稿アニメ作品数')
    plt.yscale('log')

    # バイオリンプロット
    plt.subplot(122)
    plt.violinplot([results['anime_profile_counts'], results['non_anime_profile_counts']], 
                   showmeans=True, showmedians=True)
    plt.xticks([1, 2], ['アニメあり', 'アニメなし'])
    plt.title('プロフィールのアニメ有無による投稿アニメ作品数の分布 (バイオリンプロット)')
    plt.ylabel('投稿アニメ作品数')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def main():
    input_file = 'data/count_works_by_tweet_user_core.csv'
    output_file = 'plot/plot_conpare_count_works_by_core_and_non.png'

    anime_profile_counts, non_anime_profile_counts = load_user_anime_data(input_file)
    results = analyze_data(anime_profile_counts, non_anime_profile_counts)
    plot_results(results, output_file)

    print(f"総ユーザー数: {results['total_users']:,}")
    print(f"プロフィールにアニメを含むユーザー数: {results['anime_profile_users']:,} ({results['anime_profile_users']/results['total_users']:.2%})")
    print(f"プロフィールにアニメを含まないユーザー数: {results['non_anime_profile_users']:,} ({results['non_anime_profile_users']/results['total_users']:.2%})")
    print(f"アニメ含むプロフィールの平均投稿アニメ作品数: {results['avg_anime_profile']:.2f}")
    print(f"アニメ含まないプロフィールの平均投稿アニメ作品数: {results['avg_non_anime_profile']:.2f}")
    print(f"アニメ含むプロフィールの中央値投稿アニメ作品数: {results['median_anime_profile']:.2f}")
    print(f"アニメ含まないプロフィールの中央値投稿アニメ作品数: {results['median_non_anime_profile']:.2f}")
    print(f"t検定結果: t統計量 = {results['t_statistic']:.4f}, p値 = {results['p_value']:.4e}")

if __name__ == "__main__":
    main()