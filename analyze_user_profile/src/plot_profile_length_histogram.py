import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def get_profile_length(filename):
    profiles = {}
    with open(filename, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                user_id = data['user_id']
                profile = data['profile']
                profiles[user_id] = profile
            except json.JSONDecodeError:
                continue
    return profiles

def gather_profile_lengths(directory, output_txt):
    all_profiles = {}
    
    # user_profileディレクトリ内の全ての.jsonlファイルを処理
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory, filename)
            file_profiles = get_profile_length(file_path)
            all_profiles.update(file_profiles)

    # プロフィールの長さを計算
    lengths = [len(profile) for profile in all_profiles.values()]
    
    # ヒストグラムのデータを保存
    with open(output_txt, 'w') as f:
        for length in lengths:
            f.write(f"{length}\n")
    
    return lengths, all_profiles

def plot_profile_histgram(lengths, all_profiles, output_png):
    # ヒストグラムを描画
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title('Distribution of Profile Lengths')
    plt.xlabel('Profile Length')
    plt.ylabel('Frequency')
    plt.savefig(output_png)
    plt.close()

    print(f"処理されたユニークユーザー数: {len(all_profiles)}")
    print(f"最短プロフィール長: {min(lengths)}")
    print(f"最長プロフィール長: {max(lengths)}")
    print("ヒストグラムは profile_length_histogram.png として保存されました")
    print("プロフィール長データは profile_lengths.txt に保存されました")

def main():
    directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/user_profile'
    output_txt = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/data/profile_lengths.txt'

    lengths, all_profiles = gather_profile_lengths(directory, output_txt)
    
    output_png = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/profile_length_histogram.png'
    plot_profile_histgram(lengths, all_profiles, output_png)


if __name__ == "__main__":
    main()