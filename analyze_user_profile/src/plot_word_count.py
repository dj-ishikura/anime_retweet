import json
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import japanize_matplotlib

plt.rcParams['font.family'] = 'IPAexGothic'

def get_anime_info(anime_info, id):
    title = anime_info.loc[id, 'title']
    weekly_tweet_user_clusters = anime_info.loc[id, 'weekly_tweet_user_clusters']
    mean_tweet_user_clusters = anime_info.loc[id, 'mean_tweet_user_clusters']
    
    w_labels = ["上昇", "下降", "山型", "横ばい"]
    m_labels = ["多い", "中くらい", "低い"]

    return title, w_labels[weekly_tweet_user_clusters], m_labels[mean_tweet_user_clusters]

def process_file(file_path):
    words = Counter()
    unique_users = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data.get('user_id')
            if user_id not in unique_users:
                nouns = data.get('extracted_nouns', [])
                words.update(nouns)
                unique_users.add(user_id)
    
    return words, len(unique_users)

def get_top_words(words, top_n=10):
    return words.most_common(top_n)


def plot_top_words(top_words, title, ax):
    words, counts = zip(*top_words)  # top_words はすでに望む順序になっているはず
    x_pos = range(len(words))
    
    ax.bar(x_pos, counts)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel('出現回数')
    
    # 各バーの上に出現回数を表示
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    # x軸のラベルが重ならないように調整
    plt.tight_layout()

def main(input_dir, output_json, output_pdf, anime_info):
    results = []
    
    with PdfPages(output_pdf) as pdf:
        for filename in os.listdir(input_dir):
            if filename.endswith('.jsonl'):
                id = filename.split('.')[0]
                title, w_label, m_label = get_anime_info(anime_info, id)
                
                file_path = os.path.join(input_dir, filename)
                words, user_count = process_file(file_path)
                
                top_words = get_top_words(words)
                
                result = {
                    'id': id,
                    'title': title,
                    'user_count': user_count,
                    'weekly_trend': w_label,
                    'mean_user_count': m_label,
                    'top_words': dict(top_words)
                }
                results.append(result)
                
                # 図の作成
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_top_words(top_words, f"{title}\n{w_label}, {m_label}", ax)
                pdf.savefig(fig)
                plt.close(fig)
                
                print(f'Processed {filename}')
    
    # 結果をJSONファイルに保存
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_json}")
    print(f"Plots saved to {output_pdf}")

if __name__ == '__main__':
    input_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/tweet_user_profile_wakachi'
    output_json = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/data/top_1_percent_words.json'
    output_pdf = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/plot_top_1_percent_words.pdf'
    anime_info = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col=0)
    main(input_directory, output_json, output_pdf, anime_info)