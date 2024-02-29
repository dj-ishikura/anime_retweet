import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 日本語対応
import japanize_matplotlib  # 日本語ラベル対応

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# データを読み込む
data = load_jsonl('/work/n213304/learn/anime_retweet_2/analyze_url/results/count_url.jsonl')

for anime in data:
    total_urls = sum(anime['urls'].values())
    sorted_urls = sorted(anime['urls'].items(), key=lambda item: item[1], reverse=True)
    top_10_urls = sorted_urls[:10]
    others = total_urls - sum(value for _, value in top_10_urls)

    # 円グラフのデータ準備
    labels = [url[0] for url in top_10_urls] + ['その他']
    sizes = [url[1] for url in top_10_urls] + [others]

    # 円グラフのプロット
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # 円形を保持

    plt.title(anime['title'])
    plt.savefig(f'/work/n213304/learn/anime_retweet_2/analyze_url/plot_url/{anime["id"]}.png')
    plt.close()