import os
import pandas as pd
from urllib.parse import urlparse
import json

def load_data(tweet_dir, df_class):
    anime_url_count = []

    for file_name in os.listdir(tweet_dir):
        if file_name.endswith('.jsonl'):
            id = os.path.splitext(file_name)[0]
            tweet_file_path = os.path.join(tweet_dir, file_name)
            df = pd.read_json(tweet_file_path, lines=True, dtype={'tweet_id': str})
            title = df_class.loc[id, 'title']

            # URLを抽出し集計
            url_counts = df['urls'].explode().dropna().apply(lambda x: urlparse(x).netloc).value_counts().to_dict()
            
            anime_url_count.append({
                "id": id,
                "title": title,
                "urls": url_counts
            })

    return anime_url_count

def save_as_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n')

def export_url_counts(anime_url_count, output_path):
    # 全アニメ作品のURLデータを統合
    total_url_counts = {}
    for anime in anime_url_count:
        for url, count in anime['urls'].items():
            total_url_counts[url] = total_url_counts.get(url, 0) + count

    # URLの出現回数でソート
    sorted_url_counts = sorted(total_url_counts.items(), key=lambda x: x[1], reverse=True)

    # TSVファイルにソートされたデータを出力
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('URL\tCount\n')
        for url, count in sorted_url_counts:
            file.write(f'{url}\t{count}\n')

    return total_url_counts

def print_top_10_sites(anime_url_count):
    total_url_counts = export_url_counts(anime_url_count, 'results/count_url_overall.tsv')

    # 出現回数が最も多い上位10位のサイトを特定
    top_10_sites = sorted(total_url_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # 上位10位のサイトとその出現回数を表示
    print("出現回数が最も多い上位10位のサイト:")
    for url, count in top_10_sites:
        print(f"{url}: {count}")

def main():
    tweet_dir = '/work/n213304/learn/anime_retweet_2/analyze_url/tweet_url'

    df_class = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col="id")

    anime_url_count = load_data(tweet_dir, df_class)
    save_as_jsonl(anime_url_count, 'results/count_url.jsonl')

    # 上位10位のサイトを表示する関数を呼び出し
    print_top_10_sites(anime_url_count)

if __name__ == "__main__":
    main()