import json
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import japanize_matplotlib
from collections import Counter
import pandas as pd

plt.rcParams['font.family'] = 'IPAexGothic'

# ワードクラウドを生成する関数
def generate_wordcloud(text, title):
    font_path = "/work/n213304/learn/anime_retweet_2/font/NotoSansJP-VariableFont_wght.ttf"
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path)
    wordcloud.generate_from_frequencies(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    return plt.gcf()

def get_anime_info(anime_info, id):
    title = anime_info.loc[id, 'title']
    weekly_tweet_user_clusters = anime_info.loc[id, 'weekly_tweet_user_clusters']
    mean_tweet_user_clusters = anime_info.loc[id, 'mean_tweet_user_clusters']
    
    w_labels = ["上昇", "下降", "山型", "横ばい"]
    m_labels = ["多い", "中くらい", "低い"]

    return title, w_labels[weekly_tweet_user_clusters], m_labels[mean_tweet_user_clusters]

# メイン処理
def main(input_dir, output_pdf, anime_info):
    with PdfPages(output_pdf) as pdf:
        for filename in os.listdir(input_dir):
            if filename.endswith('.jsonl'):
                id = filename.split('.')[0]
                title, w_label, m_label = get_anime_info(anime_info, id)
                
                file_path = os.path.join(input_dir, filename)
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
                
                if words:
                    fig = generate_wordcloud(words, f'{title}, {w_label}, {m_label}')
                    pdf.savefig(fig)
                    plt.close(fig)
                
                print(f'Processed {filename}')

if __name__ == '__main__':
    input_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/tweet_user_profile_noun_concat'  # 入力ディレクトリ
    output_pdf = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/wordclouds_noun_concat.pdf'  # 出力PDFファイル
    anime_info = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col=0)
    main(input_directory, output_pdf, anime_info)