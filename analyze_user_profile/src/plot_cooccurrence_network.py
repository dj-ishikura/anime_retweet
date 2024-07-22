import json
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import japanize_matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

# フォントパスの指定
font_path = "/work/n213304/learn/anime_retweet_2/font/NotoSansJP-VariableFont_wght.ttf"
font_prop = FontProperties(fname=font_path)

def get_anime_info(anime_info, id):
    title = anime_info.loc[id, 'title']
    weekly_tweet_user_clusters = anime_info.loc[id, 'weekly_tweet_user_clusters']
    mean_tweet_user_clusters = anime_info.loc[id, 'mean_tweet_user_clusters']
    
    w_labels = ["上昇", "下降", "山型", "横ばい"]
    m_labels = ["多い", "中くらい", "低い"]

    return title, w_labels[weekly_tweet_user_clusters], m_labels[mean_tweet_user_clusters]

def process_file(file_path):
    nouns = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            nouns.extend(data.get('extracted_nouns', []))
    return nouns

def create_cooccurrence_network(nouns, min_edge_weight=5):
    cooccurrence = Counter()
    for sentence in nouns:
        for pair in combinations(sentence, 2):
            cooccurrence[tuple(sorted(pair))] += 1
    
    G = nx.Graph()
    for (word1, word2), count in cooccurrence.items():
        if count >= min_edge_weight:
            G.add_edge(word1, word2, weight=count)
    
    return G

def plot_network(G, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[w/5 for w in edge_weights], alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=16, font_family='IPAexGothic')
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def main(input_dir, output_dir, anime_info):
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_path = os.path.join(output_dir, 'cooccurrence_networks.pdf')
    with PdfPages(pdf_path) as pdf:
        for filename in os.listdir(input_dir):
            if filename.endswith('.jsonl'):
                id = filename.split('.')[0]
                title, w_label, m_label = get_anime_info(anime_info, id)
                
                file_path = os.path.join(input_dir, filename)
                nouns = process_file(file_path)
                
                G = create_cooccurrence_network(nouns)
                
                fig = plot_network(G, f'{title}\n{w_label}, {m_label}')
                pdf.savefig(fig)
                plt.close(fig)
                
                # 個別のPNG画像も保存
                png_path = os.path.join(output_dir, f'{id}_cooccurrence_network.png')
                fig.savefig(png_path)
                
                print(f'Processed {filename}')
            break

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()] + plt.rcParams['font.sans-serif']
  
    input_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/tweet_user_profile_wakachi'
    output_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/cooccurrence_networks'
    anime_info = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col=0)
    main(input_directory, output_directory, anime_info)