import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import japanize_matplotlib
from datetime import datetime
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.decomposition import PCA
from collections import defaultdict
from matplotlib.colors import BoundaryNorm
import argparse

def create_directory_structure(base_path):
    """必要なディレクトリ構造を作成する"""
    # 現在の日時を取得し、分単位までのフォーマットに変換
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    dirs = {
        'results': os.path.join(base_path, 'results'),
        'models': os.path.join(base_path, 'models'),
        'clusters': os.path.join(base_path, 'results', f'clusters_{timestamp}')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def load_tweet_embeddings(input_path):
    """ツイートの埋め込みデータを読み込む"""
    data = np.load(input_path)
    embeddings = data['embeddings'][:100000]  
    tweet_ids = data['tweet_ids'][:100000]    
    processed_texts = data['processed_texts'][:100000]  
    return embeddings, tweet_ids, processed_texts

def cluster_tweets(embeddings, processed_texts, args):
    """ツイートをクラスタリングする"""
    umap_model = UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state
    )

    hdbscan_model = HDBSCAN(
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        prediction_data=args.prediction_data,
        gen_min_span_tree=args.gen_min_span_tree
    )

    topic_model = BERTopic(
        umap_model=umap_model, 
        hdbscan_model=hdbscan_model, 
        verbose=True
    )
    
    topics, probabilities = topic_model.fit_transform(processed_texts, embeddings=embeddings)
    return topics, probabilities, topic_model

def process_tweets_streaming(tweet_ids, processed_texts, topics, probabilities, dirs):
    """ツイートをストリーミング処理してクラスターごとのファイルに書き出す"""
    cluster_counts = defaultdict(int)
    
    # まずクラスターごとのツイートを収集
    cluster_tweets = defaultdict(list)
    for tweet_id, text, topic, prob in zip(tweet_ids, processed_texts, topics, 
                                         probabilities.max(axis=1) if len(probabilities.shape) > 1 else probabilities):
        cluster = int(topic)
        tweet_data = {
            'tweet_id': tweet_id.item() if hasattr(tweet_id, 'item') else tweet_id,
            'text': text,
            'cluster': cluster,
            'probability': float(prob)
        }
        cluster_tweets[cluster].append(tweet_data)
        cluster_counts[cluster] += 1
    
    # クラスターごとに一括で書き出し
    for cluster, tweets in cluster_tweets.items():
        filename = os.path.join(dirs['clusters'], f'cluster_{cluster:02d}.jsonl')
        with open(filename, 'w', encoding='utf-8') as f:
            for tweet in tweets:
                json.dump(tweet, f, ensure_ascii=False)
                f.write('\n')
    
    return cluster_counts

def main(args):
    # ディレクトリ構造の作成
    dirs = create_directory_structure(args.base_output_path)
    
    # ツイートの埋め込みデータを読み込む
    print("Loading data...")
    embeddings, tweet_ids, processed_texts = load_tweet_embeddings(args.input_path)
    print(f"Loaded {len(tweet_ids)} tweets for processing")

    # クラスタリングを実行
    print("Starting clustering...")
    topics, probabilities, topic_model = cluster_tweets(embeddings, processed_texts, args)
    print("Clustering completed")

    # ストリーミング処理でクラスターごとにファイル出力
    print("Processing tweets and saving to files...")
    cluster_counts = process_tweets_streaming(tweet_ids, processed_texts, topics, probabilities, dirs)

    # 確率値の保存
    prob_file = os.path.join(dirs['results'], 'topic_probabilities.npz')
    np.savez_compressed(prob_file, probabilities=probabilities)
    
    # トピックモデルの保存
    model_file = os.path.join(dirs['models'], 'topic_model')
    topic_model.save(model_file)
    
    # クラスター統計を表示
    print("\nCluster Statistics:")
    print(f"Number of clusters: {len(cluster_counts)}")
    print(f"Largest cluster size: {max(cluster_counts.values())}")
    print(f"Smallest cluster size: {min(cluster_counts.values())}")
    print(f"Average cluster size: {sum(cluster_counts.values()) / len(cluster_counts):.2f}")
    
    # 各クラスターのサイズを表示
    print("\nDetailed cluster sizes:")
    for cluster, count in sorted(cluster_counts.items()):
        print(f"Cluster {cluster:2d}: {count:5d} tweets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERTopic clustering with customizable UMAP and HDBSCAN parameters')
    
    # 必須のパス引数
    parser.add_argument('--input_path', type=str, default='/work/n213304/learn/anime_retweet_2/embedding_tweeet/processed_data/tweet_embeddings.npz',
                        help='Path to input embeddings file')
    parser.add_argument('--base_output_path', type=str, default='/work/n213304/learn/anime_retweet_2/BERTopic',
                        help='Base path for output files')
    
    # UMAPパラメータ
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--n_components', type=int, default=5,
                        help='UMAP n_components parameter')
    parser.add_argument('--min_dist', type=float, default=0.0,
                        help='UMAP min_dist parameter')
    parser.add_argument('--metric', type=str, default='cosine',
                        help='UMAP metric parameter')
    parser.add_argument('--random_state', type=int, default=0,
                        help='UMAP random_state parameter')
    
    # HDBSCANパラメータ
    parser.add_argument('--min_samples', type=int, default=10,
                        help='HDBSCAN min_samples parameter')
    parser.add_argument('--min_cluster_size', type=int, default=10,
                        help='HDBSCAN min_cluster_size parameter')
    parser.add_argument('--cluster_selection_epsilon', type=float, default=0.7,
                        help='HDBSCAN cluster_selection_epsilon parameter')
    parser.add_argument('--prediction_data', type=bool, default=True,
                        help='HDBSCAN prediction_data parameter')
    parser.add_argument('--gen_min_span_tree', type=bool, default=True,
                        help='HDBSCAN gen_min_span_tree parameter')

    args = parser.parse_args()
    main(args)