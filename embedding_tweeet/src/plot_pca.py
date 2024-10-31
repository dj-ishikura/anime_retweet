import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns
from scipy.stats import gaussian_kde
import os

def load_embeddings(embedding_file_path: str, n_samples: int = -1):
    """埋め込みベクトルを読み込む。
    Args:
        embedding_file_path: 埋め込みベクトルのファイルパス
        n_samples: サンプル数。-1の場合は全データを使用。
    """
    print("データ読み込み中...")
    data = np.load(embedding_file_path)
    embeddings = data['embeddings']
    tweet_ids = data['tweet_ids']
    processed_texts = data['processed_texts']
    
    total_samples = len(embeddings)
    
    if n_samples > 0 and n_samples < total_samples:
        # ランダムにサンプリング
        indices = np.random.choice(total_samples, n_samples, replace=False)
        embeddings = embeddings[indices]
        tweet_ids = tweet_ids[indices]
        processed_texts = processed_texts[indices]
        print(f"データをサンプリング: {n_samples} サンプル")
    else:
        n_samples = total_samples
        print("全データを使用")
    
    print(f"データ読み込み完了:")
    print(f"埋め込みベクトルの形状: {embeddings.shape}")
    
    return embeddings, tweet_ids, processed_texts, n_samples

def create_visualization(embeddings, tweet_ids, processed_texts, output_dir, n_samples, n_clusters=2):
    """PCAと可視化の実行"""
    # PCA
    print("PCAを実行中...")
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(embeddings)
    
    # クラスタリング
    print("クラスタリングを実行中...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(transformed)
    
    # プロットの作成
    print("可視化を作成中...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # サンプル数に応じてタイトルを設定
    if n_samples == len(embeddings):
        title = 'PCA Visualization Analysis (All Data)'
        file_suffix = 'all'
    else:
        title = f'PCA Visualization Analysis (n={n_samples:,})'
        file_suffix = f'n{n_samples}'
    
    fig.suptitle(title, fontsize=16)
    
    # 1. オリジナルの散布図
    ax = axes[0, 0]
    ax.scatter(transformed[:, 0], transformed[:, 1], 
              alpha=0.1, s=1, c='blue')
    ax.set_title('Original Scatter Plot')
    ax.grid(True, alpha=0.3)
    
    # 2. 密度プロット
    ax = axes[0, 1]
    hist = ax.hist2d(transformed[:, 0], transformed[:, 1], 
                     bins=100, cmap='viridis')
    plt.colorbar(hist[3], ax=ax)
    ax.set_title('Density Plot')
    ax.grid(True, alpha=0.3)
    
    # 3. クラスタリング結果
    ax = axes[1, 0]
    scatter = ax.scatter(transformed[:, 0], transformed[:, 1], 
                        c=clusters, alpha=0.5, s=1, cmap='tab10')
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], 
              c='red', marker='x', s=200, linewidths=3)
    ax.set_title('Clustering Results')
    ax.grid(True, alpha=0.3)
    
    # 4. 特徴的なポイントの抽出
    ax = axes[1, 1]
    ax.scatter(transformed[:, 0], transformed[:, 1], 
              alpha=0.1, s=1, c='gray')
    
    # 端のポイントを見つける
    edges = []
    for dim in [0, 1]:
        for func in [np.argmin, np.argmax]:
            idx = func(transformed[:, dim])
            edges.append(idx)
            ax.scatter(transformed[idx, 0], transformed[idx, 1], 
                      c='red', s=100)
            ax.annotate(f'Tweet {len(edges)}', 
                       (transformed[idx, 0], transformed[idx, 1]),
                       xytext=(10, 10), textcoords='offset points')
    
    ax.set_title('Notable Points')
    ax.grid(True, alpha=0.3)
    
    # プロットの保存
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'pca_analysis_{file_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 特徴的なツイートの内容を保存
    with open(os.path.join(output_dir, f'notable_tweets_{file_suffix}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Notable Tweets ({title}):\n\n")
        for i, idx in enumerate(edges, 1):
            f.write(f"Tweet {i} (位置: PC1={transformed[idx, 0]:.2f}, "
                   f"PC2={transformed[idx, 1]:.2f}):\n")
            f.write(f"Cluster: {clusters[idx]}\n")
            f.write(f"Text: {processed_texts[idx]}\n\n")
        
        # 各クラスタの中心付近のツイートも保存
        f.write("\nCluster Centers Examples:\n\n")
        for i in range(n_clusters):
            cluster_points = transformed[clusters == i]
            center = centers[i]
            # 中心に最も近い点を見つける
            distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
            center_idx = np.where(clusters == i)[0][np.argmin(distances)]
            
            f.write(f"Cluster {i} Center Example:\n")
            f.write(f"Text: {processed_texts[center_idx]}\n\n")

def main():
    embedding_file = '/work/n213304/learn/anime_retweet_2/embedding_tweeet/processed_data/tweet_embeddings.npz'
    output_dir = '/work/n213304/learn/anime_retweet_2/embedding_tweeet/result/pca'
    n_samples = -1  # -1 for all data, または具体的な数値を指定
    
    try:
        embeddings, tweet_ids, processed_texts, actual_samples = load_embeddings(embedding_file, n_samples)
        create_visualization(embeddings, tweet_ids, processed_texts, output_dir, actual_samples)
        print("\n処理完了!")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()