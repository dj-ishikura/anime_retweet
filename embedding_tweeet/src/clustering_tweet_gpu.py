import numpy as np
import cupy as cp
from cuml.cluster import KMeans as cuKMeans
from cuml import metrics
from tqdm import tqdm
import time
from datetime import timedelta
import os

def load_embeddings(embedding_file_path: str):
    """埋め込みベクトルを読み込む"""
    print("データ読み込み中...")
    data = np.load(embedding_file_path)
    embeddings = data['embeddings']
    tweet_ids = data['tweet_ids']
    processed_texts = data['processed_texts']
    
    print(f"データ読み込み完了:")
    print(f"埋め込みベクトルの形状: {embeddings.shape}")
    print(f"ツイート数: {len(tweet_ids):,}")
    
    return embeddings, tweet_ids, processed_texts

def find_optimal_k_gpu(embeddings: np.ndarray, k_range: range, random_state: int = 42):
    """GPUを使用して最適なクラスタ数を探索"""
    print("\nGPUを使用してクラスタ数の最適化を開始...")
    
    # データをGPUに転送
    embeddings_gpu = cp.asarray(embeddings)
    
    inertias = {}
    silhouette_scores = {}
    
    for k in tqdm(k_range, desc="クラスタ数の評価"):
        # GPU版K-meansの実行
        kmeans = cuKMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(embeddings_gpu)
        
        # inertiaの取得
        inertias[k] = kmeans.inertia_
        
        # シルエットスコアの計算（k>1の場合のみ）
        if k > 1:
            labels_gpu = kmeans.labels_
            silhouette_scores[k] = metrics.silhouette_score(
                embeddings_gpu, 
                labels_gpu,
                random_state=random_state
            )
    
    # 最適なkの決定（シルエットスコアが最大のk）
    optimal_k = max(silhouette_scores.items(), key=lambda x: x[1])[0]
    
    return optimal_k, inertias, silhouette_scores

def perform_clustering_gpu(embeddings: np.ndarray, n_clusters: int, random_state: int = 42):
    """GPUを使用してクラスタリングを実行"""
    print(f"\nGPUでクラスタリングを実行 (k={n_clusters})...")
    start_time = time.time()
    
    # データをGPUに転送
    embeddings_gpu = cp.asarray(embeddings)
    
    # クラスタリングの実行
    kmeans = cuKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings_gpu)
    
    # ラベルをCPUに転送
    labels_cpu = labels.get()
    
    end_time = time.time()
    print(f"クラスタリング完了: 処理時間 {timedelta(seconds=int(end_time - start_time))}")
    
    return labels_cpu

def main():
    # パラメータ設定
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 20
    RANDOM_STATE = 42
    
    # パスの設定
    embedding_file = '/work/n213304/learn/anime_retweet_2/embedding_tweeet/processed_data/tweet_embeddings.npz'
    output_dir = '/work/n213304/learn/anime_retweet_2/embedding_tweeet/result/clustering'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # データの読み込み
        embeddings, tweet_ids, processed_texts = load_embeddings(embedding_file)
        
        # GPUメモリの状態を表示
        print("\nGPUメモリ使用状況:")
        print(f"総メモリ: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.2f} GB")
        print(f"空きメモリ: {cp.cuda.runtime.memGetInfo()[0] / 1e9:.2f} GB")
        
        start_time = time.time()
        
        try:
            # 最適なクラスタ数の決定
            optimal_k, inertias, silhouette_scores = find_optimal_k_gpu(
                embeddings,
                range(MIN_CLUSTERS, MAX_CLUSTERS + 1),
                RANDOM_STATE
            )
            
            print(f"\n最適なクラスタ数: {optimal_k}")
            print(f"シルエットスコア: {silhouette_scores[optimal_k]:.3f}")
            
            # 最適なkでクラスタリングを実行
            labels = perform_clustering_gpu(embeddings, optimal_k, RANDOM_STATE)
            
            # 結果の保存
            output_file = os.path.join(output_dir, 'clustering_results_gpu.npz')
            np.savez(
                output_file,
                labels=labels,
                optimal_k=optimal_k,
                silhouette_scores=silhouette_scores[optimal_k]
            )
            print(f"\n結果を保存: {output_file}")
            
        finally:
            # GPUメモリの解放
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        
        end_time = time.time()
        print(f"\n総処理時間: {timedelta(seconds=int(end_time - start_time))}")
        print("処理完了!")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()