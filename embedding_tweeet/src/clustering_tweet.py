import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import time
from datetime import timedelta
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

def plot_optimization_metrics(k_range: range, 
                            evaluation_results: dict, 
                            output_dir: str,
                            optimal_k: int):
    """
    エルボー法とシルエットスコアの結果をプロット
    
    Args:
        k_range (range): 評価したkの範囲
        evaluation_results (dict): 各kに対する評価結果
        output_dir (str): 出力ディレクトリ
        optimal_k (int): 選択された最適なk
    """
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # データの準備
    k_values = list(k_range)
    inertias = [evaluation_results[k]['inertia'] for k in k_values]
    silhouette_scores = [evaluation_results[k]['silhouette_score'] for k in k_values]
    
    # エルボー曲線のプロット
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Selected k={optimal_k}')
    ax1.set_title('Elbow Method', fontsize=14, pad=15)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 変化率の計算とプロット（2次曲線）
    inertia_changes = np.diff(inertias)
    relative_changes = -inertia_changes / inertias[:-1] * 100
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(k_values[1:], relative_changes, 'g.-', alpha=0.5, label='Change Rate (%)')
    ax1_twin.set_ylabel('Relative Change (%)', color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    
    # シルエットスコアのプロット
    ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Selected k={optimal_k}')
    ax2.set_title('Silhouette Score', fontsize=14, pad=15)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # レイアウトの調整
    plt.tight_layout()
    
    # グラフの保存
    plt.savefig(os.path.join(output_dir, 'elbow_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 数値データをCSVとして保存
    metrics_df = pd.DataFrame({
        'k': k_values,
        'inertia': inertias,
        'silhouette_score': silhouette_scores,
        'relative_change': [0] + relative_changes.tolist()  # 最初のkには変化率なし
    })
    metrics_df.to_csv(os.path.join(output_dir, 'clustering_metrics.csv'), index=False)
    
    # 詳細な分析結果をテキストファイルとして保存
    with open(os.path.join(output_dir, 'elbow_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("Clustering Optimization Analysis\n")
        f.write("==============================\n\n")
        f.write(f"Optimal number of clusters (k): {optimal_k}\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write("------------------\n")
        for k in k_values:
            f.write(f"\nk = {k}:\n")
            f.write(f"  Inertia: {evaluation_results[k]['inertia']:.2f}\n")
            f.write(f"  Silhouette Score: {evaluation_results[k]['silhouette_score']:.3f}\n")
            if k > k_values[0]:
                idx = k - k_values[0] - 1
                f.write(f"  Relative Change: {relative_changes[idx]:.1f}%\n")
        
        f.write("\nAnalysis Summary:\n")
        f.write("-----------------\n")
        
        # 最大の変化率が発生したポイントを特定
        max_change_k = k_values[1:][np.argmax(relative_changes)]
        f.write(f"Largest relative change occurs at k = {max_change_k}\n")
        
        # シルエットスコアの最大値
        max_silhouette_k = k_values[np.argmax(silhouette_scores)]
        f.write(f"Highest silhouette score occurs at k = {max_silhouette_k}\n")

def update_find_optimal_k_with_sample(embeddings: np.ndarray, 
                                    k_range: range,
                                    sample_size: int = 10000,
                                    random_state: int = 42) -> tuple:
    """
    サンプルデータを使用して最適なクラスタ数を探索（エルボー法の分析を含む）
    """
    print(f"\nサンプルデータ({sample_size:,}件)で最適なクラスタ数を探索...")
    
    if sample_size < len(embeddings):
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
    
    results = {}
    
    for k in tqdm(k_range, desc="クラスタ数の評価"):
        kmeans = MiniBatchKMeans(n_clusters=k, 
                                random_state=random_state,
                                batch_size=1024,
                                n_init=10)
        labels = kmeans.fit_predict(sample_embeddings)
        
        results[k] = {
            'inertia': float(kmeans.inertia_),
            'silhouette_score': float(silhouette_score(sample_embeddings, 
                                                     labels,
                                                     sample_size=min(5000, len(sample_embeddings))))
            if k > 1 else 0
        }
    
    # シルエットスコアに基づく最適なkの決定
    optimal_k = max([(k, info['silhouette_score']) for k, info in results.items()
                    if k > 1], key=lambda x: x[1])[0]
    
    return optimal_k, results[optimal_k]['silhouette_score'], results

def load_embeddings(embedding_file_path: str, sample_size: int = None):
    """埋め込みベクトルを読み込む（オプションでサンプリング）"""
    print("データ読み込み中...")
    data = np.load(embedding_file_path)
    embeddings = data['embeddings']
    tweet_ids = data['tweet_ids']
    processed_texts = data['processed_texts']
    
    if sample_size and sample_size < len(embeddings):
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        tweet_ids = tweet_ids[indices]
        processed_texts = processed_texts[indices]
    
    print(f"データ読み込み完了:")
    print(f"埋め込みベクトルの形状: {embeddings.shape}")
    print(f"ツイート数: {len(tweet_ids):,}")
    
    return embeddings, tweet_ids, processed_texts

def find_optimal_k_with_sample(embeddings: np.ndarray, 
                             k_range: range,
                             sample_size: int = 10000,
                             random_state: int = 42) -> Tuple[int, float, Dict]:
    """サンプルデータを使用して最適なクラスタ数を探索"""
    print(f"\nサンプルデータ({sample_size:,}件)で最適なクラスタ数を探索...")
    
    if sample_size < len(embeddings):
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
    
    results = {}
    
    for k in tqdm(k_range, desc="クラスタ数の評価"):
        kmeans = MiniBatchKMeans(n_clusters=k, 
                                random_state=random_state,
                                batch_size=1024,
                                n_init=10)
        labels = kmeans.fit_predict(sample_embeddings)
        
        results[k] = {
            'inertia': float(kmeans.inertia_),
            'silhouette_score': float(silhouette_score(sample_embeddings, 
                                                     labels,
                                                     sample_size=min(5000, len(sample_embeddings))))
            if k > 1 else 0
        }
    
    # 最適なkの決定（シルエットスコアが最大のk）
    optimal_k = max([(k, info['silhouette_score']) for k, info in results.items()
                    if k > 1], key=lambda x: x[1])[0]
    
    return optimal_k, results[optimal_k]['silhouette_score'], results

def perform_clustering_minibatch(embeddings: np.ndarray,
                               tweet_ids: np.ndarray,
                               processed_texts: np.ndarray,
                               n_clusters: int,
                               batch_size: int = 1024,
                               random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """MiniBatchKMeansを使用してクラスタリングを実行"""
    print(f"\nクラスタリングを実行 (k={n_clusters})...")
    print(f"バッチサイズ: {batch_size:,}")
    
    start_time = time.time()
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                            random_state=random_state,
                            batch_size=batch_size,
                            n_init=10)
    
    # バッチ処理でフィット
    total_batches = len(embeddings) // batch_size + (1 if len(embeddings) % batch_size != 0 else 0)
    
    with tqdm(total=total_batches, desc="クラスタリング進行状況") as pbar:
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            kmeans.partial_fit(batch)
            pbar.update(1)
    
    # 最終的なラベルの予測
    labels = kmeans.predict(embeddings)
    
    # クラスタの統計情報を収集
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_stats[cluster_id] = {
            'size': int(np.sum(cluster_mask)),
            'percentage': float(np.mean(cluster_mask) * 100),
            'sample_tweets': processed_texts[cluster_mask][:5].tolist()
        }
    
    end_time = time.time()
    print(f"クラスタリング完了: 処理時間 {timedelta(seconds=int(end_time - start_time))}")
    
    return labels, cluster_stats

def save_results(output_dir: str,
                labels: np.ndarray,
                tweet_ids: np.ndarray,
                processed_texts: np.ndarray,
                optimal_k: int,
                evaluation_results: Dict,
                cluster_stats: Dict):
    """結果を保存"""
    # 1. メタデータとクラスタ統計をJSONで保存
    metadata = {
        'optimal_k': optimal_k,
        'total_tweets': len(labels),
        'clustering_evaluation': evaluation_results,
        'cluster_statistics': cluster_stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'clustering_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 2. クラスタリング結果をCSVで保存
    results_df = pd.DataFrame({
        'tweet_id': tweet_ids,
        'text': processed_texts,
        'cluster': labels
    })
    results_df.to_csv(os.path.join(output_dir, 'clustering_results.csv'), 
                      index=False, encoding='utf-8')
    
    # 3. クラスタサイズの分布を可視化
    plt.figure(figsize=(12, 6))
    cluster_sizes = [stats['size'] for stats in cluster_stats.values()]
    plt.bar(range(len(cluster_sizes)), cluster_sizes)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Tweets')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. サマリーテキストファイルの作成
    with open(os.path.join(output_dir, 'clustering_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Clustering Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Performed at: {metadata['timestamp']}\n")
        f.write(f"Optimal number of clusters: {optimal_k}\n")
        f.write(f"Total tweets analyzed: {len(labels):,}\n\n")
        
        f.write("Cluster Statistics:\n")
        for cluster_id, stats in cluster_stats.items():
            f.write(f"\nCluster {cluster_id}:\n")
            f.write(f"  Size: {stats['size']:,} tweets\n")
            f.write(f"  Percentage: {stats['percentage']:.1f}%\n")
            f.write("  Sample tweets:\n")
            for i, tweet in enumerate(stats['sample_tweets'][:3], 1):
                f.write(f"    {i}. {tweet[:100]}...\n")

def main():
    # パラメータ設定
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 100
    SAMPLE_SIZE = 10000
    BATCH_SIZE = 1024
    RANDOM_STATE = 42
    
    # パスの設定
    embedding_file = '/work/n213304/learn/anime_retweet_2/embedding_tweeet/processed_data/tweet_embeddings.npz'
    output_dir = '/work/n213304/learn/anime_retweet_2/embedding_tweeet/result/clustering'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # データの読み込み
        embeddings, tweet_ids, processed_texts = load_embeddings(embedding_file)
        
        # 最適なkの探索
        optimal_k, best_score, evaluation_results = update_find_optimal_k_with_sample(
            embeddings,
            range(MIN_CLUSTERS, MAX_CLUSTERS + 1),
            SAMPLE_SIZE,
            RANDOM_STATE
        )
        
        # エルボー法の分析結果をプロット
        plot_optimization_metrics(
            range(MIN_CLUSTERS, MAX_CLUSTERS + 1),
            evaluation_results,
            output_dir,
            optimal_k
        )
        
        print(f"\n最適なクラスタ数: {optimal_k}")
        print(f"サンプルデータでのシルエットスコア: {best_score:.3f}")
        print(f"\nエルボー法の分析結果が{output_dir}に保存されました:")
        print("- elbow_analysis.png: エルボー曲線とシルエットスコアのグラフ")
        print("- clustering_metrics.csv: 詳細な評価指標データ")
        print("- elbow_analysis.txt: 分析レポート")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()