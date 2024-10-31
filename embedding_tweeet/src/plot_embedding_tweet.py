import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from tqdm import tqdm
import time
from datetime import timedelta
import os

def load_embeddings(embedding_file_path: str, subset_size: int = None) -> tuple:
    """
    埋め込みベクトルを読み込み、必要に応じてサブセットを作成
    
    Args:
        embedding_file_path (str): 埋め込みベクトルが保存されたnpzファイルのパス
        subset_size (int, optional): 使用するデータ数。Noneの場合は全データを使用
    
    Returns:
        tuple: (embeddings, tweet_ids, processed_texts)
    """
    print("データ読み込み中...")
    data = np.load(embedding_file_path)
    embeddings = data['embeddings']
    tweet_ids = data['tweet_ids']
    processed_texts = data['processed_texts']
    
    print(f"元のデータ:")
    print(f"埋め込みベクトルの形状: {embeddings.shape}")
    print(f"ツイート数: {len(tweet_ids):,}")
    
    if subset_size is not None and subset_size < len(embeddings):
        # ランダムにサブセットを選択
        indices = np.random.choice(len(embeddings), subset_size, replace=False)
        embeddings = embeddings[indices]
        tweet_ids = tweet_ids[indices]
        processed_texts = processed_texts[indices]
        print(f"\nテスト用サブセット:")
        print(f"埋め込みベクトルの形状: {embeddings.shape}")
        print(f"ツイート数: {len(tweet_ids):,}")
    
    return embeddings, tweet_ids, processed_texts

def reduce_dimensions_umap(embeddings: np.ndarray, 
                         n_neighbors=15, 
                         min_dist=0.1,
                         metric='euclidean') -> np.ndarray:
    """
    UMAPで次元削減を行う
    
    Args:
        embeddings (np.ndarray): 元の埋め込みベクトル
        n_neighbors (int): 近傍点の数
        min_dist (float): 最小距離パラメータ
        metric (str): 距離メトリック
    
    Returns:
        np.ndarray: 2次元に削減された埋め込みベクトル
    """
    print("\nUMAPで次元削減を開始...")
    print(f"パラメータ設定:")
    print(f"- n_neighbors: {n_neighbors}")
    print(f"- min_dist: {min_dist}")
    print(f"- metric: {metric}")
    
    start_time = time.time()
    
    # UMAP実行（進捗バーはUMAP自体の実装に依存）
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    
    result = reducer.fit_transform(embeddings)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"UMAP完了: 処理時間 {timedelta(seconds=int(elapsed_time))}")
    
    return result

def plot_embeddings(reduced_embeddings: np.ndarray, 
                   title: str = "Tweet Embeddings Visualization",
                   save_path: str = None) -> None:
    """
    埋め込みベクトルの散布図を作成
    
    Args:
        reduced_embeddings (np.ndarray): 2次元に削減された埋め込みベクトル
        title (str): プロットのタイトル
        save_path (str, optional): 保存先のパス
    """
    print(f"\nUMAPの結果をプロット中...")
    
    # プロットのスタイル設定
    plt.style.use('default')
    plt.figure(figsize=(12, 8))
    
    # 散布図のプロット
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
               alpha=0.5, s=10, c='#1f77b4')  # matplotlib default blue
    
    # タイトルとラベルの設定
    plt.title(title + "\n(UMAP)", pad=20)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    
    # グリッドの追加
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # レイアウトの調整
    plt.tight_layout()
    
    if save_path:
        # 保存先ディレクトリの作成
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"プロットを{save_path}に保存しました")
    
    plt.show()

def main():
    # テスト用の設定
    TEST_MODE = False  # テストモードのフラグ
    SUBSET_SIZE = 10000  # テスト用のデータ数
    
    # 埋め込みファイルのパス
    embedding_file = '/work/n213304/learn/anime_retweet_2/embedding_tweeet/processed_data/tweet_embeddings.npz'
    output_dir = '/work/n213304/learn/anime_retweet_2/embedding_tweeet/result'
    
    try:
        # データの読み込み
        embeddings, tweet_ids, processed_texts = load_embeddings(
            embedding_file,
            subset_size=SUBSET_SIZE if TEST_MODE else None
        )
        
        # UMAPで次元削減
        reduced_umap = reduce_dimensions_umap(
            embeddings,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean'
        )
        
        # 結果の可視化
        save_path = os.path.join(
            output_dir,
            f'plot_tweet_embeddings_umap{"_test" if TEST_MODE else ""}.png'
        )
        
        plot_embeddings(
            reduced_umap,
            title=f"Tweet Embeddings (n={len(embeddings):,})",
            save_path=save_path
        )
        
        print("\n処理完了!")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()