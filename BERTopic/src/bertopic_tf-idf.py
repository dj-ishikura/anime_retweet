import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict, Tuple
import os

def load_jsonl(file_path: str) -> List[Dict]:
    """JSONLファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def get_count_vectorizer() -> CountVectorizer:
    """BERTopicデフォルトのCountVectorizer設定を再現"""
    return CountVectorizer(
        lowercase=False,  # 日本語なので小文字化しない
        stop_words=None,  # ストップワードは使用しない
        min_df=2,        # 最低2回出現する単語を対象
        max_df=0.95,     # 95%以上の文書に出現する単語は除外
        token_pattern=None,  # すでにトークナイズ済みなのでパターンは不要
        tokenizer=lambda x: x.split()  # スペース区切りでトークン化
    )

def calculate_c_tf_idf(documents_per_topic: Dict[int, List[str]], 
                      vectorizer: CountVectorizer) -> Tuple[pd.DataFrame, np.ndarray]:
    """c-TF-IDF (class-based TF-IDF) を計算"""
    documents_per_topic = {
        topic: ' '.join(docs) for topic, docs in documents_per_topic.items()
    }
    
    topic_words = vectorizer.fit_transform(documents_per_topic.values())
    vocabulary = np.array(vectorizer.get_feature_names_out())
    word_frequencies = topic_words.toarray()
    n_samples_per_class = word_frequencies.sum(axis=1)
    total_samples = n_samples_per_class.sum()
    
    tf = word_frequencies / n_samples_per_class[:, np.newaxis]
    idf = np.log((total_samples + 1) / (word_frequencies.sum(axis=0) + 1)) + 1
    c_tf_idf = tf * idf
    
    c_tf_idf_scores = pd.DataFrame(
        c_tf_idf,
        index=list(documents_per_topic.keys()),
        columns=vocabulary
    )
    
    return c_tf_idf_scores, vocabulary

def get_top_n_words(c_tf_idf_scores: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    各クラスターでの重要度上位n個の単語を抽出しDataFrameとして返す
    
    Args:
        c_tf_idf_scores: c-TF-IDFスコアのDataFrame
        n: 抽出する単語数
    
    Returns:
        重要単語とそのスコアを含むDataFrame
    """
    rows = []
    for cluster in c_tf_idf_scores.index:
        words = c_tf_idf_scores.loc[cluster]
        top_n = words.nlargest(n)
        
        # 各単語とスコアをDataFrameの行として追加
        for word, score in zip(top_n.index, top_n.values):
            rows.append({
                'cluster': cluster,
                'word': word,
                'c_tf_idf': score,
            })
    
    # DataFrameを作成し、クラスターとスコアでソート
    df = pd.DataFrame(rows)
    df = df.sort_values(['cluster', 'c_tf_idf'], ascending=[True, False])
    return df

def save_results(top_words_df: pd.DataFrame, output_dir: str):
    """結果をTSVファイルとして保存"""
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # TSVファイルとして保存
    output_file = os.path.join(output_dir, 'cluster_c-tf-idf.tsv')
    top_words_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    print(f"\n特徴語をTSVファイルに保存しました: {output_file}")

def main(input_file: str, output_dir: str):
    print("データを読み込んでいます...")
    data = load_jsonl(input_file)
    
    print("クラスターごとに文書を整理しています...")
    documents_per_topic = {}
    for item in data:
        cluster = item['cluster']
        if cluster not in documents_per_topic:
            documents_per_topic[cluster] = []
        documents_per_topic[cluster].append(item['tokenized_text'])
    
    print("特徴語を抽出しています...")
    vectorizer = get_count_vectorizer()
    c_tf_idf_scores, vocabulary = calculate_c_tf_idf(documents_per_topic, vectorizer)
    
    print("各クラスターの重要単語を抽出しています...")
    top_words_df = get_top_n_words(c_tf_idf_scores, n=20)  # 上位20語を抽出
    
    # 結果を保存
    save_results(top_words_df, output_dir)
    
    # 結果の表示（確認用）
    print("\n各クラスターの特徴語:")
    for cluster in sorted(top_words_df['cluster'].unique()):
        print(f"\nCluster {cluster}:")
        cluster_words = top_words_df[top_words_df['cluster'] == cluster]
        for _, row in cluster_words.iterrows():
            print(f"{row['word']}: {row['c_tf_idf']:.4f}")

if __name__ == "__main__":
    input_file = "/work/n213304/learn/anime_retweet_2/BERTopic/results/clusters_wakachi.jsonl"
    output_dir = "/work/n213304/learn/anime_retweet_2/BERTopic/results"
    main(input_file, output_dir)