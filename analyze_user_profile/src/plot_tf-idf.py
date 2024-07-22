import json
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def get_anime_info(anime_info, id):
    title = anime_info.loc[id, 'title']
    weekly_tweet_user_clusters = anime_info.loc[id, 'weekly_tweet_user_clusters']
    mean_tweet_user_clusters = anime_info.loc[id, 'mean_tweet_user_clusters']
    
    w_labels = ["上昇", "下降", "山型", "横ばい"]
    m_labels = ["多い", "中くらい", "低い"]

    return title, w_labels[weekly_tweet_user_clusters], m_labels[mean_tweet_user_clusters]

def process_file(file_path):
    unique_users = set()
    profiles = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data.get('user_id')
            if user_id not in unique_users:
                nouns = data.get('extracted_nouns', [])
                profiles.append(' '.join(nouns))
                unique_users.add(user_id)
    
    return len(unique_users), profiles

def collect_data(input_dir, anime_info):
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            id = filename.split('.')[0]
            title, w_label, m_label = get_anime_info(anime_info, id)
            
            file_path = os.path.join(input_dir, filename)
            user_count, profiles = process_file(file_path)
            
            results.append({
                'title': title,
                'user_count': user_count,
                'profiles': profiles,
                'w_label': w_label,
                'm_label': m_label
            })
    return pd.DataFrame(results)

def analyze_rare_words(input_dir, anime_info, top_n=10, min_df=2):
    # データの収集
    df = collect_data(input_dir, anime_info)
    
    # すべてのプロフィールを結合
    all_profiles = [' '.join(profile) for profile in df['profiles']]
    
    # TF-IDF vectorizerの設定
    tfidf = TfidfVectorizer(min_df=min_df)  # 最低2回出現する単語のみを考慮
    tfidf_matrix = tfidf.fit_transform(all_profiles)
    
    # 各単語のTF-IDFスコアの平均を計算
    word_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    word_scores = word_scores / tfidf_matrix.shape[0]  # ドキュメント数で割って正規化
    
    # 単語とスコアのペアを作成
    word_score_pairs = list(zip(tfidf.get_feature_names_out(), word_scores))
    
    # スコアで降順ソート
    word_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # 上位N個のレアな単語を取得
    rare_words = word_score_pairs[:top_n]
    
    return rare_words, df

# メイン処理
def main(input_dir, output_file, anime_info):
    rare_words, df = analyze_rare_words(input_dir, anime_info)
    
    # 結果の出力
    print("Top 10 rare words:")
    for word, score in rare_words:
        print(f"{word}: {score}")
    
    # 詳細なデータをCSVファイルに出力
    df_rare_words = pd.DataFrame(rare_words, columns=['word', 'tfidf_score'])
    df_rare_words.to_csv(output_file, index=False, encoding='utf-8')

if __name__ == '__main__':
    input_directory = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/tweet_user_profile_wakachi'
    output_file = '/work/n213304/learn/anime_retweet_2/analyze_user_entropy/data/rare_words.csv'
    anime_info = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col=0)
    main(input_directory, output_file, anime_info)