import pandas as pd
import os
from collections import defaultdict
import json

def merge_jsonl_files(directory_path, output_path):
    # すべてのJSONLファイルのリストを取得するのだ
    jsonl_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.jsonl')]
    
    # すべてのデータフレームをリストに読み込むのだ
    dataframes = [pd.read_json(file, lines=True) for file in jsonl_files]

    merged_df = pd.read_json(jsonl_files[0], lines=True)

    for file in jsonl_files[1:]:
        df = pd.read_json(file, lines=True)
        merged_df = pd.merge(merged_df, df, on='user_id', how='outer')
        del df  # 新しく読み込んだデータフレームを削除してメモリを解放するのだ
    
    # 日別のツイートカウントのデータフレームを取得するのだ
    daily_tweet_counts_df = merged_df.drop(columns='user_id')

    # ユーザーごとのツイートカウントの合計、平均、および中央値を計算するのだ
    merged_df['total_tweet_count'] = daily_tweet_counts_df.sum(axis=1)
    merged_df['mean_tweet_count'] = daily_tweet_counts_df.mean(axis=1)
    merged_df['median_tweet_count'] = daily_tweet_counts_df.median(axis=1)

    # 結果をJSONL形式で保存するのだ
    merged_df.to_json(output_path, orient='records', lines=True)


def merge_jsonl_files_2(directory_path, output_path):
    # ユーザIDをキーとして、ツイート数の合計を値とする辞書を初期化するのだ
    user_tweet_counts = defaultdict(int)
    
    # すべてのJSONLファイルのリストを取得するのだ
    jsonl_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.jsonl')]
    
    # 各JSONLファイルを順に処理するのだ
    for file in jsonl_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                user_id = data['user_id']
                tweet_count = sum(value for key, value in data.items() if key != 'user_id')
                user_tweet_counts[user_id] += tweet_count
    
    # 結果をJSONL形式で保存するのだ
    with open(output_path, 'w', encoding='utf-8') as f:
        for user_id, tweet_count in user_tweet_counts.items():
            f.write(json.dumps({'user_id': user_id, 'total_tweet_count': tweet_count}) + '\n')

# ディレクトリパスと出力ファイルパスを指定して関数を呼び出すのだ
merge_jsonl_files_2('user_anime_tweet_count', 'user_anime_tweet_count.jsonl')
