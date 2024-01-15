import json
from pathlib import Path
import pandas as pd

def load_original_tweets_as_df(file_path):
    return pd.read_json(file_path, lines=True, dtype={'tweet_id': str})

def load_predictions_as_df(file_path):
    return pd.read_json(file_path)

def save_json(data: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# 元のツイートデータを読み込む
original_tweet_dir = "/work/n213304/learn/anime_retweet_2/extra_anime_tweet_text_kikan?/"

# prediction ディレクトリ内の各ファイルに対して処理を行う
prediction_dir = Path('/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/prediction copy')
for prediction_file in prediction_dir.glob('*.json'):
    print(f"{prediction_file} ...")
    original_fime_name = prediction_file.stem
    original_tweets_df = load_original_tweets_as_df(original_tweet_dir + original_fime_name+'.jsonl')
    predictions_df = load_predictions_as_df(prediction_file)
    print(len(original_tweets_df))
    print(len(predictions_df))

    # 行数が同じであることを確認
    assert len(original_tweets_df) == len(predictions_df)
    # tweet_id を置き換え
    predictions_df['tweet_id'] = original_tweets_df['tweet_id']
    
    # 修正したデータを保存
    fixed_file_path = prediction_dir / f'{prediction_file.stem}.json'
    data_dict = predictions_df.to_dict(orient='records')
    save_json(data_dict, fixed_file_path)

