# -*- coding: utf-8 -*-
"""ツイートjsonからメモリ効率良くランダムに100件のデータを抽出するプログラム"""

import sys
import json
import random
import codecs
from typing import Dict, List, Iterator

from twitter import parse_tweet
from log import get_logger

logger = get_logger(__name__)

def process_tweet(tweet_json: Dict) -> Dict:
    """ツイートを処理し、適切なデータを返す"""
    if "retweeted_status" in tweet_json:
        # リツイートの場合、オリジナルツイートのデータを使用
        return tweet_json["retweeted_status"]
    return tweet_json

def reservoir_sampling(stream: Iterator[Dict], k: int) -> List[Dict]:
    """リザーバーサンプリングを使用してストリームからkサンプルを選択"""
    reservoir = []
    for i, item in enumerate(stream):
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

def extract_random_tweets(input_stream, num_tweets: int = 1000) -> List[Dict]:
    """入力ストリームからランダムに指定数のツイートを抽出する"""
    def tweet_stream():
        for tweet_json in parse_tweet(input_stream):
            try:
                yield process_tweet(tweet_json)
            except KeyError as e:
                logger.debug(f'KeyError: {e}')

    return reservoir_sampling(tweet_stream(), num_tweets)

def save_tweets(tweets: List[Dict], output_file: str):
    """ツイートをファイルに保存する"""
    with open(output_file, 'w', encoding='utf-8') as file:
        for tweet_json in tweets:
            tweet_id = tweet_json["id_str"]
            file.write(f'{tweet_id}\t{json.dumps(tweet_json, ensure_ascii=False)}\n')

def main():
    """メイン関数"""
    sys.stdin = codecs.getreader(sys.stdin.encoding)(sys.stdin.detach(), errors='ignore')
    
    random_tweets = extract_random_tweets(sys.stdin)
    print(f"Extracted {len(random_tweets)} random tweets")
    
    output_file_path = sys.argv[1]
    save_tweets(random_tweets, output_file_path)

if __name__ == '__main__':
    main()