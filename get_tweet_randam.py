# -*- coding: utf-8 -*-
'''
ツイートjsonから, 必要なデータを抽出
'''

import sys
import json
import glob
import os
import gzip
import random

from twitter import parse_tweet
from log import get_logger
logger = get_logger(__name__)
    
if __name__ == '__main__':

    import codecs
    sys.stdin = codecs.getreader(sys.stdin.encoding)(sys.stdin.detach(), errors='ignore')

    tweet_data = {}
    for tweet_json in parse_tweet(sys.stdin):
        try:
            
            # 0から2までのランダムな整数を生成
            random_number = random.randint(0, 2)
            print(random_number)

            if "retweeted_status" in tweet_json:
                tweet_json = tweet_json["retweeted_status"]
            if "id_str" in tweet_json and tweet_json["id_str"] not in tweet_data:
                tweet_id = tweet_json["id_str"]

                if len(tweet_data) < 50 and random_number == 0:
                    if "text" in tweet_json or "full_text":
                        tweet_data[tweet_id] = tweet_json
            
        except KeyError as e:
            # logger.debug('KeyError: %s', e)
            pass
        except TypeError as t:
            # logger.debug('TypeError: %s', t)
            pass

# tweet_dataをツイートIDでソート
print(len(tweet_data))
sorted_tweet_data = sorted(tweet_data.items(), key=lambda x: x[0])

# 保存するファイルのパス
output_file_path = sys.argv[1]

with open(output_file_path, 'w', encoding='utf-8') as file:
    for tweet_id, tweet_json in sorted_tweet_data:
        # ツイートIDとツイートJSONをタブ区切りの形式でファイルに書き込む
        file.write(f'{tweet_id}\t{json.dumps(tweet_json, ensure_ascii=False)}\n')
