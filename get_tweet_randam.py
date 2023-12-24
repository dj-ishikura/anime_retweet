# -*- coding: utf-8 -*-
'''
ツイートjsonから, 必要なデータを抽出
'''

import sys
import json
import glob
import os
import gzip

from twitter import parse_tweet
from log import get_logger
logger = get_logger(__name__)
    
if __name__ == '__main__':

    import codecs
    sys.stdin = codecs.getreader(sys.stdin.encoding)(sys.stdin.detach(), errors='ignore')

    tweet_data = {}
    for tweet in parse_tweet(sys.stdin):
        try:
            import random
            # 0から2までのランダムな整数を生成
            random_number = random.randint(0, 99)

            if "retweeted_status" in tweet:
                tweet = tweet["retweeted_status"]
            if "id_str" in tweet and tweet["id_str"] not in tweet_data:
                tweet_id = tweet["id_str"]

                if len(tweet_data) > 50 and random_number == 0:
                    if "text" in tweet:
                        tweet_data[tweet_id] = tweet["text"]
                    elif "full_text" in tweet:
                        tweet_data[tweet_id] = tweet["full_text"]
            
        except KeyError as e:
            # logger.debug('KeyError: %s', e)
            pass
        except TypeError as t:
            # logger.debug('TypeError: %s', t)
            pass
