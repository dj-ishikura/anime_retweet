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

def print_info(tweet, hashtag_set):
    try:
        if "retweeted_status" in tweet:
            id = tweet["retweeted_status"]["id_str"]
        else:
            id = tweet["id_str"]
        if "entities" in tweet:
            hashtag_list = [hashtag["text"] for hashtag in tweet["entities"]["hashtags"]]
            common_hashtags = hashtag_set & set(hashtag_list)
            if common_hashtags:
                for h in common_hashtags:
                    print(id + '\t' + h + '\t' + json.dumps(tweet, ensure_ascii=False))
    
    except KeyError as e:
        # logger.debug('KeyError: %s', e)
        pass
    except TypeError as t:
        # logger.debug('TypeError: %s', t)
        pass
    

def get_keywords_from_txt():
    # テキストファイルからキーワードリストを読み込む
    with open("./anime_hashtag_list_2022.tsv", 'r') as f:
        lines = [line.strip().split('\t')[1] for line in f.readlines()]
    return set(lines)
    
if __name__ == '__main__':

    import codecs
    sys.stdin = codecs.getreader(sys.stdin.encoding)(sys.stdin.detach(), errors='ignore')

    hashtag_set = get_keywords_from_txt()

    for tweet in parse_tweet(sys.stdin):
        print_info(tweet, hashtag_set)
