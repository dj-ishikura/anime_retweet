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

def print_info(tweet, hashtag):
    try:
        tweet = tweet["data"]
        if "referenced_tweets" in tweet:
            id = tweet["referenced_tweets"]["id"]
        else:
            id = tweet["id"]
        if "entities" in tweet:
            hashtag_list = [h["tag"] for h in tweet["entities"]["hashtags"]]
            if hashtag in hashtag_list:
                print(id + '\t' + hashtag + '\t' + json.dumps(tweet, ensure_ascii=False))

    except KeyError as e:
        # logger.debug('KeyError: %s', e)
        pass
    except TypeError as t:
        # logger.debug('TypeError: %s', t)
        pass
    
    
if __name__ == '__main__':

    import codecs
    sys.stdin = codecs.getreader(sys.stdin.encoding)(sys.stdin.detach(), errors='ignore')

    hashtag = "水星の魔女"
    for tweet in parse_tweet(sys.stdin):
        print_info(tweet, hashtag)
