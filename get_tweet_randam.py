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
        
        if "retweeted_status" in tweet:
            tweet = tweet["retweeted_status"]
        id = tweet["id_str"]

        if "entities" in tweet:
            hashtag_list = [h["text"] for h in tweet["entities"]["hashtags"]]
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

    id_set = {}
    for tweet in parse_tweet(sys.stdin):
        try:
        
        if "retweeted_status" in tweet:
            tweet = tweet["retweeted_status"]
        id = tweet["id_str"]
        

        if "entities" in tweet:
            hashtag_list = [h["text"] for h in tweet["entities"]["hashtags"]]
            if hashtag in hashtag_list:
                    print(id + '\t' + hashtag + '\t' + json.dumps(tweet, ensure_ascii=False))

        id_set.add(id)
    
        except KeyError as e:
            # logger.debug('KeyError: %s', e)
            pass
        except TypeError as t:
            # logger.debug('TypeError: %s', t)
            pass
