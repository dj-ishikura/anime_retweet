# -*- coding: utf-8 -*-
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob
import matplotlib.pyplot as plt
import pytz
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

def extract_urls_from_tweet(tweet):
    """
    ツイートデータから拡張URLを抽出する関数。
    """
    urls = []
    if "entities" in tweet and "urls" in tweet["entities"]:
        urls = [url["expanded_url"] for url in tweet["entities"]["urls"] if "expanded_url" in url]
    return urls

def extract_media_info(tweet):
    media_info = []
    
    if "extended_entities" in tweet and "media" in tweet["extended_entities"]:
        for media in tweet["extended_entities"]["media"]:
            media_details = {
                "media_url": media.get("media_url_https", ""),
                "type": media.get("type", ""),
                "expanded_url": media.get("expanded_url", ""),
            }
            media_info.append(media_details)
    
    return media_info

def extract_text(tweet):
    if "full_text" in tweet:
        text = tweet["full_text"]
    elif "text" in tweet:
        text = tweet["text"]
    return text

def read_jsonl(input_path, tweet_ids):
    tweet_data = {}
    for tweet_id in tweet_ids:
        tweet_data[tweet_id] = {"tweet_id": tweet_id, "text": "", "created_at": "", "media": [], "urls": []}

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # タブで区切られた行を分割し、2列目（0から数えて1）をJSONとして解析
            line = line.replace(",", "\t", 2)
            json_string = line.split('\t')[2]
            tweet = json.loads(json_string.rstrip('\n|\r'))
            # ツイートのみを取得
            if "retweeted_status" in tweet:
                tweet = tweet["retweeted_status"]
                tweet_id = tweet["id_str"]
                
                if tweet_id in tweet_data: 
                    if tweet_data[tweet_id]["text"] == "":
                        tweet_data[tweet_id]["text"] = extract_text(tweet)

                    if tweet_data[tweet_id]["created_at"] == "":
                        tweet_data[tweet_id]["created_at"] = tweet.get("created_at", "")

                    if not tweet_data[tweet_id]["media"]:
                        tweet_data[tweet_id]["media"] = extract_media_info(tweet)

                    if not tweet_data[tweet_id]["urls"]:
                        tweet_data[tweet_id]["urls"] = extract_urls_from_tweet(tweet)
                        
    return tweet_data

def get_tweet_ids(tweet_path):
    # tweet_id 列を文字列型として読み込む
    df = pd.read_json(tweet_path, lines=True, dtype={'tweet_id': str})
    return df['tweet_id'].tolist()

def write_jsonl(output_path, tweet_data):
    with open(output_path, 'w', encoding='utf-8') as file:
        for tweet_id, tweet_info in tweet_data.items():
            # 辞書をJSON文字列に変換
            json_line = json.dumps(tweet_info, ensure_ascii=False)
            file.write(json_line + '\n')

if __name__ == "__main__":
    import sys

    retweet_path = sys.argv[1]
    tweet_path = sys.argv[2]
    output_jsonl = sys.argv[3]
    tweet_ids = get_tweet_ids(tweet_path)

    tweet_data = read_jsonl(retweet_path, tweet_ids)
    
    write_jsonl(output_jsonl, tweet_data)
    
