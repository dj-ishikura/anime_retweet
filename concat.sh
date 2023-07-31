#!/bin/bash
# find ./retweet_data_うまよん -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 -u > ./anime_tweet_concat/2020-07-113.csv
find ./tweet_user_list -name "*.txt" -exec cat {} \; | sort -u > tweet_user_list.txt
