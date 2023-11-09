#!/bin/bash
# find ./retweet_data_うまよん -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 -u > ./anime_tweet_concat/2020-07-113.csv
# find ./tweet_user_list -name "*.txt" -exec cat {} \; | sort -u > tweet_user_list.txt
# find ./retweet_data_アニメウマ娘 -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 > ./anime_retweet_concat/2021-01-191.csv
# find ./retweet_data_ガル学 -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 -u > ./anime_tweet_concat/2022-01-430.csv
# find ./retweet_data_ガル学 -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 -u > ./anime_tweet_concat/2020-04-76.csv
# find ./retweet_data_ガル学 -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 > ./anime_retweet_concat/2020-04-76.csv
# find ./retweet_data_ガル学 -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 > ./anime_retweet_concat/2022-01-430.csv
# find ./retweet_user_list -name "*.txt" -exec cat {} \; | sort -u > retweet_user_list.txt
# find ./retweet_data_うまよん -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 > ./anime_retweet_concat/2020-07-113.csv
# find ./retweet_user_list -name "*.txt" -exec cat {} \; | sort -u > retweet_user_list.txt
find ./retweet_data_転スラ -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 -u > ./anime_tweet_concat/2021-01-192.csv
find ./retweet_data_転スラ -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 > ./anime_retweet_concat/2021-01-192.csv
find ./retweet_data_転スラ -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 -u > ./anime_tweet_concat/2021-04-275.csv
find ./retweet_data_転スラ -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 > ./anime_retweet_concat/2021-04-275.csv
find ./retweet_data_転スラ -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 -u > ./anime_tweet_concat/2021-07-327.csv
find ./retweet_data_転スラ -name "*.tsv" -exec cat {} \; | tr '\t' ',' | sort -t, -k1,1 > ./anime_retweet_concat/2021-07-327.csv