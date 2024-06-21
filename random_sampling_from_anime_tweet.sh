#!/bin/bash

OUTPUT_DIR="random_sanpling_from_anime_tweet"
mkdir -p $OUTPUT_DIR

SAMPLE_SIZE=100
target_file="2022-10-582"
python3 random_sampling_from_anime_tweet.py "./extra_anime_tweet_text_kikan/${target_file}.jsonl" "${OUTPUT_DIR}/${target_file}.csv" "$SAMPLE_SIZE"

target_file="2022-10-588"
python3 random_sampling_from_anime_tweet.py "./extra_anime_tweet_text_kikan/${target_file}.jsonl" "${OUTPUT_DIR}/${target_file}.csv" "$SAMPLE_SIZE"

target_file="2022-04-484"
python3 random_sampling_from_anime_tweet.py "./extra_anime_tweet_text_kikan/${target_file}.jsonl" "${OUTPUT_DIR}/${target_file}.csv" "$SAMPLE_SIZE"

target_file="2023-01-605"
python3 random_sampling_from_anime_tweet.py "./extra_anime_tweet_text_kikan/${target_file}.jsonl" "${OUTPUT_DIR}/${target_file}.csv" "$SAMPLE_SIZE"