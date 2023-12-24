#!/bin/bash

input_dir="./track_tweet"
output_file="./result/anime_retention_rate.csv"
echo "id,retention_rate" > $output_file

output_dir="retention_rate"
mkdir -p $output_dir

for file in $input_dir/*.csv; do
    id=$(echo "$file" | awk -F'/' '{print $3}' | awk -F'_' '{print $1}')
    echo $id
    python retention_tweets_user.py $file "${output_dir}/${id}.csv"  >> $output_file
done
