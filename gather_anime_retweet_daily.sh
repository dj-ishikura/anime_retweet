#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2

# サブディレクトリ内のすべてのtsvファイルを読み取ります
date=$(basename $INPUT_DIR)

for file in $INPUT_DIR/*.tsv; do
    # Initialize previous hashtag
    prev_hashtag_name=""
    output_file=""

    while IFS=$'\t' read -r tweet_id hashtag_name tweet_json; do
        # ハッシュタグ名の行を追加します
        # ハッシュタグ名をサブディレクトリ名として使用します
        if [[ "$hashtag_name" != "$prev_hashtag_name" ]]; then
            subdir="$OUTPUT_DIR/$hashtag_name"
            mkdir -p $subdir
            output_file="$subdir/$date.tsv"
            prev_hashtag_name=$hashtag_name
        fi
        echo -E "$tweet_id"'\t'"$hashtag_name"'\t'"$tweet_json" >> $output_file

    done < "$file"
done
