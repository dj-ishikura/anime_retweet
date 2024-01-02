#!/bin/bash

# gather_retweet_daily.sh
# 特定の日のツイートを取得する

dir=$1
date=$2
output_dir=$3

# ディレクトリ内の各ファイルをループします
for file in "$dir"/json_$date*.txt.gz; do
    # ファイルであることを確認します
    if [ -f "$file" ]; then
        FILE_NAME="$(basename "$file" .txt.gz)"
        # 出力ファイルを指定します
        outfile=$output_dir/$FILE_NAME.tsv
        echo $outfile
        if [ ! -e $outfile ]; then # 出力ファイルが存在しない場合
            
            zcat $file | cut -f 2 | python3 get_tweet_randam.py $outfile
        fi
    fi
done
