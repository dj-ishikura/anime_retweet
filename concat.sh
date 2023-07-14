#!/bin/bash

# ディレクトリを指定します
dir="./retweet_data_suisei"

# ディレクトリ内のすべてのサブディレクトリを探索します
for subdir in $(find $dir -type d); do
    # サブディレクトリ内のすべての.tsvファイルを結合します
    cat $subdir/*.tsv >> temp.tsv
done
# 結合した.tsvファイルを.csvに変換します
tr '\t' ',' < temp.tsv > temp.csv
sort -t ',' -k1,1 temp.csv > ./anime_retweet_concat_2022/2022-10-555.csv
rm temp.tsv
rm temp.csv