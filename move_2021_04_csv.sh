#!/bin/bash

# ディレクトリを指定します
dir="count_tweet"
mv_dir="count_tweet_2021_04"

# 範囲内の日付を含むcsvファイルを探す
for file in "$dir"/*.csv; do
    if awk -F, 'NR > 1 && $1 >= "2021-04-14" && $1 <= "2021-04-20"' "$file" | grep -q .; then
        echo "Removing $file and associated png"
        # 対応するpngファイル名を取得
        png_file="${file%.*}.png"
        # csvとpngファイルを削除
        mv $file $mv_dir
        mv $png_file $mv_dir
    fi
done
