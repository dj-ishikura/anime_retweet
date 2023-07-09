#!/bin/bash

# ディレクトリを設定します
source_directory="./official_twitter_aka/official_twitter_aka_bag"
target_directory="./official_twitter_aka/official_twitter_aka"

# source_directory内の全てのTSVファイルをループします
for file in "$source_directory"/*.tsv; do
    # ファイルが1行だけであるかどうかを確認します
    if [ $(wc -l < "$file") -eq 1 ]; then
        # ファイルが1行だけである場合、それをtarget_directoryに移動します
        echo "$file"
        mv "$file" "$target_directory"
    fi
done
