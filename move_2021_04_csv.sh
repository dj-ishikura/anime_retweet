#!/bin/bash

# ディレクトリを配列で定義するのだ
dirs=("count_tweet" "count_retweet")
count=0

# 各ディレクトリに対して操作を行うのだ
for dir in "${dirs[@]}"; do
    # 移動先のディレクトリ名を動的に生成するのだ
    mv_dir="${dir}_2021_04"

    # もし移動先のディレクトリが存在しなければ、作成する
    if [ ! -d "$mv_dir" ]; then
        mkdir "$mv_dir"
    fi

    # 範囲内の日付を含むcsvファイルを探す
    for file in "$dir"/*.csv; do
        if awk -F, 'NR > 1 && $1 >= "2021-04-14" && $1 <= "2021-04-20"' "$file" | grep -q .; then
            echo "Moving $file and associated png"
            # 対応するpngファイル名を取得
            png_file="${file%.*}.png"
            # csvとpngファイルを移動
            mv "$file" "$mv_dir"
            mv "$png_file" "$mv_dir"
            # カウントアップ
            count=$((count+1))
        fi
    done
done

echo "Total files moved: $count"
