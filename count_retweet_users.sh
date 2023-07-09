#!/bin/bash

# Noと日付範囲を取得するためのCSVファイルを指定します
tsv_file=./anime_info/anime_info_complete_hashtag_edit.tsv 

input_dir="./anime_retweet_data_concat"
output_dir="count_retweet"
mkdir -p $output_dir

# CSVファイルの最初の行（ヘッダー行）をスキップします
IFS="$(echo -e '\t')"
tail -n +2 "$tsv_file" | while read LINE; do
    LINE=($LINE)
    # extraの日付をタイムスタンプに変換し、比較可能な形式にします
    title=${LINE[0]}
    before_start_date=${LINE[6]}
    after_end_date=${LINE[7]}
    No=${LINE[10]}
    echo $No
    echo $before_start_date
    echo $after_end_date
    input_file="${input_dir}/${No}.tsv"
    for period in 1 2 3 4; do
        # ジョブの数を取得します
        job_count=$(qstat -u $USER | wc -l)

        # ジョブの数が100以上の場合、ジョブの完了を待ちます
        while [[ $job_count -ge 100 ]]; do
            sleep 10
            job_count=$(qstat -u $USER | wc -l)
        done
        output_csv="${output_dir}/${No}_${period}_week_retweet_counts.csv"
        output_png="${output_dir}/${No}_${period}_week_retweet_counts.png"
        if [ ! -f "$output_csv" ] || [ ! -f "$output_png" ]; then
            J=$No M=128 qcmd python count_retweet_users.py $input_file $period $output_csv $output_png $No
        # ここで何かのコマンドを実行します
        fi
        sleep 0.1
    done
done
