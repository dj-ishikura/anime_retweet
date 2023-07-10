#!/bin/bash

function wait_for_jobs() {
    # ジョブの数を取得します
    job_count=$(qstat -u $USER | wc -l)

    # ジョブの数が100以上の場合、ジョブの完了を待ちます
    while [[ $job_count -ge 100 ]]; do
        sleep 10
        job_count=$(qstat -u $USER | wc -l)
    done
}

# 入力ディレクトリを指定します
INPUT_DIR=./anime_retweet_2022

# 出力ディレクトリを指定します
OUTPUT_DIR=./anime_retweet_concat_2022

# idとハッシュタグの対応を格納したtsvファイルを指定します
ID_HASHTAG_FILE=./anime_hashtag_list_2022.tsv

# 出力ディレクトリを作成します（既に存在する場合は何もしません）
mkdir -p $OUTPUT_DIR

# idとハッシュタグの対応を格納した連想配列を作成します
declare -A ID_HASHTAG_MAP
while IFS=$'\t' read -r id hashtag; do
    # 出力ファイル名を設定します
    output_file="$OUTPUT_DIR/$id.csv"
    wait_for_jobs
    # サブディレクトリ内の全てのファイルを結合し、1列目のidでソートして出力ファイルに書き込みます
    subdir="$INPUT_DIR/$hashtag"
    qcmd "cat $subdir/* | sort -t ',' -k1,1 > $output_file"
    sleep 1
done < $ID_HASHTAG_FILE
