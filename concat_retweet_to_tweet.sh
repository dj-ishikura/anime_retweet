#!/bin/bash

function wait_for_jobs() {
    local josb=$1
    # ジョブの数を取得します
    job_count=$(qstat -u $USER | wc -l)

    # ジョブの数が100以上の場合、ジョブの完了を待ちます
    while [[ $job_count -ge $josb ]]; do
        sleep 10
        job_count=$(qstat -u $USER | wc -l)
    done
}

# ファイルを格納しているディレクトリ
INPUT_DIR=./anime_retweet/ウマ娘

# 一時的にソートされたファイルを格納するディレクトリ
TEMP_DIR=./temp_sorted_files
mkdir -p $TEMP_DIR

# 各ファイルをソートして一時ディレクトリに保存
for file in $INPUT_DIR/*.csv; do
    base_name=$(basename "$file")
    wait_for_jobs 100
    qcmd "sort -t, -k1,1 -u ${file} > ${TEMP_DIR}/${base_name}"
done
echo "kaku file sort done"

cat $TEMP_DIR/*.csv | sort -t, -k1,1 -u > ./anime_tweet_concat/2021-01-191.csv

# rm -r $TEMP_DIR
