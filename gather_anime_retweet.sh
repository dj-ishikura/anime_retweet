#!/bin/bash

# 入力ディレクトリを指定します
input_dir="./retweet_data_2022"

# 出力ディレクトリを指定します
output_dir="./anime_retweet_2022"
mkdir -p $output_dir

# 入力ディレクトリ内のすべてのサブディレクトリを読み取ります
for subdir in $input_dir/*; do

    # ジョブの数を取得します
    job_count=$(qstat -u $USER | wc -l)

    # ジョブの数が100以上の場合、ジョブの完了を待ちます
    while [[ $job_count -ge 100 ]]; do
        sleep 10
        job_count=$(qstat -u $USER | wc -l)
    done

    # サブディレクトリごとの処理を行うスクリプトを呼び出します
    qcmd bash gather_anime_retweet_daily.sh $subdir $output_dir
    sleep 1
done

