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

input_dir="./anime_tweet_concat"
output_dir="tweet_user_list"
mkdir -p $output_dir

for file in $input_dir/*.csv; do
    id=$(basename $file .csv)
    # id="2022-10-555"
    file="${input_dir}/${id}.csv"
    wait_for_jobs

    output_csv="${output_dir}/${id}.txt"
    if [ ! -f "$output_csv" ]; then
        echo $output_csv
        J=$id M=24 qcmd python create_tweet_user_list.py $file $output_csv $id
        sleep 1
    fi
done
