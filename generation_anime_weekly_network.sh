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

input_dir="weekly_tweet_users_list"
output_dir="weekly_anime_network"
mkdir -p $output_dir
chmod +x $output_dir

for jsonl_path in $input_dir/*.jsonl; do
    id=$(basename $jsonl_path .jsonl)
    output_subdir="${output_dir}/${id}"
    mkdir -p $output_subdir
    output_csv="${output_subdir}/${id}.csv"
    if [ ! -f "$output_csv" ]; then
        echo $jsonl_path
        wait_for_jobs
        J=$id M=32 qcmd python generation_anime_weekly_network.py $jsonl_path $output_subdir
        # python generation_anime_weekly_network.py $jsonl_path $output_subdir
        sleep 1
    fi
done
