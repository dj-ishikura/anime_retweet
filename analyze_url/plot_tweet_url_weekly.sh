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

tweet_url_dir="tweet_url"
output_dir="tweet_url_weekly"
mkdir -p $output_dir

for tweet_url_file in $tweet_url_dir/*.jsonl; do
    id=$(basename $tweet_url_file .jsonl)

    wait_for_jobs

    output_csv="${output_dir}/${id}.csv"
    output_png="${output_dir}/${id}.png"

    if [ ! -f "$output_csv" ] || [ ! -f "$output_png" ]; then
        echo $output_csv
        J=$id qcmd python src/plot_url_weekly.py $tweet_url_file $output_csv $output_png $id
        # python src/plot_url_weekly.py $tweet_url_file $output_csv $output_png $id
        sleep 1
    fi
done
