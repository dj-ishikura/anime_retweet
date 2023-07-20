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

input_dir="./anime_retweet_concat_2022"
output_dir="count_tweet_2022_2"
mkdir -p $output_dir

for file in $input_dir/*.csv; do
    id=$(basename $file .csv)
    # id="2022-10-555"
    file="${input_dir}/${id}.csv"
    for period in 1; do
        wait_for_jobs
    
        output_csv="${output_dir}/${id}_${period}_week_tweet_counts.csv"
        output_png="${output_dir}/${id}_${period}_week_tweet_counts.png"
        if [ ! -f "$output_csv" ] || [ ! -f "$output_png" ]; then
            echo $output_csv
            # J=$id M=192 qcmd python count_tweet_users.py $file $period $output_csv $output_png $id
            python count_tweet_users.py $file $period $output_csv $output_png $id
            sleep 1
        fi
    done
done
