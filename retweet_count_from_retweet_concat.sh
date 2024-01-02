#!/bin/bash
source get_memory_allocation.sh
function wait_for_jobs() {
    # ジョブの数を取得します
    job_count=$(qstat -u $USER | wc -l)

    # ジョブの数が100以上の場合、ジョブの完了を待ちます
    while [[ $job_count -ge 100 ]]; do
        sleep 10
        job_count=$(qstat -u $USER | wc -l)
    done
}

input_dir="./anime_retweet_concat"
output_dir="retweet_count"
mkdir -p $output_dir

for file in $input_dir/*.csv; do
    id=$(basename $file .csv)
    file="${input_dir}/${id}.csv"
    wait_for_jobs

    output_csv="${output_dir}/${id}.csv"
    if [ ! -f "$output_csv" ]; then
        echo $output_csv
        mem=$(get_memory_allocation $file)
        J=$id M=$mem qcmd python retweet_count_from_retweet_concat.py $file $output_csv $id
        # python count_tweet_users.py $file $period $output_csv $output_png $id
        sleep 1
    fi
done
