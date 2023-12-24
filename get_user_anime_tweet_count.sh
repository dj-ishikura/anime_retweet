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

input_dir="./anime_tweet_concat"
output_dir="user_anime_tweet_count"
mkdir -p $output_dir

for file in $input_dir/*.csv; do
    id=$(basename $file .csv)
    file="${input_dir}/${id}.csv"

    wait_for_jobs

    output_jsonl="${output_dir}/${id}.jsonl"
    if [ ! -f "$output_jsonl" ]; then
        echo $output_jsonl
        mem=$(get_memory_allocation $file)
        J=$id M=$mem qcmd python get_user_anime_tweet_count.py $file $output_jsonl $id
        # python count_tweet_users.py $file $period $output_jsonl $output_png $id
        sleep 1
    fi
done
