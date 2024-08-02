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

input_dir="/work/n213304/learn/anime_retweet_2/extra_anime_retweet_concat"
output_dir="count_tweet"
mkdir -p $output_dir

for file in $input_dir/*.csv; do
    id=$(basename $file .csv)
    # id="2021-01-192"
    file="${input_dir}/${id}.csv"
    wait_for_jobs

    output_csv="${output_dir}/${id}.csv"
    output_png="${output_dir}/${id}.png"
    if [ ! -f "$output_csv" ] || [ ! -f "$output_png" ]; then
        echo $output_csv
        mem=$(get_memory_allocation $file)
        J=$id M=$mem qcmd python src/count_tweet.py $file $output_csv $output_png $id
        sleep 1
    fi
done
