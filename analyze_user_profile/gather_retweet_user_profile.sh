#!/bin/bash

function get_memory_allocation() {
    local file=$1
    
    local size=$(du -ab $file | awk '{print $1}')
    # バイトをギガバイトに変換
    local mem=$(echo "scale=2; $size / 1024 / 1024 / 1024" | bc)
    # メモリを2倍にします
    mem=$(echo "scale=2; $mem * 2" | bc)
    # もしメモリが1GB未満なら1GBに、192GBより大ければ192GBに制限します
    if (( $(echo "$mem < 1" | bc -l) )); then
        mem=2
    elif (( $(echo "$mem > 192" | bc -l) )); then
        mem=192
    fi
    # 整数値で返す
    printf "%.0f\n" $mem
}

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
output_dir="retweet_user_profile"
mkdir -p $output_dir

for input_file in $input_dir/*.csv; do
    id=$(basename $input_file .csv)

    wait_for_jobs

    output_jsonl="${output_dir}/${id}.jsonl"
    if [ ! -f "$output_jsonl" ]; then
        echo $output_jsonl
        mem=$(get_memory_allocation $input_file)
        J=$id M=$mem qcmd python ./src/get_retweet_user_profile.py $input_file $output_jsonl $id
        sleep 1
    fi
done
