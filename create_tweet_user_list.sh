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

function get_memory_allocation() {
    local size=$1
    # バイトをギガバイトに変換
    local mem=$(echo "scale=2; $size / 1024 / 1024 / 1024" | bc)
    # メモリを2倍にします
    mem=$(echo "scale=2; $mem * 2" | bc)
    # もしメモリが1GB未満なら1GBに、192GBより大ければ192GBに制限します
    if (( $(echo "$mem < 1" | bc -l) )); then
        mem=1
    elif (( $(echo "$mem > 192" | bc -l) )); then
        mem=192
    fi
    # 整数値で返す
    printf "%.0f\n" $mem
}

input_dir="./anime_tweet_concat"
# input_dir="./anime_retweet_concat"
output_dir="tweet_user_list"
mkdir -p $output_dir

for file in $input_dir/*.csv; do
    id=$(basename $file .csv)
    id="2020-04-76"
    # id="2022-01-430"
    file="${input_dir}/${id}.csv"
    wait_for_jobs

    output_csv="${output_dir}/${id}.txt"
    if [ ! -f "$output_csv" ]; then
        echo $output_csv
        size=$(du -ab $file | awk '{print $1}')
        mem=$(get_memory_allocation $size)
        J=$id M=32 qcmd python create_tweet_user_list.py $file $output_csv $id
        sleep 1
    fi
    break
done
