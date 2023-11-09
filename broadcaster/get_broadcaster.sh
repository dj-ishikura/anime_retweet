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

input_dir="../html_files/anime_wiki_page"
output_dir="data"
mkdir -p $output_dir

for file in $input_dir/*.html; do
    id=$(basename $file .html)

    wait_for_jobs

    size=$(du -ab $file | awk '{print $1}')
    mem=$(get_memory_allocation $size)

    output_file="${output_dir}/${id}.csv"
    if [ ! -f "$output_file" ]; then
        echo $output_file
        J=$id M=$mem qcmd python get_broadcaster.py $file $output_file
        sleep 1
    fi
done
