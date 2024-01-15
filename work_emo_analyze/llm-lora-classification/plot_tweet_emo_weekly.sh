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

tweet_object_dir="/work/n213304/learn/anime_retweet_2/anime_retweet_concat"
tweet_emo_dir="./prediction_"
output_dir="tweet_emo_weekly"
mkdir -p $output_dir

for tweet_emo_file in $tweet_emo_dir/*.json; do
    id=$(basename $tweet_emo_file .json)
    # id="2022-10-582"
    tweet_object_file="${tweet_object_dir}/${id}.csv"


    wait_for_jobs

    output_csv="${output_dir}/${id}.csv"
    output_png="${output_dir}/${id}.png"

    if [ ! -f "$output_csv" ] || [ ! -f "$output_png" ]; then
        echo $output_csv
        mem=$(get_memory_allocation $tweet_object_file)
        # J=$id M=$mem qcmd python src/plot_tweet_emo_weekly.py $tweet_object_file $tweet_emo_file $output_csv $output_png $id
        python src/plot_tweet_emo_weekly.py $tweet_object_file $tweet_emo_file $output_csv $output_png $id
        sleep 1
    fi
done
