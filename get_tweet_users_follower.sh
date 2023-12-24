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
    local file=$1
    
    local size=$(du -ab $file | awk '{print $1}')
    # バイトをギガバイトに変換
    local mem=$(echo "scale=2; $size / 1024 / 1024 / 1024" | bc)
    # メモリを2倍にします
    mem=$(echo "scale=2; $mem * 2" | bc)
    # もしメモリが1GB未満なら1GBに、192GBより大ければ192GBに制限します
    if (( $(echo "$mem < 1" | bc -l) )); then
        mem=8
    elif (( $(echo "$mem > 192" | bc -l) )); then
        mem=192
    fi
    # 整数値で返す
    printf "%.0f\n" $mem
}

# weekly_tweet_users_listディレクトリ内の全てのファイルに対して処理を行う
output_dir="weekly_tweet_users_follower"
mkdir -p $output_dir

for file in weekly_tweet_users_list/*.jsonl; do
  wait_for_jobs
  anime_id=$(basename "$file" .jsonl) # ファイル名からアニメIDを取得
  output_file="$output_dir/${anime_id}.jsonl" # 出力ファイルのパスを設定

  mem=$(get_memory_allocation $file)

  # Pythonプログラムを呼び出して処理を行う
  if [ ! -f "$output_file" ]; then
    echo $output_file
    J=$id M=$mem qcmd python get_tweet_users_follower.py "$file" "$output_file"
    sleep 1
  fi
done
