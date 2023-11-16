#!/bin/bash

function wait_for_jobs() {
    local jobs=$1
    # ジョブの数を取得します
    job_count=$(qstat -u $USER | wc -l)

    # ジョブの数が指定数以上の場合、ジョブの完了を待ちます
    while [[ $job_count -ge $jobs ]]; do
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

# 入力と出力ディレクトリを設定
INPUT_DIRECTORY="anime_retweet_concat"
OUTPUT_DIRECTORY="anime_tweet_concat"

# 出力ディレクトリが存在しない場合は作成
if [ ! -d $OUTPUT_DIRECTORY ]; then
    mkdir $OUTPUT_DIRECTORY
fi

# 入力ディレクトリ内のすべてのCSVファイルを処理
for file in $INPUT_DIRECTORY/*.csv; do
    # 出力ファイル名を設定（拡張子なし）
    filename=$(basename "$file")
    output_file="${OUTPUT_DIRECTORY}/${filename}"

    # 出力ファイルが既に存在する場合はスキップ
    if [ -f "$output_file" ]; then
        echo "スキップ: $output_file は既に存在します。"
        continue
    fi

    # ファイルサイズに基づいてメモリ割り当てを計算
    file_size=$(stat -c%s "$file")
    mem_alloc=$(get_memory_allocation $file_size)

    # ジョブの数を確認してからジョブを投げる
    wait_for_jobs 100
    M=${mem_alloc} qcmd "sort -u -t, -k1,1 ${file} > ${output_file}"
done
