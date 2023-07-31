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


# 入力ディレクトリを指定します
INPUT_DIR=./anime_retweet

# 出力ディレクトリを指定します
OUTPUT_DIR=./anime_retweet_concat

# idとハッシュタグの対応を格納したtsvファイルを指定します
ID_HASHTAG_FILE=./anime_hashtag_list_2020-2023.csv

# 出力ディレクトリを作成します（既に存在する場合は何もしません）
mkdir -p $OUTPUT_DIR

# idとハッシュタグの対応を格納した連想配列を作成します
declare -A ID_HASHTAG_MAP
while IFS=$',' read -r id hashtag; do
    # 出力ファイル名を設定します
    output_file="$OUTPUT_DIR/$id.csv"
    wait_for_jobs
    # サブディレクトリ内の全てのファイルを結合し、1列目のidでソートして出力ファイルに書き込みます
    subdir="$INPUT_DIR/$hashtag"
    # ディレクトリのサイズを取得し、必要なメモリを計算します
    size=$(du -sb $subdir | awk '{print $1}')
    mem=$(get_memory_allocation $size)
    # ファイルが存在しない場合、またはソートが成功した場合にのみファイルを出力します
    if [ ! -f $output_file ] && M=${mem} qcmd "cat $subdir/* | sort -t ',' -k1,1 > $output_file"; then
        echo "Sorting completed successfully for $id"
        sleep 1
    fi
    
done < $ID_HASHTAG_FILE
