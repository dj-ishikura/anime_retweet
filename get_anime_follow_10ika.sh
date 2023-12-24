#!/bin/bash
source wait_for_jobs.sh
source get_memory_allocation.sh

output_dir="anime_follow_10ika_2"  # 出力ディレクトリのパスを設定するのだ
follower_data_path="tweet_user_follower_number.jsonl"  # フォロワー数/フォロー数データのパスを設定するのだ

mkdir -p $output_dir

for jsonl_file in weekly_tweet_users_list/*.jsonl; do
    base_name=$(basename $jsonl_file .jsonl)
    output_file="${output_dir}/${base_name}.jsonl"
    
    # 既にPNGとCSVファイルが存在する場合はスキップするのだ
    if [[ -f $output_file ]]; then
        echo "Skipping $output_file because output files already exist."
        continue
    fi

    mem=$(get_memory_allocation $jsonl_file)
    M=$mem qcmd python3 get_anime_follow_10ika.py $jsonl_file $output_file
    # python3 get_anime_follow_10ika.py $jsonl_file $output_file
    sleep 1
    wait_for_jobs
done