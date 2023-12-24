#!/bin/bash
source wait_for_jobs.sh
source get_memory_allocation.sh

output_dir="follow_user_distribution"  # 出力ディレクトリのパスを設定するのだ
follower_data_path="tweet_user_follower_number.jsonl"  # フォロワー数/フォロー数データのパスを設定するのだ

mkdir -p $output_dir

for jsonl_file in weekly_tweet_users_list/*.jsonl; do
    base_name=$(basename $jsonl_file .jsonl)
    png_file="${output_dir}/${base_name}.png"
    csv_file="${output_dir}/${base_name}.csv"
    
    # 既にPNGとCSVファイルが存在する場合はスキップするのだ
    if [[ -f $png_file && -f $csv_file ]]; then
        echo "Skipping $base_name because output files already exist."
        continue
    fi

    mem=$(get_memory_allocation $jsonl_file)
    M=$mem qcmd python3 plot_anime_follow_user_distribution.py $jsonl_file $output_dir
    # python3 plot_anime_follow_user_distribution.py $jsonl_file $output_dir
    sleep 1
    wait_for_jobs
done
