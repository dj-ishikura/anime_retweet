#!/bin/bash
source wait_for_jobs.sh
source get_memory_allocation.sh

output_dir="follow_user_distribution"  # 出力ディレクトリのパスを設定するのだ
follower_data_path="tweet_user_follower_number.jsonl"  # フォロワー数/フォロー数データのパスを設定するのだ

mkdir -p $output_dir

for jsonl_file in weekly_tweet_users_list/*.jsonl; do
    mem=get_memory_allocation $jsonl_file
    M=$mem qcmd python3 plot_mean_follow.py $jsonl_file $output_dir
    wait_for_jobs
    break
done