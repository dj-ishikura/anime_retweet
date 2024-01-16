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

retweet_dir="/work/n213304/learn/anime_retweet_2/anime_retweet_concat"
tweet_text_dir="/work/n213304/learn/anime_retweet_2/extra_anime_tweet_text_kikan"
output_dir="tweet_url"
mkdir -p $output_dir

for tweet_file in $tweet_text_dir/*.jsonl; do
    id=$(basename $tweet_file .jsonl)
    # id="2022-10-582"
    retweet_file="${retweet_dir}/${id}.csv"

    wait_for_jobs

    output_jsonl="${output_dir}/${id}.jsonl"

    if [ ! -f "$output_jsonl" ]; then
        echo $output_jsonl
        J=$id qcmd python src/get_url_from_retweet_concat.py $retweet_file $tweet_file $output_jsonl
        # python src/plot_tweet_emo_weekly.py $retweet_file $tweet_file $output_csv $output_png $id
        sleep 1
    fi
done
