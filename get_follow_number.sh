#!/bin/bash

mkdir -p follow_number

function wait_for_jobs() {
    local max_jobs=$1
    # ジョブの数を取得するのだ
    local job_count=$(qstat -u $USER | wc -l)

    # ジョブの数が指定された最大数以上の場合、ジョブの完了を待つのだ
    while [[ $job_count -ge $max_jobs ]]; do
        sleep 10
        job_count=$(qstat -u $USER | wc -l)
    done
}

user_ids=()
count=0

while IFS= read -r user_id; do
    # 出力ファイルが存在する場合はスキップするのだ
    if [[ -f "follow_number/${user_id}.jsonl" ]]; then
        continue
    fi

    user_ids+=("$user_id")
    count=$((count + 1))

    if (( count % 1000 == 0 )); then
        wait_for_jobs 200
        M=2 qcmd python3 get_follow_number.py "${user_ids[@]}"
        user_ids=()
        count=0
    fi
done < tweet_user_list.txt

# 最後のバッチを処理するのだ
if (( count > 0 )); then
    M=2 qcmd python3 get_follow_number.py "${user_ids[@]}"
fi

wait_for_jobs 1

# すべてのジョブが終了したら、結果をマージするのだ
: > tweet_user_follower_number.jsonl # ファイルを初期化するのだ
for file in follow_number/*.jsonl; do
    cat "$file" >> tweet_user_follower_number.jsonl
    echo "" >> tweet_user_follower_number.jsonl
done

# resultsディレクトリを削除するのだ
# rm -r follow_number

