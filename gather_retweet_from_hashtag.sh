#!/bin/bash

# 特定のディレクトリを指定します
RT_TWEET_PATH='/work/my016/mirror/twitter-rt/ja'

# 出力先のディレクトリを作成
output_dir="retweet_data_転スラ"
mkdir -p $output_dir

# 検索するディレクトリの範囲を決定します
start_date="2021-01-04"
end_date="2021-09-29"

# 開始日から終了日までの各日に対してループします
current_date=$start_date
while [[ "$current_date" < "$end_date" ]]; do
    dir_date=$(echo "$current_date" | cut -d '-' -f 1,2)
    dir="$RT_TWEET_PATH/$dir_date"
    if [[ -d "$dir" ]]; then
        echo "$dir meets the date range"

        # ジョブの数を取得します
        job_count=$(qstat -u $USER | wc -l)

        # ジョブの数が100以上の場合、ジョブの完了を待ちます
        while [[ $job_count -ge 100 ]]; do
            sleep 10
            job_count=$(qstat -u $USER | wc -l)
        done
            
        W=48 J=$current_date qcmd bash gather_retweet_from_hashtag_daily.sh $dir $current_date $output_dir
        sleep 1
    fi
    # 次の日に進みます
    current_date=$(date -I -d "$current_date + 1 day")
done
