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

# 入力ファイルのディレクトリを指定
followers_directory="/work/my016/mirror/twitter-following-followers-batch/2023-07_anime/followers"
following_directory="/work/my016/mirror/twitter-following-followers-batch/2023-07_anime/following"

# 入力ディレクトリごとにループを回す
for dir in $followers_directory $following_directory; do
  previous_user_id=""
  output_file=""
  
  # ディレクトリに応じてファイル名のサフィックスを設定
  if [ "$dir" == "$followers_directory" ]; then
    suffix="_followers.txt"
  else
    suffix="_following.txt"
  fi

  # gzipファイルごとにループを回す
  for file in $dir/*.txt.gz; do
    wait_for_jobs
    # 別のシェルスクリプトを呼び出して、ファイルとサフィックスを引数として渡す
    M=1 qcmd ./get_follower_list_sub.sh "$file" "$suffix"
    sleep 1
  done
done

echo "All data has been processed."

