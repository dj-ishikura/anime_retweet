#!/bin/bash

file="$1"
suffix="$2"
output_directory="./follow_user"
mkdir -p $output_directory

previous_user_id=""
output_file=""

# ファイルの中身を1行ずつ処理
zcat $file | while IFS=';' read -r user_id following_user_id; do
  # ユーザIDが前の行と同じであれば、フォローされているユーザIDを出力ファイルに追加
  if [ "$user_id" == "$previous_user_id" ]; then
    echo $following_user_id >> $output_file
  else
    # 新しいユーザIDの場合、新しい出力ファイルを開始
    output_file="$output_directory/$user_id$suffix"
    echo $following_user_id > $output_file
  fi
  previous_user_id="$user_id"
done
