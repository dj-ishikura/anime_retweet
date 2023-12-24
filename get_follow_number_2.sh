#!/bin/bash

# すべてのジョブが終了したら、結果をマージするのだ
: > tweet_user_follower_number.jsonl # ファイルを初期化するのだ
for file in follow_number/*.jsonl; do
    cat "$file" >> tweet_user_follower_number.jsonl
    echo "" >> tweet_user_follower_number.jsonl
done

# resultsディレクトリを削除するのだ
# rm -r results
