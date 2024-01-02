#!/bin/bash

# ディレクトリとファイルパス
retweet_dir="anime_retweet_concat"
tweet_dir="anime_tweet_concat"
remove_file="anime_data_loss_data.csv"

# 除外するIDのリストを生成
remove_ids=$(cut -d ',' -f 1 $remove_file | tail -n +2)
num_remove_ids=$(echo "$remove_ids" | wc -l)
echo "除外するIDの数: $num_remove_ids"

# 集計用の変数
total_retweets=0
total_tweets=0
num_files=0

# ツイートデータとリツイートデータのファイル名が同じであると仮定
for file in "$retweet_dir"/*; do
    echo $file
    id=$(basename "$file" .csv)
    if echo "$remove_ids" | grep -q "$id"; then
        continue
    fi
    retweet_count=$(wc -l < "$file")
    tweet_file="${tweet_dir}/${id}.csv"
    if [ -f "$tweet_file" ]; then
        tweet_count=$(wc -l < "$tweet_file")
    else
        tweet_count=0
    fi
    total_retweets=$((total_retweets + retweet_count))
    total_tweets=$((total_tweets + tweet_count))
    num_files=$((num_files + 1))
done

# 平均を計算
average_retweets=0
average_tweets=0
if [ $num_files -gt 0 ]; then
    average_retweets=$((total_retweets / num_files))
    average_tweets=$((total_tweets / num_files))
fi

# 結果の出力

echo "合計リツイート数: $total_retweets"
echo "合計ツイート数: $total_tweets"
echo "平均リツイート数: $average_retweets"
echo "平均ツイート数: $average_tweets"
