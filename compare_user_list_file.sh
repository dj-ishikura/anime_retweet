#!/bin/bash

# 入力ファイル名を指定するのだ
fileA="tweet_user_list/2020-04-76.txt"
fileB="retweet_user_list/2020-04-76.txt"


# 両方のファイルをソートするのだ
sort $fileA > sortedA.txt
sort $fileB > sortedB.txt

# comm コマンドで差分を取得するのだ
# -1 オプションでファイルAだけにある行を出力する
# -2 オプションでファイルBだけにある行を出力する
# -3 オプションで両方のファイルにある行を出力する
comm -1 sortedA.txt sortedB.txt

# ソート済みの一時ファイルを削除するのだ
rm sortedA.txt sortedB.txt
