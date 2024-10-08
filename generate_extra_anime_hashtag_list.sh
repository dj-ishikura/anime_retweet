#!/bin/bash

input_dir="/work/n213304/learn/anime_retweet_2/extra_anime_retweet_concat"
output_file="extra_anime_hashtag_list.csv"

# 入力ディレクトリが存在するか確認
if [ ! -d "$input_dir" ]; then
    echo "エラー: 指定されたディレクトリ '$input_dir' が存在しません。"
    exit 1
fi

# 出力ファイルのヘッダーを書き込む
echo "id,hashtag" > "$output_file"

# 指定されたディレクトリ内のCSVファイルを処理する
for file in "$input_dir"/*.csv; do
    if [ -f "$file" ]; then
        # ファイル名から拡張子を除いたものをIDとして使用
        id=$(basename "$file" .csv)
        
        # 2列目（ハッシュタグ）の最初の値を取得（ヘッダーを除く）
        hashtag=$(awk -F',' 'NR==2 {print $2; exit}' "$file")
        
        # 結果を出力ファイルに追記
        echo "$id,$hashtag" >> "$output_file"
    fi
done

echo "処理が完了しました。結果は $output_file に保存されています。"