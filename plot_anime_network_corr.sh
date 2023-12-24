#!/bin/bash
source wait_for_jobs.sh
source get_memory_allocation.sh

output_dir="weekly_anime_network_corr"  # 出力ディレクトリのパスを設定するのだ
input_dir="weekly_anime_network"

mkdir -p $output_dir

for sub_dir in $input_dir/*; do
    # サブディレクトリ内のCSVファイルをループするのだ
    for file in "$sub_dir"/*.csv; do
        if [ -f "$file" ]; then
            base_name=$(basename $file .csv)
            output_file="${output_dir}/${base_name}.csv"
            echo "Processing file: $file"
            
            # ここでCSVファイルを処理するコードを書くことができるのだ
            # 例: python your_script.py "$file"
            # 既にPNGとCSVファイルが存在する場合はスキップするのだ
            if [ -f $output_file ]; then
                echo "Skipping $base_name because output files already exist."
                continue
            fi

            M=1 qcmd python3 plot_anime_network_corr.py $file $output_file
            # python3 plot_anime_network_corr.py $file $output_file
            sleep 1
            wait_for_jobs 200
        fi
    done
done