import json
import random
import csv

def sample_random_data(input_file_path, output_file_path, sample_size):
    """
    JSONLファイルからランダムにデータを取得して新しいCSVファイルに保存する関数。

    :param input_file_path: 入力JSONLファイルのパス
    :param output_file_path: 出力CSVファイルのパス
    :param sample_size: 取得するデータの数
    """
    # JSONLファイルを読み込む
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # ランダムにデータを取得
    random_sample = random.sample(lines, sample_size)

    # サンプルデータをCSVファイルとして保存
    with open(output_file_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['tweet_id', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for line in random_sample:
            tweet = json.loads(line)
            writer.writerow(tweet)

    print(f"ランダムサンプルデータが {output_file_path} に保存されました。")

if __name__ == "__main__":
    import argparse

    # コマンドライン引数を解析
    parser = argparse.ArgumentParser(description='JSONLファイルからランダムにデータを取得して新しいCSVファイルに保存する')
    parser.add_argument('input_file_path', type=str, help='入力JSONLファイルのパス')
    parser.add_argument('output_file_path', type=str, help='出力CSVファイルのパス')
    parser.add_argument('sample_size', type=int, help='取得するデータの数')

    args = parser.parse_args()

    sample_random_data(args.input_file_path, args.output_file_path, args.sample_size)
