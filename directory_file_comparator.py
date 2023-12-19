import os

def list_unique_files(dir1, dir2):
    # ディレクトリ内のファイルとサブディレクトリの名前を取得するのだ
    files_in_dir1 = set(os.listdir(dir1))
    files_in_dir2 = set(os.listdir(dir2))
    
    # 2つのディレクトリを比較し、片方にしか存在しないファイル名を見つけるのだ
    unique_in_dir1 = files_in_dir1 - files_in_dir2
    unique_in_dir2 = files_in_dir2 - files_in_dir1
    
    return unique_in_dir1, unique_in_dir2

# ディレクトリパスを指定するのだ
dir1 = "anime_retweet_concat"
dir2 = "anime_tweet_concat"

# 関数を呼び出して結果を取得するのだ
unique_in_dir1, unique_in_dir2 = list_unique_files(dir1, dir2)

# 結果を出力するのだ
print(f"Unique files in {dir1}: {unique_in_dir1}")
print(f"Unique files in {dir2}: {unique_in_dir2}")
