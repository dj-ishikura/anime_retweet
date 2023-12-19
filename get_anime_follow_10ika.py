import os
import json
import pandas as pd
import sys

def collect_data(jsonl_file, follower_data, output_file_path):
    user_data_df = pd.read_json(jsonl_file, lines=True)
    user_ids_set = set()

    for user_ids in user_data_df['user_ids']:
        user_ids_set.update(user_ids)

    merged_data = follower_data[follower_data['user_id'].isin(user_ids_set)]

    id = jsonl_file.split('/')[-1].replace('.jsonl', '')
    total_users = len(merged_data)
    
    data_dict = {"id": id, "total_users": total_users}

    for i in range(8):  # フォロワー数の範囲は0から7
        followers_count_range = len(merged_data[(merged_data['followers_count'] > 10**(i-1)) & (merged_data['followers_count'] <= 10**i)])
        data_dict[f"followers_{10**i}_or_less"] = followers_count_range

    for i in range(7):  # フォロー数の範囲は0から6
        following_count_range = len(merged_data[(merged_data['following_count'] > 10**(i-1)) & (merged_data['following_count'] <= 10**i)])
        data_dict[f"following_{10**i}_or_less"] = following_count_range

    # 辞書をJSONLファイルに保存するのだ
    with open(output_file_path, 'w') as file:
        json.dump(data_dict, file)

if __name__ == "__main__":
    follower_data_path = "tweet_user_follower_number.jsonl"
    follower_data = pd.read_json(follower_data_path, lines=True)
    collect_data(sys.argv[1], follower_data, sys.argv[2])
