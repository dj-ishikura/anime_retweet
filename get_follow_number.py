import os
import json
import sys

dir_path = "follow_user"

def get_user_data(user_id):
    user_data = {}
    user_data["user_id"] = user_id
    try:
        # フォロワーの数を取得するのだ
        with open(os.path.join(dir_path, f'{user_id}_followers.txt'), 'r') as file:
            followers_count = len(file.readlines())
        user_data["followers_count"] = followers_count

        # フォローしている人数を取得するのだ
        with open(os.path.join(dir_path, f'{user_id}_following.txt'), 'r') as file:
            following_count = len(file.readlines())
        user_data["following_count"] = following_count

    except FileNotFoundError:
        print(f"Files for user ID {user_id} not found.")
        return None

    return user_data

if __name__ == "__main__":
    user_ids = sys.argv[1:]
    for user_id in user_ids:
        user_data = get_user_data(user_id)
        if user_data:
            # 結果をJSONファイルに保存するのだ
            with open(f'follow_number/{user_id}.jsonl', 'w') as json_file:
                json.dump(user_data, json_file)
                print(f'output file : {json_file}')
