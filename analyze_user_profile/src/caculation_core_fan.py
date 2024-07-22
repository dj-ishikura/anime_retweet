import os
import json
from collections import defaultdict

def get_user_anime_data(directory):
    user_anime_data = defaultdict(lambda: {'anime_set': set(), 'anime_in_profile': False})
    
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            anime_id = os.path.splitext(filename)[0]  # ファイル名からアニメIDを取得
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        user_id = data['user_id']
                        user_anime_data[user_id]['anime_set'].add(anime_id)
                        
                        if 'アニメ' in data.get('profile', ''):
                            user_anime_data[user_id]['anime_in_profile'] = True
                    except json.JSONDecodeError:
                        continue
    
    return user_anime_data

def calculate_percentages(user_anime_data):
    total_users = len(user_anime_data)
    anime_in_profile = sum(1 for data in user_anime_data.values() if data['anime_in_profile'])
    two_or_more_anime = sum(1 for data in user_anime_data.values() if len(data['anime_set']) >= 2)
    three_or_more_anime = sum(1 for data in user_anime_data.values() if len(data['anime_set']) >= 3)
    
    anime_or_two_plus = sum(1 for data in user_anime_data.values() if data['anime_in_profile'] or len(data['anime_set']) >= 2)
    anime_or_three_plus = sum(1 for data in user_anime_data.values() if data['anime_in_profile'] or len(data['anime_set']) >= 3)
    
    print(f"総ユーザー数: {total_users}")
    print(f"プロフィールに「アニメ」を含むユーザーの割合: {anime_in_profile / total_users:.2%}")
    print(f"2作品以上のアニメについてツイートしているユーザーの割合: {two_or_more_anime / total_users:.2%}")
    print(f"3作品以上のアニメについてツイートしているユーザーの割合: {three_or_more_anime / total_users:.2%}")
    print(f"プロフィールに「アニメ」を含む or 2作品以上のアニメについてツイートしているユーザーの割合: {anime_or_two_plus / total_users:.2%}")
    print(f"プロフィールに「アニメ」を含む or 3作品以上のアニメについてツイートしているユーザーの割合: {anime_or_three_plus / total_users:.2%}")

def main():
    directory = 'tweet_user_profile_all_works'  # ディレクトリパスを適切に設定してください
    user_anime_data = get_user_anime_data(directory)
    calculate_percentages(user_anime_data)

if __name__ == "__main__":
    main()