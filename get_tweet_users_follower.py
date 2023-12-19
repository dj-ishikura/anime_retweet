import os
import json

def get_followers(user_id):
    followers_file = os.path.join('follow_user', f'{user_id}_followers.txt')
    if os.path.exists(followers_file):
        with open(followers_file, 'r') as file:
            return file.read().strip().split('\n')
    return []

def process_weekly_tweet_users(weekly_tweet_users_file, output_file):
    with open(weekly_tweet_users_file, 'r') as file:
        for line in file:
            weekly_data = json.loads(line.strip())
            date = weekly_data['date']
            user_ids = weekly_data['user_ids']

            followers_set = set() # Use a set to remove duplicates
            for user_id in user_ids:
                followers_set.update(get_followers(user_id))

            followers_list = list(followers_set)

            # Save the result
            result = {
                'date': date,
                'user_ids': followers_list
            }

            with open(output_file, 'a') as out_file:
                out_file.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_weekly_tweet_users(input_file, output_file)
