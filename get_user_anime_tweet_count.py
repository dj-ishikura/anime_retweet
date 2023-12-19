from collections import defaultdict
import json
import sys

def get_user_tweet_counts(input_path):
    user_tweet_counts = defaultdict(int)
    user_tweet_ids = defaultdict(set)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # タブで区切られた行を分割し、2列目（0から数えて1）をJSONとして解析
            line = line.replace(",", "\t", 2)
            json_string = line.split('\t')[2]
            tweet = json.loads(json_string.rstrip('\n|\r'))

            # ツイートのみを取得
            if "retweeted_status" in tweet:
                tweet = tweet["retweeted_status"]
            
            if "created_at" in tweet and "user" in tweet:
                user_id = tweet["user"]["id_str"]
                tweet_id = tweet["id_str"]

                # ユーザがそのツイートを既にツイートしていないか確認
                if tweet_id not in user_tweet_ids[user_id]:
                    user_tweet_ids[user_id].add(tweet_id)
                    user_tweet_counts[user_id] += 1

    return user_tweet_counts

# 出力データのフォーマットをJSONLにするための関数
def save_as_jsonl(data, output_path, id):
    with open(output_path, 'w', encoding='utf-8') as f:
        for user_id, tweet_count in data.items():
            f.write(json.dumps({"user_id": user_id, f'{id}': tweet_count}) + '\n')

# 使用例
input_path = sys.argv[1]
output_path = sys.argv[2]
id = sys.argv[3]
tweet_counts = get_user_tweet_counts(input_path)
save_as_jsonl(tweet_counts, output_path, id)
