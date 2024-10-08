import json
from collections import defaultdict

def check_duplicates(file_path):
    tweet_counts = defaultdict(int)
    duplicate_count = 0
    total_count = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            total_count += 1
            try:
                tweet = json.loads(line)
                tweet_id = tweet.get('tweet_id')
                if tweet_id:
                    tweet_counts[tweet_id] += 1
                    if tweet_counts[tweet_id] > 1:
                        duplicate_count += 1
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line}")

    duplicates = {tweet_id: count for tweet_id, count in tweet_counts.items() if count > 1}

    print(f"Total tweets processed: {total_count}")
    print(f"Number of unique tweet IDs: {len(tweet_counts)}")
    print(f"Number of duplicate tweets: {duplicate_count}")
    print(f"Number of tweet IDs with duplicates: {len(duplicates)}")

    if duplicates:
        print("\nDuplicate tweet IDs and their counts:")
        for tweet_id, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            print(f"Tweet ID: {tweet_id}, Count: {count}")

if __name__ == "__main__":
    file_path = "/work/n213304/learn/anime_retweet_2/random_tweets_text_2022_7-9.jsonl"  # JSONLファイルのパスを指定してください
    check_duplicates(file_path)