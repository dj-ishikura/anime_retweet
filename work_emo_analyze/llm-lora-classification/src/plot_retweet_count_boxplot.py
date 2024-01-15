import sys
import pandas as pd
import matplotlib.pyplot as plt
import json

def load_and_merge_data(tweet_object_file, tweet_emo_file):
    # JSON ファイルの読み込み
    with open(tweet_emo_file, 'r', encoding='utf-8') as file:
        predictions_data = json.load(file)

    # CSV ファイルの読み込み
    retweet_data = pd.read_csv(tweet_object_file)

    # tweet_id 列を文字列型に変換
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df['tweet_id'] = predictions_df['tweet_id'].astype(str)
    retweet_ids = set(predictions_df['tweet_id'])

    retweet_data['tweet_id'] = retweet_data['tweet_id'].astype(str)
    tweet_emo_ids = set(retweet_data['tweet_id'])

    # 結合されなかった tweet_id を特定
    unmerged_ids_predictions = retweet_ids - tweet_emo_ids
    unmerged_ids_retweet = tweet_emo_ids - retweet_ids

    print(f"結合されなかった tweet_id の数 (predictions): {len(unmerged_ids_predictions)}")
    print(f"結合されなかった tweet_id の数 (retweet): {len(unmerged_ids_retweet)}")

    return predictions_df.merge(retweet_data, on='tweet_id')

def plot_boxplot(merged_df, output_png, id):
    plt.figure(figsize=(10, 6))
    boxplot = merged_df.boxplot(column='retweet_count', by='predictions', patch_artist=True)

    colors = ['lightcoral', 'khaki', 'lightblue']  # ポジティブ、ニュートラル、ネガティブの色

    # 各箱に色を設定
    for patch, color in zip(boxplot.artists, colors):
        patch.set_facecolor(color)

    # 各箱の色を設定するために他の要素にもアクセスすることが必要な場合があります
    # 例えば、線の色を設定する場合などです

    plt.title(f'Boxplot of Retweets by Sentiment Category for {id}')
    plt.suptitle('')
    plt.xlabel('Predictions Category')
    plt.ylabel('Retweet Count')
    plt.savefig(output_png)


def main(tweet_object_file, tweet_emo_file, output_png, id):
    merged_df = load_and_merge_data(tweet_object_file, tweet_emo_file)
    plot_boxplot(merged_df, output_png, id)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python plot_retweet_count_boxplot.py <tweet_object_file> <tweet_emo_file> <output_png> <id>")
        sys.exit(1)

    tweet_object_file = sys.argv[1]
    tweet_emo_file = sys.argv[2]
    output_png = sys.argv[3]
    id = sys.argv[4]

    main(tweet_object_file, tweet_emo_file, output_png, id)
