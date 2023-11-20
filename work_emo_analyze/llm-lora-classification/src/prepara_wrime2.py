import pandas as pd
import os

# ファイルの読み込み
file_path = '../wrime/wrime-ver2.tsv'
df = pd.read_csv(file_path, sep='\t')

# 必要な列のみを選択
columns = [
    'Sentence', 'UserID', 'Train/Dev/Test',
    'Avg. Readers_Joy', 'Avg. Readers_Sadness', 'Avg. Readers_Anticipation',
    'Avg. Readers_Surprise', 'Avg. Readers_Anger', 'Avg. Readers_Fear',
    'Avg. Readers_Disgust', 'Avg. Readers_Trust', 'Avg. Readers_Sentiment'
]
df = df[columns]

# ディレクトリの作成
output_dir = 'datasets/wrime2'
os.makedirs(output_dir, exist_ok=True)

# データの分割とTSV形式での保存
for split in ['train', 'dev', 'test']:
    split_df = df[df['Train/Dev/Test'] == split]
    output_file_path = os.path.join(output_dir, f'{split}.tsv')
    split_df.to_csv(output_file_path, sep='\t', index=False)

# データの分割
train_df = df[df['Train/Dev/Test'] == 'train']
dev_df = df[df['Train/Dev/Test'] == 'dev']
test_df = df[df['Train/Dev/Test'] == 'test']

# 各データセットにおけるユニークなUserIDの数を計算
unique_users_train = train_df['UserID'].nunique()
unique_users_dev = dev_df['UserID'].nunique()
unique_users_test = test_df['UserID'].nunique()

# 結果の表示
print(f"Unique UserIDs in Train: {unique_users_train}")
print(f"Unique UserIDs in Dev: {unique_users_dev}")
print(f"Unique UserIDs in Test: {unique_users_test}")