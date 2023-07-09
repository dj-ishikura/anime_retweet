import pandas as pd
import re

# 終了日が正しい形式（"年月日"）になっているかどうかをチェックする正規表現パターン
pattern1 = r'\d{4}年\d{1,2}月\d{1,2}日'
pattern2 = r'\d{1,2}月\d{1,2}日'

# CSVファイルを読み込む
df = pd.read_csv('anime_data.csv')

# 作成されていないページの場合、削除する
df = df[~df['リンク'].str.contains('index.php')]

# 開始日と終了日を分ける
df[['開始日', '終了日']] = df['開始日-終了日'].str.split('-', expand=True)

# 年の情報を追加する
df['年'] = df['放送期間'].apply(lambda x: x.split('年')[0])

# 削除した行数を表示
print(f"削除した行数: {len(df[df['終了日'].isna()])}")

# 終了日がない行を削除
df = df.dropna(subset=['終了日'])

# 年月日以外の情報が含まれていることがあるので、それを直す
df['開始日'] = df['開始日'].apply(lambda x: 
                                re.search(pattern1, x).group() if re.search(pattern1, x) else 
                                re.search(pattern2, x).group() if re.search(pattern2, x) else 
                                None)
# データフレームの'終了日'列に対して適用します
df['終了日'] = df['終了日'].apply(lambda x: 
                                re.search(pattern1, x).group() if re.search(pattern1, x) else 
                                re.search(pattern2, x).group() if re.search(pattern2, x) else 
                                None)

# 終了日に年が含まれていない場合、開始日に年を追加する
df['開始日'] = df.apply(lambda row: row['年'] + '年' + row['開始日'], axis=1)

# 終了日に年が含まれていない場合、終了日に年を追加する
df['終了日'] = df.apply(lambda row: row['年'] + '年' + row['終了日'] if row['終了日'] is not None and '年' not in row['終了日'] else row['終了日'], axis=1)

# 年の列を削除する
df = df.drop(columns=['年'])

# 列の新しい順序を指定します
new_order = ['開始日-終了日', '開始日', '終了日', '作品名', '制作会社', '主放送局・系列', '話数', '放送期間', 'リンク']

# 列の順序を更新します
df = df.reindex(columns=new_order)

# 削除した行数を表示
print(f"削除した行数: {len(df[df['終了日'].isna()])}")

# 終了日がない行を削除
df = df.dropna(subset=['終了日'])

# 各行にidを振る
df = df.reset_index(drop=True)

# 開始日から月を取得し、数値に変換
df['月'] = df['開始日'].str.extract(r'(\d+)月', expand=False)

# 月が1桁の場合は0を追加して2桁にする
df['月'] = df['月'].apply(lambda x: f'0{x}' if x is not None and len(x) == 1 else x)

# "年-月-番号"の形式で新しい番号を作成
df['id'] = df['開始日'].str[:4] + '-' + df['月'] + '-' + df.index.astype(str)

df = df.drop(columns=['月'])

# id列を最初に移動
df = df.set_index('id').reset_index()

# CSVファイルに出力する
df.to_csv('anime_data_updated.csv', index=False)
