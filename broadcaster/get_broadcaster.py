import sys
from bs4 import BeautifulSoup
import numpy as np
import re

# コマンドライン引数からHTMLファイルのパスを取得するのだ

html_file_path = sys.argv[1]
# html_file_path = "../html_files/anime_wiki_page/2021-07-319.html"

output_file = sys.argv[2]
# output_file = 'a.csv'

# HTMLファイルを読み込むのだ
with open(html_file_path, 'r', encoding='utf-8') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')

# id="放送局"を持つh3タグを取得するのだ
span_tag = soup.find('span', id=lambda x: x and '放送局' in x)
if span_tag is None:
    print("指定したidのh3タグが見つからないのだ！")
    sys.exit(1)

# h3タグの次のタグから次のh3タグが来るまでのタグを取得するのだ
tables = []
current_tag = span_tag.find_next()
while current_tag and ('DVD' not in current_tag.get('id', '')) and ('BD' not in current_tag.get('id', '')):
    if current_tag.name == 'table':
        tables.append(current_tag)
    current_tag = current_tag.find_next()


import csv
import pandas as pd

# テーブルの内容をDataFrameに変換する関数なのだ
def table_to_dataframe(table, type_name):
    if ('日本国内' not in table.get_text()) and ('NHK' not in table.get_text()) and ('フジテレビ' not in table.get_text()):
        return pd.DataFrame()

    rows = table.find_all('tr')
    header = [col.get_text().strip().split(' ')[0] for col in rows[0].find_all('th')]
    header_length = len(header)

    data = []
    for row in rows[1:]:
        columns = row.find_all('td')
        column_data = [col.get_text().strip() for col in columns]
        # 列数がヘッダと一致する場合のみ、データを追加するのだ
        if len(column_data) == header_length:
            data.append(column_data)
    
    df = pd.DataFrame(data, columns=header)

    # 列名を変更するのだ
    if any([col for col in df.columns if '期間' in col]):
        col_name = [col for col in df.columns if '期間' in col][0]
        df.rename(columns={col_name: '期間'}, inplace=True)
    
    if any([col for col in df.columns if '配信開始' in col]):
        col_name = [col for col in df.columns if '配信開始' in col][0]
        df.rename(columns={col_name: '期間'}, inplace=True)

    if any([col for col in df.columns if '放送日' in col]):
        col_name = [col for col in df.columns if '放送日' in col][0]
        df.rename(columns={col_name: '期間'}, inplace=True)

    if any([col for col in df.columns if '時間' in col]):
        col_name = [col for col in df.columns if '時間' in col][0]
        df.rename(columns={col_name: '時間'}, inplace=True)

    if any([col for col in df.columns if '日時' in col]):
        col_name = [col for col in df.columns if '日時' in col][0]
        df.rename(columns={col_name: '時間'}, inplace=True)

    # 配信サイトが複数ある場合、それを1つずつの行に分けるのだ
    if '配信サイト' in df.columns:
        df = df.drop('配信サイト', axis=1).join(df['配信サイト'].str.split('\n', expand=True).stack().reset_index(level=1, drop=True).rename('配信サイト'))
        df = df.reset_index(drop=True)

    # 配信か放送かを示す列を追加するのだ
    df['タイプ'] = type_name

    return df

dataframes = []
# 各テーブルの内容をDataFrameに変換するのだ
for table in tables:
    # テーブルの内容によって、「放送」か「配信」のラベルを決定するのだ
    if '放送局' in table.get_text():
        df = table_to_dataframe(table, '放送')
    else:
        df = table_to_dataframe(table, '配信')
    dataframes.append(df)

# 全てのDataFrameを結合するのだ
result_df = pd.concat(dataframes, ignore_index=True)

def convert_date_format(date_str):
    if date_str:
        match = re.match(r"(\d+)年(\d+)月(\d+)日", date_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return ''

# 期間の欠損値を前の行の値で補完するのだ
if '期間' in result_df.columns:
    result_df['期間'] = result_df['期間'].replace('', np.nan)
    result_df['期間'].fillna(method='ffill', inplace=True)
    result_df['開始'] = result_df['期間'].str.split('-').str[0].str.strip()
    result_df['終了'] = result_df['期間'].str.split('-').str[1].str.strip().fillna('')  # fillnaを追加して欠損値を空文字にするのだ
    
    # 年の情報が終了日にない場合、開始日の年を追加するのだ
    for i, row in result_df.iterrows():
        if '年' not in row['終了'] and '以降' not in row['開始']:
            year = row['開始'].split('年')[0]
            result_df.at[i, '終了'] = year + '年' + row['終了']
    result_df['開始'] = result_df['開始'].apply(convert_date_format)
    result_df['終了'] = result_df['終了'].apply(convert_date_format)

if '時間' in result_df.columns:
    result_df['時間'] = result_df['時間'].replace('', np.nan)
    result_df['時間'].fillna(method='ffill', inplace=True)

# 前に出したい列名をリストとして指定するのだ
cols_to_front = ['開始', '終了', 'タイプ', '放送局', '配信サイト', '時間']

# 新しい列順序を生成するのだ
new_cols = cols_to_front + [col for col in df if col not in cols_to_front]

# 列の順序を変更するのだ
result_df = result_df[new_cols]

result_df.to_csv(output_file, index=False)


