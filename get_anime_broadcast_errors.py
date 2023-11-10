import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
from dateutil import parser
import pytz

# 話数から数字のみを抽出して、数値かどうかを判断する関数
def extract_numeric_part(episode_str):
    # episode_strがNaN（float型）の場合、空の文字列に変換する
    episode_str = '' if pd.isna(episode_str) else episode_str
    
    # 数字のみを抽出する
    numeric_part = re.sub(r'[^\d.]+', '', episode_str)
    
    # 先頭のゼロを削除する（1つ以上の先頭ゼロにマッチする）
    numeric_part = numeric_part.lstrip('0')
    
    # 数値が空の文字列であれば、そのまま空の文字列を返す
    return numeric_part if numeric_part else ''

def is_integer_episode(episode_str):
    # 話数から数字の部分を抽出する
    numeric_part = extract_numeric_part(episode_str)
    
    # 抽出した部分が整数形式かどうかをチェックする
    return numeric_part.isdigit() and not numeric_part.startswith('0')

def get_anime_period(anime_id, anime_data_df):
    # '開始日' と '終了日' は、すでに文字列として格納されているため、
    # ここで直接 parse_japanese_date 関数を呼び出す。
    
    start_date = anime_data_df.loc[anime_id, '開始日']
    end_date = anime_data_df.loc[anime_id, '終了日']
    start_date = datetime.strptime(start_date, "%Y年%m月%d日")
    end_date = datetime.strptime(end_date, "%Y年%m月%d日")
    print(start_date)
    return start_date, end_date

def check_all_integer_episodes(broadcast_df):
    # '話数'列のすべての値が整数かどうかをチェック
    return all(broadcast_df['話数'].apply(is_integer_episode))

# 不備のあるアニメを記録する関数
def record_irregular_anime(anime_id, title, output_df):
    output_df = output_df.append({'id': anime_id, 'title': title}, ignore_index=True)
    return output_df

# 放送日が1週間ごとかどうかを確認する関数
def check_weekly_broadcast(dates):
    for i in range(len(dates) - 1):
        if dates[i + 1] - dates[i] != timedelta(days=7):
            return False
    return True

def parse_date(date_str):
    # 日付のフォーマットに合わせて datetime オブジェクトに変換
    # フォーマットが合致しない場合は NaT を返す
    return pd.to_datetime(date_str, format='%Y年%m月%d日', errors='coerce')

# ディレクトリ内のTSVファイルを処理する関数
def process_tsv_files(directory_path, anime_data_df, output_filename):
    irregular_anime_df = pd.DataFrame(columns=['id', 'title'])

    for tsv_file in os.listdir(directory_path):
        print(tsv_file)
        if tsv_file.endswith('.tsv'):
            anime_id = tsv_file.split('.')[0]
            broadcast_df = pd.read_csv(os.path.join(directory_path, tsv_file), sep='\t', dtype={'話数':str})

            # '初放送日'列が存在するかをチェックし、存在する行のみを処理
            if '初放送日' in broadcast_df.columns:
                # '初放送日'列をdatetimeオブジェクトに変換
                broadcast_df['初放送日'] = pd.to_datetime(broadcast_df['初放送日'], format='%Y年%m月%d日', errors='coerce')
                
                # Pandas Timestampからdatetime.datetimeオブジェクトに変換
                broadcast_df['初放送日'] = broadcast_df['初放送日'].dt.to_pydatetime()
                
                # 放送期間を取得
                start_date, end_date = get_anime_period(anime_id, anime_data_df)

                # 放送期間内の放送日データを抽出
                period_broadcast_df = broadcast_df[
                    (broadcast_df['初放送日'] >= start_date) &
                    (broadcast_df['初放送日'] <= end_date)
                ]

            # 話数が整数でないエピソード、または1週間ごとに放送されていない場合は不備として記録
            irregular_list = []
            period_broadcast_df['話数'] = period_broadcast_df['話数'].apply(extract_numeric_part)
            print(period_broadcast_df['話数'])
            print(check_all_integer_episodes(period_broadcast_df))
            if not check_all_integer_episodes(period_broadcast_df):
                irregular_list.append("総集編or漢数字")

            if not check_weekly_broadcast(period_broadcast_df['初放送日'].tolist()):
                irregular_list.append("週とび")

            if irregular_list != []:
                title = anime_data_df.loc[anime_id, '作品名']
                irregular_anime_df = irregular_anime_df.append({'id': anime_id, 'title': title, 'reason':irregular_list}, ignore_index=True)

    # 不備のあるアニメの情報をCSVファイルに保存
    irregular_anime_df = irregular_anime_df.sort_values(by='id')
    irregular_anime_df = irregular_anime_df.reset_index(drop=True)
    irregular_anime_df.to_csv(output_filename, index=False, sep='\t')


# メイン関数
def main():
    anime_data_filename = 'anime_data_updated.csv'  # アニメの放送情報があるCSVファイルのパス
    anime_data_df = pd.read_csv(anime_data_filename, index_col=0)
    directory_path = 'ep_list'  # TSVファイルがあるディレクトリのパス
    output_filename = 'anime_broadcast_errors.tsv'  # 出力するCSVファイル名

    # 処理を実行
    process_tsv_files(directory_path, anime_data_df, output_filename)

if __name__ == "__main__":
    main()
