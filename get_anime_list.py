import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_link(table):
    link_list = []
    for row in table.find_all('tr'):
        columns = row.find_all('td')
        # アニメの詳細ページへのリンクを取得
        if len(columns) > 1:
            link = columns[1].find('a')
            link = 'https://ja.wikipedia.org' + link.get('href')
            link_list.append(link)
    return link_list

# URLを指定します
url = "https://ja.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC%E3%81%AE%E3%83%86%E3%83%AC%E3%83%93%E3%82%A2%E3%83%8B%E3%83%A1%E4%BD%9C%E5%93%81%E4%B8%80%E8%A6%A7_(2020%E5%B9%B4%E4%BB%A3_%E5%89%8D%E5%8D%8A)"

# URLからHTMLを取得します
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# ページ内の全てのテーブルを取得します
tables = soup.find_all('table', {'class': 'wikitable'})

# 各テーブルからデータを取得し、データフレームのリストを作成します
dfs = []
for table in tables:
    # テーブルのタイトル（放送期間）を取得します
    title = table.find_previous('h3')
    if title:
        broadcast_period = title.get_text(strip=True)
        print(f"放送期間: {broadcast_period}")

    # テーブルからデータを取得します
    df = pd.read_html(str(table), header=0)[0]
    df.columns = df.columns.str.replace('\[.*\]', '', regex=True)  # ヘッダーから"[]"とその中の情報を削除します
    df.columns = df.columns.str.replace(' ', '', regex=True)  # ヘッダーから空白を削除します
    df.columns = df.columns.str.replace('‑', '-')  # ヘッダーの"‑"を"-"に変更します
    df['放送期間'] = broadcast_period  # 放送期間を新たな列として追加します
    link_list = get_link(table)
    df['リンク'] = link_list
    dfs.append(df)

# 全てのデータフレームを結合します
df_all = pd.concat(dfs)

# 結果をCSVファイルに出力します
df_all.to_csv('anime_data.csv', index=False)
