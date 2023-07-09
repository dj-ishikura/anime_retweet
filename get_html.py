import os
import requests

def save_html(url, folder, filename):
    # ページを取得
    res = requests.get(url)
    res.raise_for_status()

    # フォルダが存在しない場合は作成
    if not os.path.exists(folder):
        os.makedirs(folder)

    # HTMLをファイルに書き出す
    with open(os.path.join(folder, filename), 'w', encoding='utf-8') as f:
        f.write(res.text)

url = "https://ja.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC%E3%81%AE%E3%83%86%E3%83%AC%E3%83%93%E3%82%A2%E3%83%8B%E3%83%A1%E4%BD%9C%E5%93%81%E4%B8%80%E8%A6%A7_(2020%E5%B9%B4%E4%BB%A3_%E5%89%8D%E5%8D%8A)"
folder = "html_files"
filename = "anime_2020.html"

# HTMLを取得して保存
save_html(url, folder, filename)
