from bs4 import BeautifulSoup

html = """
<h2><span id=".E5.A4.96.E9.83.A8.E3.83.AA.E3.83.B3.E3.82.AF"></span><span class="mw-headline" id="外部リンク"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">外部リンク</font></font></span></h2>
<ul><li><a rel="nofollow" class="external text" href="https://g-witch.net/">機動戦士ガンダム 水星の魔女 公式サイト</a></li>
<li><a rel="nofollow" class="external text" href="https://www.mbs.jp/g-witch/">機動戦士ガンダム 水星の魔女</a> - 毎日放送による番組サイト</li>
<li><a rel="nofollow" class="external text" href="https://twitter.com/g_witch_m">機動戦士ガンダム 水星の魔女</a> (@g_witch_m) - <a href="/wiki/Twitter" title="ツイッター">Twitter</a></li>
<li><a rel="nofollow" class="external text" href="https://www.tiktok.com/@g_witch_m">機動戦士ガンダム 水星の魔女 公式</a> (@g_witch_m) - <a href="/wiki/TikTok" title="チクタク">TikTok</a></li>
<li><a rel="nofollow" class="external text" href="https://www.onsen.ag/program/g-witch">機動戦士ガンダム 水星の魔女〜アスティカシア高等専門学園 ラジオ委員会〜</a></li></ul>
"""

soup = BeautifulSoup(html, 'html.parser')

twitter_links = []

# Find the h2 tag with the text "外部リンク"
h2_tag = soup.find('span', {'class': 'mw-headline'}, string='外部リンク')
print(h2_tag)
# Find the next sibling ul tag
ul_tag = h2_tag.find_next_sibling('ul')
print(ul_tag)
# Find all the li tags within the ul tag
li_tags = ul_tag.find_all('li')

# Iterate over each li tag
for li in li_tags:
    # Find the a tag within the li tag
    a_tag = li.find('a', {'class': 'external text'})
    # Check if the href attribute of the a tag contains "twitter.com"
    if 'twitter.com' in a_tag['href']:
        # If it does, append it to the twitter_links list
        twitter_links.append(a_tag['href'])

print(twitter_links)
