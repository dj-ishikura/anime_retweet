import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties # *日本語対応
import japanize_matplotlib

# CSVファイルからデータフレームを作成
# 前提として、R_valueという列が存在しているCSVファイルがあると仮定
df1 = pd.read_csv('model_tweet_concat.csv')

# 2つ目のCSVファイルからデータフレームを作成
df2 = pd.read_csv('result/class_anime_list.csv')

# df1とdf2を'id'でマージする
merged_df = pd.merge(df1, df2, on='id')

# クラスごとに異なる色でプロット
# クラスごとに異なる色でプロット
classes = merged_df['class'].unique()
for c in classes:
    subset = merged_df[merged_df['class'] == c]
    plt.scatter(subset['id'], subset['R_value'], label=f'{c}')

# グラフ設定
plt.xlabel('id')
plt.ylabel('R値')
plt.legend(title='クラス')
plt.xticks([])  # これで横軸の目盛りラベルを非表示にします
plt.savefig('plot_model_tweet_2.png')
