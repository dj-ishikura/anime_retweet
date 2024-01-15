import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties # 日本語対応
import japanize_matplotlib

# Excelファイルのパスを指定する
file_path = 'resampling_predicted_tweet_nega.xlsx'
data = pd.read_excel(file_path)

# 予測が0で、手動ラベルが1または2のデータをフィルタリングする
filtered_data = data[(data['predictions'] == 0) & (data['manual_label'].isin([1, 2]))]

# プロットの設定（1行2列のサブプロット）
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# キーワードバイアスによる分布を棒グラフでプロットする
sns.countplot(data=filtered_data, x='manual_label', hue='keyword_bias', ax=axs[0])
axs[0].set_title('単語に引っ張られて誤分類をした数')
axs[0].set_xlabel('手動ラベル')
axs[0].set_ylabel('Count')
axs[0].legend(title='Keyword Bias', loc='upper right')
axs[0].set_xticklabels(['ニュートラル', 'ポジティブ'])

# 手動ラベルの割合を円グラフでプロットする
manual_label_counts = data['manual_label'].value_counts().reindex([2, 1, 0], fill_value=0)
manual_label_percentages = (manual_label_counts / manual_label_counts.sum()) * 100
axs[1].pie(manual_label_percentages, labels=['ポジティブ', 'ニュートラル', 'ネガティブ'], 
           colors=['lightcoral', 'khaki', 'lightblue'], autopct='%1.1f%%')
axs[1].set_title('モデルがネガティブと分類したものに対し, 手動でラベル付けした結果')

# 画像を保存する
plt.savefig("./resampling_predicted_analyze_nega.png")
plt.close()
