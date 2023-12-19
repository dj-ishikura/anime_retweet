import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ファイルの読み込み
df = pd.read_csv('anime_retention_rate.csv')

# ヒストグラムの作成
max_rate = df['retention_rate'].max()
bins = [i/10 for i in range(11)]
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='retention_rate', bins=bins, color='skyblue', kde=True)

# 平均値の計算
average = df['retention_rate'].mean()

# 平均値をプロット
plt.axvline(average, color='red', linestyle='--')
plt.text(average, 5, 'Average = {:.2f}'.format(average), rotation=90)

plt.title('Histogram of Retention Rate')
plt.xlabel('Retention Rate')
plt.ylabel('Frequency')
plt.savefig('result/retention_rate_histogram_2.png')
