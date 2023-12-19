import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ファイルの読み込み
anime_class_df = pd.read_csv('result/class_anime_list.csv')
retention_rate_df = pd.read_csv('result/anime_retention_rate.csv')
mean_df = pd.read_csv('result/tweet_mean.csv')
growth_df = pd.read_csv('result/count_tweet_4times_growth_rates_and_counts.csv')

# データフレームをマージします
df = pd.merge(anime_class_df, retention_rate_df, on='id')
df = pd.merge(df, mean_df, on='id')

# クラスと色の対応
color_dict = {'trend': 'red', 'hit': 'green', 'miner': 'blue'}

# クラスごとのヒストグラムを作成します
plt.figure(figsize=(10,6))

for class_name in ['miner', 'hit', 'trend']:
# for class_name in ['hit', 'trend']:
    subset = df[df['class'] == class_name]
    sns.histplot(data=subset, x='retention_rate', bins=[i/10 for i in range(11)], 
    kde=False, label=class_name, color=color_dict[class_name], alpha=0.4)

    # クラスごとの平均と標準偏差を求めます
    average = subset['retention_rate'].mean()
    std_dev = subset['retention_rate'].std()
    print('{} : average {}, std {}'.format(class_name, average, std_dev))

    # 平均と標準偏差をプロットします
    plt.axvline(average, color=color_dict[class_name], linestyle='--')
    # plt.text(average+0.02, 5, '{} Avg = {:.2f}\n{} Std Dev = {:.2f}'.format(class_name, average, class_name, std_dev), rotation=90, color=color_dict[class_name])

# グラフの設定
plt.title('Histogram of Retention Rate by Class')
plt.xlabel('Retention Rate')
plt.ylabel('Frequency')
plt.legend(title='Class')
plt.savefig('result/retention_rate_histogram_by_class.png')

plt.figure(figsize=(10,6))

# クラスごとに散布図をプロットします
for class_name in ['hit', 'trend']:
    subset = df[df['class'] == class_name]
    
    # 散布図をプロットします
    plt.scatter(subset['retention_rate'], subset['mean'], label=class_name, color=color_dict[class_name], alpha=0.4)
    
    # クラスごとの平均をプロットします
    average = subset['mean'].mean()
    plt.axhline(average, color=color_dict[class_name], linestyle='--')
    average = subset['retention_rate'].mean()
    plt.axvline(average, color=color_dict[class_name], linestyle='--')

# グラフの設定
plt.title('Scatter Plot of Retention Rate by Class')
plt.xlabel('Retention Rate')
plt.ylabel('Mean')
plt.legend(title='Class')
plt.savefig('result/retention_rate_scatter_by_class.png')

# 増加率と定着率
df = pd.merge(df, growth_df, on='id')
plt.figure(figsize=(10,6))

# クラスごとに散布図をプロットします
for class_name in ['miner', 'hit', 'trend']:
    subset = df[df['class'] == class_name]
    
    # 散布図をプロットします
    plt.scatter(subset['retention_rate'], subset['average_growth_rate'], label=class_name, color=color_dict[class_name], alpha=0.4)
    
    # クラスごとの平均をプロットします
    average = subset['average_growth_rate'].mean()
    plt.axhline(average, color=color_dict[class_name], linestyle='--')
    average = subset['retention_rate'].mean()
    plt.axvline(average, color=color_dict[class_name], linestyle='--')

# グラフの設定
plt.title('Scatter Plot of Retention Growth Rate by Class')
plt.xlabel('Retention Rate')
plt.ylabel('average growth rate')
plt.legend(title='Class')
plt.savefig('result/retention_growth_rate_scatter_by_class.png')

from scipy import stats

# 'trend'クラスの定着率
trend_retention_rate = df[df['class'] == 'trend']['retention_rate']

# 'trend'以外のクラスの定着率
# other_retention_rate = df[df['class'] == 'hit']['retention_rate']
other_retention_rate = df[df['class'] != 'trend']['retention_rate']
# t検定を実行
t, p = stats.ttest_ind(trend_retention_rate, other_retention_rate)

print(f"t-test: t = {t}, p = {p}")

# 'trend'クラスの定着率
trend_retention_rate = df[df['class'] == 'trend']['retention_rate']

# 'trend'以外のクラスの定着率
other_retention_rate = df[df['class'] == 'hit']['retention_rate']
# t検定を実行
t, p = stats.ttest_ind(trend_retention_rate, other_retention_rate)

print(f"t-test: t = {t}, p = {p}")

df.to_csv('result/anime_class_retention_rate.csv', index=False)
df[df['retention_rate']>0.5].to_csv('result/anime_class_retention_rate_over_0.5.csv', index=False)