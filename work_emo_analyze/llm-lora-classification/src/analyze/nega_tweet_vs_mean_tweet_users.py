import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 日本語対応
import japanize_matplotlib  # 日本語ラベル対応

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr

def load_data(tweet_emo_dir, tweet_user_dir):
    # データをロードして返す
    data_list = []
    for file_name in os.listdir(tweet_emo_dir):
        if file_name.endswith('.csv'):
            id = os.path.splitext(file_name)[0]
            emo_file_path = os.path.join(tweet_emo_dir, file_name)
            user_file_path = os.path.join(tweet_user_dir, id+"_1_week_tweet_counts.csv")
            data_list.append((id, emo_file_path, user_file_path))
    return data_list

def calculate_emotion_data(id, emo_file_path, user_file_path, df_class, df_title):
    df = pd.read_csv(emo_file_path)
    df_tweet_user = pd.read_csv(user_file_path)

    mean_weekly_tweet_user = df_tweet_user["tweet_users_count"].mean()

    # 各感情の合計ツイート数を計算
    total_positive = df['positive'].sum()
    total_neutral = df['neutral'].sum()
    total_negative = df['negative'].sum()

    # 全ツイート数の合計を計算
    total_tweet_counts = df['tweet_count'].sum()

    emotion_data = {
        'id': id,
        'title': df_title.loc[id, '作品名'],
        'ネガティブ_radio': total_negative / total_tweet_counts,
        'ニュートラル_radio': total_neutral / total_tweet_counts,
        'ポジティブ_radio': total_positive / total_tweet_counts,
        'ネガティブ': total_negative,
        'ニュートラル': total_neutral,
        'ポジティブ': total_positive,
        'mean_weekly_tweet_user': mean_weekly_tweet_user,
        'weekly_tweet_user_clusters': df_class.loc[id, 'weekly_tweet_user_clusters']
    }

    return emotion_data

def plot_emotion_correlation_radio(data_list, emotion):
    plt.figure(figsize=(12, 8))
    cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
    cluster_data = {0: {'x': [], 'y': []}, 1: {'x': [], 'y': []}, 2: {'x': [], 'y': []}, 3: {'x': [], 'y': []}}

    # 平均週間ツイートユーザ数の最大値を取得
    max_tweet_user = max(data['mean_weekly_tweet_user'] for data in data_list)

    for data in data_list:
        cluster = data['weekly_tweet_user_clusters']
        color = cluster_colors[cluster]
        # 平均週間ツイートユーザ数を正規化
        normalized_tweet_user = data['mean_weekly_tweet_user'] / max_tweet_user
        cluster_data[cluster]['x'].append(data[emotion+'_radio'])
        cluster_data[cluster]['y'].append(normalized_tweet_user)
        plt.scatter(data[emotion+'_radio'], normalized_tweet_user, color=color)
    
    # 線形回帰とプロット
    for cluster in cluster_data:
        x_values = cluster_data[cluster]['x']
        y_values = cluster_data[cluster]['y']
        if x_values and y_values:
            correlation = np.corrcoef(x_values, y_values)[0, 1]
            A = np.vstack([x_values, np.ones(len(x_values))]).T
            m, c = np.linalg.lstsq(A, y_values, rcond=None)[0]
            plt.plot(x_values, m*np.array(x_values) + c, color=cluster_colors[cluster], alpha=0.5)
            equation = f"y = {m:.2f}x + {c:.2f}"
            plt.text(max(x_values), m*max(x_values) + c, f"{equation}\nr = {correlation:.2f}", color=cluster_colors[cluster], alpha=0.7)

    plt.xlabel(f'{emotion.capitalize()} ツイートの割合')
    plt.ylabel('正規化された平均週間ツイートユーザ数')
    plt.title(f'アニメ作品別 {emotion.capitalize()} ツイートの割合 vs 正規化された平均週間ツイートユーザ数')
    plt.legend([plt.Line2D([0], [0], color=color, marker='o', linestyle='') for color in cluster_colors.values()], cluster_colors.keys())
    plt.tight_layout()
    plt.savefig(f"./src/analyze/{emotion}_tweet_vs_normalized_mean_tweet_users.png")
    plt.close()


def plot_emotion_correlation(data_list, emotion):
    plt.figure(figsize=(12, 8))
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
    cluster_data = {0: {'x': [], 'y': []}, 1: {'x': [], 'y': []}, 2: {'x': [], 'y': []}, 3: {'x': [], 'y': []}}
    weekly_cluster_labels = {0: '上昇', 1: '下降', 2: '横ばい', 3: 'W型(山型)'}
    mean_cluster_labels = {0: '多い', 1: '中くらい', 2: '少ない'}
    df_boundaries = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class_mean_tweet_users_boudaries.csv")


    all_emotion_values = []
    for data in data_list:
        cluster = data['weekly_tweet_user_clusters']
        color = cluster_colors[cluster]
        label = weekly_cluster_labels[cluster]
        cluster_data[cluster]['x'].append(data[emotion+'_radio'])
        cluster_data[cluster]['y'].append(data['mean_weekly_tweet_user'])
        emotion_value = data[emotion+'_radio']
        all_emotion_values.append(emotion_value)
        plt.scatter(data[emotion+'_radio'], data['mean_weekly_tweet_user'], color=color, label=label, alpha=0.7)
        # plt.text(data[emotion+'_radio'], data['mean_weekly_tweet_user'], data['title'], fontsize=9, alpha=0.7)

    xmax = max(all_emotion_values)
    sorted_boundaries = df_boundaries.sort_values(by='mean_tweet_user_clusters')
    for i in range(len(sorted_boundaries) - 1):
        current_cluster = sorted_boundaries.iloc[i]['mean_tweet_user_clusters']
        next_cluster = sorted_boundaries.iloc[i + 1]['mean_tweet_user_clusters']

        current_max = sorted_boundaries.iloc[i]['max']
        next_min = sorted_boundaries.iloc[i + 1]['min']
        middle_point = (current_max + next_min) / 2

        plt.hlines(middle_point, xmin=0, xmax=xmax, color='black', linestyles='dotted', alpha=0.7)
        plt.text(xmax * 0.95, middle_point, f'{mean_cluster_labels[current_cluster]}\n\n{mean_cluster_labels[next_cluster]}', 
        verticalalignment='center', alpha=0.7, fontsize=16)

    # 線形回帰とプロット
    for cluster in cluster_data:
        x_values = cluster_data[cluster]['x']
        y_values = cluster_data[cluster]['y']
        if x_values and y_values:
            # ピアソン相関係数とp値を計算
            correlation, p_value = pearsonr(x_values, y_values)
            # 相関の有意性をチェック
            if p_value < 0.05:
                print(f"クラスタ {cluster}: 相関係数 = {correlation:.2f}, p値 = {p_value:.3f} (有意)")
            else:
                print(f"クラスタ {cluster}: 相関係数 = {correlation:.2f}, p値 = {p_value:.3f} (非有意)")

            correlation = np.corrcoef(x_values, y_values)[0, 1]
            A = np.vstack([x_values, np.ones(len(x_values))]).T
            m, c = np.linalg.lstsq(A, y_values, rcond=None)[0]
            plt.plot(x_values, m*np.array(x_values) + c, color=cluster_colors[cluster], alpha=0.7, linewidth=2)
            equation = f"y = {m:.2f}x + {c:.2f}"
            # plt.text(max(x_values), m*max(x_values) + c, f"{equation}\nr = {correlation:.2f}", color=cluster_colors[cluster], alpha=0.7)

    plt.xlabel(f'{emotion.capitalize()} ツイートの割合', fontsize=18)
    plt.ylabel('平均週間ツイートユーザ数', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title(f'アニメ作品別 {emotion.capitalize()} ツイートの割合 vs 平均週間ツイートユーザ数')
    legend_elements = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', label=weekly_cluster_labels[cluster]) for cluster, color in cluster_colors.items()]
    plt.legend(handles=legend_elements, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"./src/analyze/{emotion}_tweet_vs_mean_tweet_users.png")
    plt.close()

def plot_emotion_correlation_number(data_list, emotion):
    plt.figure(figsize=(12, 8))
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
    cluster_data = {0: {'x': [], 'y': []}, 1: {'x': [], 'y': []}, 2: {'x': [], 'y': []}, 3: {'x': [], 'y': []}}
    weekly_cluster_labels = {0: '上昇', 1: '下降', 2: '横ばい', 3: 'W型(山型)'}
    mean_cluster_labels = {0: '多い', 1: '中くらい', 2: '少ない'}
    df_boundaries = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class_mean_tweet_users_boudaries.csv")


    all_emotion_values = []
    for data in data_list:
        cluster = data['weekly_tweet_user_clusters']
        color = cluster_colors[cluster]
        label = weekly_cluster_labels[cluster]
        cluster_data[cluster]['x'].append(data[emotion])
        cluster_data[cluster]['y'].append(data['mean_weekly_tweet_user'])
        emotion_value = data[emotion]
        all_emotion_values.append(emotion_value)
        plt.scatter(data[emotion], data['mean_weekly_tweet_user'], color=color, label=label, alpha=0.7)
        # plt.text(data[emotion+'_radio'], data['mean_weekly_tweet_user'], data['title'], fontsize=9, alpha=0.7)

    xmax = max(all_emotion_values)
    sorted_boundaries = df_boundaries.sort_values(by='mean_tweet_user_clusters')
    for i in range(len(sorted_boundaries) - 1):
        current_cluster = sorted_boundaries.iloc[i]['mean_tweet_user_clusters']
        next_cluster = sorted_boundaries.iloc[i + 1]['mean_tweet_user_clusters']

        current_max = sorted_boundaries.iloc[i]['max']
        next_min = sorted_boundaries.iloc[i + 1]['min']
        middle_point = (current_max + next_min) / 2

        plt.hlines(middle_point, xmin=0, xmax=xmax, color='black', linestyles='dotted', alpha=0.7)
        plt.text(xmax * 0.95, middle_point, f'{mean_cluster_labels[current_cluster]}\n\n{mean_cluster_labels[next_cluster]}', 
        verticalalignment='center', alpha=0.7, fontsize=16)

    # 線形回帰とプロット
    for cluster in cluster_data:
        x_values = cluster_data[cluster]['x']
        y_values = cluster_data[cluster]['y']
        if x_values and y_values:
            # ピアソン相関係数とp値を計算
            correlation, p_value = pearsonr(x_values, y_values)
            # 相関の有意性をチェック
            if p_value < 0.05:
                print(f"クラスタ {cluster}: 相関係数 = {correlation:.2f}, p値 = {p_value:.3f} (有意)")
            else:
                print(f"クラスタ {cluster}: 相関係数 = {correlation:.2f}, p値 = {p_value:.3f} (非有意)")

            correlation = np.corrcoef(x_values, y_values)[0, 1]
            A = np.vstack([x_values, np.ones(len(x_values))]).T
            m, c = np.linalg.lstsq(A, y_values, rcond=None)[0]
            plt.plot(x_values, m*np.array(x_values) + c, color=cluster_colors[cluster], alpha=0.7, linewidth=2)
            equation = f"y = {m:.2f}x + {c:.2f}"
            # plt.text(max(x_values), m*max(x_values) + c, f"{equation}\nr = {correlation:.2f}", color=cluster_colors[cluster], alpha=0.7)

    plt.xlabel(f'{emotion.capitalize()} ツイートの数', fontsize=18)
    plt.ylabel('平均週間ツイートユーザ数', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title(f'アニメ作品別 {emotion.capitalize()} ツイートの割合 vs 平均週間ツイートユーザ数')
    legend_elements = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', label=weekly_cluster_labels[cluster]) for cluster, color in cluster_colors.items()]
    plt.legend(handles=legend_elements, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"./src/analyze/{emotion}_tweet_number_vs_mean_tweet_users.png")
    plt.close()

def main():
    tweet_emo_dir = 'tweet_emo_weekly'
    tweet_user_dir = '/work/n213304/learn/anime_retweet_2/count_tweet'
    df_title = pd.read_csv('/work/n213304/learn/anime_retweet_2/anime_data_updated.csv', index_col=0)
    df_class = pd.read_csv("/work/n213304/learn/anime_retweet_2/anime_class.csv", index_col="id")

    data_list = load_data(tweet_emo_dir, tweet_user_dir)
    emotion_data_list = [calculate_emotion_data(id, emo_path, user_path, df_class, df_title) for id, emo_path, user_path in data_list]

    for emotion in ['ネガティブ', 'ニュートラル', 'ポジティブ']:
        plot_emotion_correlation(emotion_data_list, emotion)

if __name__ == "__main__":
    main()

