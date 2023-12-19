from igraph import Graph
import pandas as pd
import os
from igraph import plot  # igraphからplot関数をインポートするのだ
import sys

def get_user_data(user_ids_set, dir_path="follow_user"):
    user_data_dict = {}

    for user_id in user_ids_set:
        user_data = {"followers": [], "following": []}

        try:
            # フォロワーのIDを取得するのだ
            with open(os.path.join(dir_path, f'{user_id}_followers.txt'), 'r') as file:
                followers = [str(line.strip()) for line in file.readlines() if str(line.strip()) in user_ids_set]
            user_data["followers"] = followers

            # フォローしている人のIDを取得するのだ
            with open(os.path.join(dir_path, f'{user_id}_following.txt'), 'r') as file:
                following = [str(line.strip()) for line in file.readlines() if str(line.strip()) in user_ids_set]
            user_data["following"] = following

        except FileNotFoundError:
            print(f"Files for user ID {user_id} not found.")
            continue

        # ユーザーデータを辞書に追加するのだ
        user_data_dict[user_id] = user_data

    return user_data_dict

# ステップ2: igraphオブジェクトの作成
def create_weekly_graphs(jsonl_path, output_dir, dir_path="follow_user"):
    user_data_df = pd.read_json(jsonl_path, lines=True)

    # ユーザIDのセットを作成するのだ
    user_ids_set = set()
    for user_ids in user_data_df['user_ids']:
        user_ids_set.update(map(str, user_ids))


    # 全体のツイートしたユーザidのセット
    user_data_dict = get_user_data(user_ids_set, dir_path)
    # print(f'user_data_dict : {user_data_dict}')

    graph_stat_list = []
    for index, row in user_data_df.iterrows():
        date = row['date']
        weekly_user_ids_set = set(map(str, row['user_ids']))
        # print(f'weekly_user_ids_set : {weekly_user_ids_set}')


        # その週のユーザーデータを取得するのだ
        weekly_user_dict = {user_id: user_data_dict[user_id] for user_id in weekly_user_ids_set if user_id in user_data_dict}
        # print(f'weekly_user_dict : {weekly_user_dict}')

        # ユーザーIDと頂点IDを関連付けるマッピングを作成するのだ
        id_to_vertex = {user_id: i for i, user_id in enumerate(user_ids_set)}
        vertex_to_id = {i: user_id for i, user_id in enumerate(user_ids_set)}

        # igraph グラフの作成
        g = Graph(directed=True)

        # ユーザーIDを頂点の名前として追加するのだ
        g.add_vertices(list(map(str, weekly_user_dict.keys())))

        # エッジのリストを作成するのだ
        edges = []
        for user_id, data in weekly_user_dict.items():
            user_index = g.vs.find(name=str(user_id)).index
            for follower_id in data["followers"]:
                # フォロワーがその週のユーザー辞書に存在するかを確認するのだ
                if str(follower_id) in weekly_user_dict:
                    follower_index = g.vs.find(name=str(follower_id)).index
                    edges.append((follower_index, user_index))
            for following_id in data["following"]:
                # フォローしている人がその週のユーザー辞書に存在するかを確認するのだ
                if str(following_id) in weekly_user_dict:
                    following_index = g.vs.find(name=str(following_id)).index
                    edges.append((user_index, following_index))

        # エッジを追加するのだ
        g.add_edges(edges)

        # グラフの可視化と保存
        out_file = f'{output_dir}/{date}.png'
        graph_plot(g, out_file)

        # グラフの保存
        g.write_graphml(f'{output_dir}/{date}.graphml')

        graph_stat_list.append(graph_stat(g))
    
    df_csv = pd.DataFrame(graph_stat_list)
    id = jsonl_path.split('/')[-1].replace(".jsonl", "")
    df_csv.to_csv(f'{output_dir}/{id}.csv')

def graph_plot(graph, output_path):
    # グラフの描画設定を行うのだ
    # 以下は基本的な設定であり、必要に応じて変更可能なのだ
    layout = graph.layout("fr")
    plot(graph, output_path, layout=layout, vertex_size=5, vertex_label_size=10, edge_arrow_size=0.5)

def graph_stat(g):
    n_vertices = g.vcount()  # 頂点の数
    print(f'n_vertices : {n_vertices}')  # ここで n_vertices を印刷するのだ

    if n_vertices == 0:
        # 他の統計情報も0や無効な値に設定するのだ
        n_edges, density, diameter, avg_path_length, avg_degree, clustering_coefficient = [0] * 6
        page_rank = [0]
    else:
        n_edges = g.ecount()  # エッジの数
        density = g.density()  # 密度
        diameter = g.diameter()  # 直径
        avg_path_length = g.average_path_length()  # 平均パス長
        avg_degree = sum(g.degree()) / n_vertices  # 平均次数
        page_rank = g.pagerank()  # PageRank
        clustering_coefficient = g.transitivity_avglocal_undirected()  # クラスタ係数

    data = {
        "Number of vertices": n_vertices,
        "Number of edges": n_edges,
        "Density": density,
        "Diameter": diameter,
        "Average path length": avg_path_length,
        "Average degree": avg_degree,
        "Average PageRank": sum(page_rank) / n_vertices if n_vertices != 0 else 0,  # ここを修正するのだ
        "Clustering coefficient": clustering_coefficient
    }

    return data

if __name__ == "__main__":
    # JSONLファイルへのパスを提供するのだ
    jsonl_path = sys.argv[1]
    output_dir = sys.argv[2]
    create_weekly_graphs(jsonl_path, output_dir)