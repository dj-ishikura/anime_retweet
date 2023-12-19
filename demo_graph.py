from igraph import Graph, Plot
import random

def create_combined_network_graph(data_sets, file_name):
    # 複数のデータセットを組み合わせる
    combined_data = [edge for data_set in data_sets for edge in data_set]
    
    # ノードとエッジの追加
    followers, following = zip(*combined_data)
    users = list(set(followers + following))
    user_to_index = {user: i for i, user in enumerate(users)}
    
    edges = [(user_to_index[f], user_to_index[t]) for f, t in combined_data]
    
    # グラフの初期化
    g = Graph(directed=True)
    g.add_vertices(len(users))
    g.add_edges(edges)
    
    # グラフのレイアウトと可視化
    layout = g.layout("kk")  # Kamada-Kawayiレイアウトを使用
    plot = Plot()
    plot.add(g, layout=layout, vertex_size=20, vertex_label=users, 
             vertex_color=[random.choice(["red", "green", "blue"]) for _ in users], 
             edge_arrow_size=0.5)
    plot.save(file_name)

# データの準備 (ここでは2つのダミーデータセットを使用)
data_sets = [
    [(1, 2), (2, 3), (3, 4), (2, 4), (3, 5), (4, 5)], 
    [(3, 4), (4, 5), (5, 6), (4, 6), (5, 7), (6, 7)]
]

# 複数のデータセットを組み合わせてグラフを作成
create_combined_network_graph(data_sets, 'demo_network_graph.png')
