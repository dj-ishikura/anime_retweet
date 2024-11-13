#!/bin/bash
# 大きめのクラスタを作るための設定
# M=64 qcmd python src/bertopic_clustering.py --min_samples=5 --min_cluster_size=40 --cluster_selection_epsilon=0.9

# さらにクラスタサイズを大きくしたい場合
# M=64 qcmd python src/bertopic_clustering.py --min_samples=5 --min_cluster_size=40 --cluster_selection_epsilon=0.8

# より細かく調整したい場合
# M=64 qcmd python src/bertopic_clustering.py --min_cluster_size=100 --min_samples=10 --cluster_selection_epsilon=0.5

# 12:55
M=64 qcmd python src/bertopic_clustering.py --min_samples=10 --min_cluster_size=10 --cluster_selection_epsilon=0.0