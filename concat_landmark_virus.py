import sys
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from geopy.distance import geodesic  # 緯度経度から直線距離を計算するためのツール


def spot_distance(nodes, spot_dtf):
    direct_distance_list = []
    for _, node in nodes.iterrows():
        source = (node['y'], node['x'])
        direct_distance = []
        for _, spot in spot_dtf.iterrows():
            destination = (spot['latitude'], spot['longitude'])
            x = geodesic(source, destination).km
            direct_distance.append(x)
        direct_distance_list.append(direct_distance)
    nodes['spot_distance'] = direct_distance_list

    # 更新されたデータフレームを返す
    return nodes


def sort_virus_nodes(spot_dtf, nodes2):
    """
    出発地ノードからの距離にしたがって各ウイルス内のノードを並び替える関数。

    Args:
        spot_dtf (pandas.DataFrame): 観光スポットのデータフレーム。`virus_node`列が必要。
        nodes2 (pandas.DataFrame): 各ノードの情報を含むデータフレーム。少なくとも`osmid`と`start_distance`列が必要。

    Returns:
        dtf: 出発地に近い順に並び変わったウイルスノードリストを含むスポットデータフレーム。
    """

    virus_nodes = spot_dtf["virus_nodes"]
    virus_nodes_sorted = []

    for virus in virus_nodes:
        selected_nodes_dict = {
            node: nodes2[nodes2["osmid"] == node]["start_distance"].values[0]
            for node in virus
        }
        # 距離順にノードをソート
        sorted_nodes = sorted(selected_nodes_dict.keys(), key=lambda n: selected_nodes_dict[n])
        virus_nodes_sorted.append(sorted_nodes)
    spot_dtf["virus_nodes"] = virus_nodes_sorted
    return spot_dtf


def rechoice_landmark(landmark_dtf, spot_dtf, nodes2):
    """
    Returns:
        dtf: ウイルスノードとして選ばれているノードが除去されたランドマークが出発地に近い順に並んだデータフレーム。
    """
    virus_nodes_set = set()
    for virus_nodes in spot_dtf["virus_nodes"]:
        virus_nodes_set.add(virus_nodes)

    landmark_nodes_set = set(landmark_dtf["osmid"].to_list())
    # 選択されたランドマークがウイルスノードと重複しないかをチェック
    duplicated_nodes = landmark_nodes_set & virus_nodes_set
    print(f"{len(duplicated_nodes) = }")
    for node in duplicated_nodes:
        all_neighbors = set(G.neighbors(node))  # 近傍ノードを選択
        while True:
            flag == 0
            for neighbor in all_neighbors:
                flag = 0
                if neighbor not in virus_nodes_set: flag += 1
                if neighbor not in landmark_nodes_set: flag += 1
                if flag == 2:
                    # 旧ランドマークの削除と新ランドマークの追加
                    landmark_nodes_set.remove(node)
                    landmark_dtf = landmark_dtf[landmark_dtf['osmid'] != node]
                    landmark_nodes_set.add(neighbor)
                    new_node_dtf = nodes2[nodes2["osmid"] == neighbor]
                    landmark_dtf = pd.concat([landmark_dtf, new_node_dtf], ignore_index=True)
                    break
            if flag == 2: break
            visited += all_neighbors
            next_neighbors = set([new_neighbor for neighbor in all_neighbors for new_neighbor in G.neighbors(neighbor)])
            all_neighbors = next_neighbors-visited
    
    return landmark_dtf.sort_values('start_distance')


# カスタム変換関数
def str_to_list(val):
    return ast.literal_eval(val)


if __name__ == "__main__":
    city = sys.argv[1]  # 対象都市名：hachioji, yokohama
    frmt = int(sys.argv[2])  # 出発点・到着点の緯度経度が書かれたファイル：from-to_01.txt
    start, end = [tuple(eval(row.rstrip())) for row in open(f"data/{city}/from-to_{frmt:02d}.txt")]  # 出発点と到着点の緯度経度を設定    
    virus = sys.argv[3]  # ウイルス種類名：bank
    lbd1 = float(sys.argv[4])  # 距離減衰係数lbd1：0.05, 0.1, 0.5, 1.0, 5.0
    landmark_num = int(sys.argv[5])  # 選択するランドマークの数：100, 200

    nodes1 = pd.read_csv(f"data/{city}/nodes_{virus}.csv", index_col=0, converters={'spot_distance': str_to_list})
    spot_dtf = pd.read_csv(f"data/{city}/{virus}_spot_dtf.csv", index_col=0, converters={'virus_nodes': str_to_list})

    nodes2 = pd.read_csv(f"data/{city}/nodes_frmt{frmt:02d}.csv", index_col=0, converters={'start_distance': str_to_list})
    landmark_dtf_fp = f"data/{city}/landmark_dtf_frmt{frmt:02d}_L{landmark_num:03d}_{int(lbd1*100):03d}.csv"
    landmark_dtf = pd.read_csv(landmark_dtf_fp, index_col=0, converters={'spot_distance': str_to_list})

    G = ox.load_graphml(filepath=f"data/{city}/road_network.graphml")
    print(f"{len(spot_dtf) = }")
    spot_dtf = sort_virus_nodes(spot_dtf, nodes2)
    print(f"{len(spot_dtf) = }")    
    spot_dtf.to_csv(f"data/{city}/virus_frmt{frmt:02d}_{virus}.csv")

    print(f"{len(landmark_dtf) = }")
    landmark_dtf = rechoice_landmarks(G, landmark_dtf, spot_dtfm  nodes2)
    print(f"{len(landmark_dtf) = }")
    start_node = ox.distance.nearest_nodes(G, start[1], start[0])  # 指定された開始位置（start）に最も近いノードを探す
    end_node = ox.distance.nearest_nodes(G, end[1], end[0])  # 終了位置（end）に最も近いノードを探す
    start_row = nodes[nodes["osmid"] == start_node]  # 始点ノードのデータ行を取得
    end_row = nodes[nodes["osmid"] == end_node]  # 終点ノードのデータ行を取得
    landmark_dtf = pd.concat([start_row, landmark_dtf, end_row]).reset_index(drop=True)  # 始点・終点のデータをdtfに追加
    print(f"{len(landmark_dtf) = }")    
    vector = np.array([start[1] - end[1], end[0] - start[0]])  # 始点と終点のベクトルを計算
    landmark_xy = landmark_dtf[["y", "x"]].values.tolist()  # ランドマークの座標をリストに変換
    landmark_xy_np = np.array(landmark_xy)
    projection_value = np.dot(landmark_xy_np, vector)  # ランドマークの射影した値を計算
    landmark_dtf = landmark_dtf.copy()  # 射影した値をdtfに追加
    landmark_dtf["projection_value"] = list(projection_value)
    landmark_dtf.to_csv(f"data/{city}/landmark_frmt{frmt:02d}_L{landmark_num:03d}_{int(lbd1*100):03d}_{virus}.csv")

    nodes2 = spot_distance(nodes2, spot_dtf)
    nodes2 = nodes2[["osmid", "y", "x", "start_distance", "end_distance", "spot_distance"]]
    nodes2.to_csv(f"data/{city}/nodes_frmt{frmt:02d}_{virus}.csv")
    
