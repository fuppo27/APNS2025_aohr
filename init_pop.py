import sys
import osmnx as ox
import networkx as nx
import pandas as pd
import random as rnd
import numpy as np
import ast


def pop_info(G, landmark_dtf, pop_length, route_order, end_node, alpha=2, beta=5, threshold=0):
    """
    指定されたランドマークの順序に基づいて経路情報を計算する関数。

    この関数は、与えられたランドマークの順序に従って、Dijkstra法を用いて
    最短経路を計算し、その経路の長さ、移動時間、ノードの数、しきい値を返します。

    パラメータ:
    G (Graph): ネットワークグラフ（例えば、道路ネットワーク）。
    landmark_dtf (DataFrame): ランドマークの情報を含むDataFrame。各ランドマークのノードIDが含まれます。
    pop_length (int): 経路に含めるランドマークの数。
    route_order (list): 経路に含めるランドマークの順序を示すインデックスのリスト。
    end_node (int): 経路の終点ノードID。
    alpha (float, optional): ベータ分布のパラメータ。デフォルトは2。
    beta (float, optional): ベータ分布のパラメータ。デフォルトは5。
    threshold (float, optional): しきい値。デフォルトは0（ランダムに生成されます）。

    戻り値:
    tuple: 経路に関する情報を含むタプル。内容は以下の通りです。
        - node_num (int): 経路に含まれるノードの数。
        - route (list): 経路を構成するノードIDのリスト。
        - route_order (list): 使用したランドマークの順序。
        - length (float): 経路の総距離。
        - time (float): 経路の移動時間（分単位）。
        - threshold (float): 使用されたしきい値。
    """

    # 計算された経路を格納するリスト
    route = []

    # 各ランドマーク間の経路を計算
    for index in range(pop_length + 1):
        # 現在のランドマークと次のランドマークのノードIDを取得
        source_node = landmark_dtf.loc[route_order[index], 'osmid']
        target_node = landmark_dtf.loc[route_order[index + 1], 'osmid']

        # sourceからtargetへの最短経路をDijkstra法で計算
        path = nx.dijkstra_path(G, source=source_node, target=target_node)

        # 計算された経路の各ノードをrouteリストに追加
        for n in range(len(path) - 1):
            route.append(path[n])

    # 最後に終点ノードを経路に追加
    route.append(end_node)

    # 計算された経路をGeoDataFrameに変換
    route_gdf = ox.routing.route_to_gdf(G, route)

    # 経路の長さを計算
    length = route_gdf['length'].sum()

    # 経路の移動時間を計算（分単位）
    time = route_gdf['travel_time'].sum() / 60

    # 経路に含まれるノードの数を取得
    node_num = len(route)

    # しきい値が指定されていない場合、ベータ分布に基づいてランダムに生成
    if threshold == 0:
        threshold = np.random.beta(alpha, beta, size=1)[0]

    # 経路に関する情報をタプルで返す
    return (node_num, route, route_order, length, time, threshold)


def generator(G, landmark_dtf, pop_num, pop_length, end_node, alpha, beta):
    """
    ランダムに生成された経路を返す関数。

    この関数は、指定された数の経路をランダムに生成し、それぞれの経路に関する情報を
    データフレーム形式で返します。経路は、Dijkstra法を使用して各ランドマーク間の
    最短経路を計算します。

    Args:
        G (networkx.Graph): グラフオブジェクト（道路ネットワークなど）。
        landmark_dtf (DataFrame): ランドマークの情報を含むデータフレーム。各ランドマーク間の距離やコストを含みます。
        pop_num (int): 生成する経路の数。
        pop_length (int): 各経路に含めるランドマークの数。
        end_node (int): 経路の終点ノードID。
        alpha (float): ベータ分布のパラメータ。経路のしきい値をランダムに決定する際に使用。
        beta (float): ベータ分布のパラメータ。経路のしきい値をランダムに決定する際に使用。

    Returns:
        DataFrame: 生成された経路の情報を含むデータフレーム。
                   カラムには、ノード数、経路ノード、経路順序、経路長、移動時間、しきい値が含まれます。
    """

    # 生成された経路の情報を格納するリスト
    pop_list = []

    # ランドマークのインデックスをリストで作成 (0からlandmark_dtfのサイズ-2まで)
    landmark_index = [i for i in range(len(landmark_dtf) - 2)]

    for i in range(pop_num):
        pop_id = i + 1  # 経路のIDを設定

        # ランドマークの順序をランダムに並び替え
        route_order = rnd.sample(landmark_index, pop_length)

        # ランドマークインデックスを1から始まる番号に変換
        route_order = [i + 1 for i in route_order]

        # 経路順序をソートして、スタート地点とゴール地点の順序を維持
        route_order.sort()

        # スタート地点（0）を追加
        route_order.insert(0, 0)

        # ゴール地点（landmark_dtfのサイズ - 1）を追加
        route_order.append(len(landmark_dtf) - 1)

        # 経路の情報を取得
        pop = pop_info(G, landmark_dtf, pop_length, route_order, end_node, alpha, beta)

        # 経路の情報をリストに追加
        pop_list.append(pop)

    # 生成された経路の情報をデータフレームに変換
    pop_dtf = pd.DataFrame(pop_list, columns=['node_num', 'route_node', 'route_order', 'length', 'travel_time', 'threshold'])

    return pop_dtf


# カスタム変換関数
def str_to_list(val):
    return ast.literal_eval(val)


if __name__ == "__main__":
    city = sys.argv[1]  # 対象都市名：hachioji, yokohama
    frmt = int(sys.argv[2])  # 出発点・到着点の緯度経度が書かれたファイル：from-to_01.txt
    start, end = [tuple(eval(row.rstrip())) for row in open(f"data/{city}/from-to_{frmt:02d}.txt")]  # 出発点と到着点の緯度経度を設定
    pop_num = int(sys.argv[3])  # 生成したい個体数（100, 200）
    landmark_num = int(sys.argv[4])-2  # ランドマーク数（100）
    pop_length = int(sys.argv[5])-2  # 染色体の長さ（10, 20, 30, 40, 50）
    lbd1 = float(sys.argv[6])  # 距離減衰係数lbd1（1.0）
    alpha, beta = 2, 5  # beta関数のパラメータ

    geo_data_fp = f"data/{city}/road_network.graphml"
    G = ox.load_graphml(geo_data_fp)
    landmark_dtf_fp = f"data/{city}/landmark_dtf_frmt{frmt:02d}_L{landmark_num:03d}_{int(lbd1*100):03d}.csv"    
    landmark_dtf = pd.read_csv(landmark_dtf_fp, index_col=0, converters={'spot_distance': str_to_list})
    # 指定された開始位置（start）に最も近いノードを探す
    start_node = ox.distance.nearest_nodes(G, start[1], start[0])
    # 終了位置（end）に最も近いノードを探す
    end_node = ox.distance.nearest_nodes(G, end[1], end[0])

    pop_dtf = generator(G, landmark_dtf, pop_num, pop_length, end_node, alpha, beta)
    pop_dtf.to_csv(f"data/{city}/pop_dtf_frmt{frmt:02d}_I{pop_num:03d}_L{landmark_num:03d}_ELL{pop_length:02d}_{int(lbd1*100):03d}.csv")
