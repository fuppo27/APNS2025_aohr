import sys
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from geopy.distance import geodesic  # 緯度経度から直線距離を計算するためのツール


def distance_prob(nodes, start_row, lbd):
    """
    各ノードの選択確率を計算する関数。

    この関数は、始点から各ノードへの距離と終点から各ノードへの距離を考慮して、
    各ノードが選ばれる確率を計算します。距離が短いノードほど高い確率で選ばれます。

    パラメータ:
    nodes (DataFrame): 各ノードの情報を含むDataFrame。列には始点からの距離と終点からの距離が含まれます。
    start_row (DataFrame): 始点の情報を含む1行のDataFrame。終点への距離を含む列があります。
    lbd (float): 距離に基づく確率の減衰を調整するパラメータ。

    戻り値:
    np.ndarray: 各ノードが選ばれる確率の配列。合計は1になります。
    """

    # 始点から各ノードへの距離
    distance_su = nodes["start_distance"]

    # 終点から各ノードへの距離
    distance_ue = nodes["end_distance"]

    # 始点から終点への距離
    distance_se = start_row["end_distance"].values[0]

    # 確率を計算する（距離が短いほど選ばれやすくなるようにする）
    # lbd は距離に対する感度を調整するパラメータ
    probs = np.exp(-lbd * (distance_su + distance_ue - distance_se))

    # 確率を正規化して合計が1になるようにする
    probs = probs / np.sum(probs)

    return probs


def choice_landmark(nodes, start_row, landmark_num, lbd):
    """
    指定された数のランドマークを選択し、始点と終点を含むデータフレームを返す関数。

    Args:
        nodes (DataFrame): ノードの情報を含むデータフレーム。
        start_row (Series): 始点の情報を含むシリーズ。
        landmark_num (int): 選択するランドマークの数。
        lbd (float): 距離に基づく確率の減衰係数。

    Returns:
        DataFrame: 始点、終点、および選択されたランドマークの情報を含むデータフレーム。
    """
    # 始点からの距離に基づく確率を計算
    probs = distance_prob(nodes, start_row, lbd)
    # ランダムに指定数のランドマークを選択
    choose_idx = np.random.choice(nodes.index, size=landmark_num, p=probs, replace=False)
    # 選択されたランドマークのデータフレームを作成
    landmark_dtf = nodes.iloc[choose_idx]
    # 始点からの距離でソートしてデータフレームを返す
    return landmark_dtf.sort_values('start_distance')


def distance(source, nodes, name):
    """
    与えられた供給元の座標から各ノードまでの距離を計算
    距離を新しい列としてデータフレームに追加

    Args:
        source (list or tuple): 供給元の緯度経度を示す座標
        nodes (pd.DataFrame): 各ノードの緯度経度を含むデータフレーム
        name (str): 距離を追加する列の名前に使用する接頭辞

    Returns:
        pd.DataFrame: 各ノードまでの距離を含む新しい列が追加されたデータフレーム
    """
    source = tuple(source)  # 供給元の座標をタプルに変換
    # 各ノードとの距離を計算
    for idx, node in nodes.iterrows():
        destination = (node['y'], node['x'])  # ノードの緯度経度を取得
        x = geodesic(source, destination).km  # 直線距離を取得
        nodes.at[idx, f'{name}_distance'] = x  # 距離を新しい列に追加
    
    return nodes  # 更新されたデータフレームを返す


if __name__ == "__main__":
    city = sys.argv[1]  # 対象都市名：hachioji, yokohama
    frmt = int(sys.argv[2])  # 出発点・到着点の緯度経度が書かれたファイル：from-to_01.txt
    start, end = [tuple(eval(row.rstrip())) for row in open(f"data/{city}/from-to_{frmt:02d}.txt")]  # 出発点と到着点の緯度経度を設定
    lbd1 = float(sys.argv[3])  # 距離減衰係数lbd1：0.05, 0.1, 0.5, 1.0, 5.0
    landmark_num = int(sys.argv[4])  # 選択するランドマークの数：100, 200
    
    G = ox.load_graphml(filepath=f"data/{city}/road_network.graphml")
    nodes = pd.read_csv(f"data/{city}/nodes.csv", index_col=0)
    nodes = distance(start, nodes, "start")  # 始点からの距離を計算して、nodesデータフレームに追加
    nodes = distance(end, nodes, "end")  # 終点からの距離を計算して、nodesデータフレームに追加
    nodes = nodes[["osmid", "y", "x", "start_distance", "end_distance"]]  # 必要なカラムだけをフィルタリングして取得
    nodes.to_csv(f"data/{city}/nodes_frmt{frmt:02d}.csv")

    start_node = ox.distance.nearest_nodes(G, start[1], start[0])  # 指定された開始位置（start）に最も近いノードを探す
    start_row = nodes[nodes["osmid"] == start_node]  # 始点ノードのデータ行を取得
    landmark_dtf = choice_landmark(nodes, start_row, landmark_num-2, lbd1)  # ランドマークを選択する
    landmark_dtf.to_csv(f"data/{city}/landmark_dtf_frmt{frmt:02d}_L{landmark_num:03d}_{int(lbd1*100):03d}.csv")
