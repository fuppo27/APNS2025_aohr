import sys
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from geopy.distance import geodesic  # 緯度経度から直線距離を計算するためのツール


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


def make_virus_nodes(G, spot_dtf, virus_length, viral_distance, lbd):
    """
    ターゲットノードを中心に、指定距離内のノードからウイルスのように拡散する
    ランドマークを選定し、その情報を更新する関数。

    Args:
        G (networkx.Graph): ネットワークグラフオブジェクト
        spot_dtf (pandas.DataFrame): 観光スポットのデータフレーム。`nearest_node`列が必要。
        virus_length (int): ランドマークとして選定するノードの数。
        viral_distance (float): 探索範囲の最大距離（キロメートル）。
        lbd (float): 距離に基づく選択のスケールパラメータ。

    Returns:
        list: 選定されたウイルスノードのリスト。各リスト内には選定されたノードのIDが含まれる。
    """

    virus_nodes = []  # 選定されたウイルスノードを格納するリスト
    spot_nodes = spot_dtf["nearest_node"].tolist()  # 観光スポットノードのリスト
    check_node_set = set()

    for node in spot_nodes:
        # ターゲットノードを中心に指定距離内のノードとその距離を取得
        nearby_nodes = nx.single_source_dijkstra_path_length(G, node, cutoff=viral_distance, weight='length')
        if len(nearby_nodes) <= 4:
            spot_dtf = spot_dtf.drop(spot_dtf.index[spot_dtf["nearest_node"] == node])
            continue
        del nearby_nodes[node]  # 自身のノードをリストから削除

        # ノードの距離とIDをリストとして抽出
        distances = np.array(list(nearby_nodes.values()))  # 距離のリスト
        node_ids = list(nearby_nodes.keys())  # ノードIDのリスト

        # 距離に基づいて確率を計算（距離が短いほど選ばれやすい）
        probabilities = np.exp(-lbd * (distances / 1000))  # 距離の単位をキロメートルに変換
        probabilities /= probabilities.sum()  # 確率の合計を1に正規化

        while True:
            # 指定数のノードを確率に基づいてランダムに選択
            selected_nodes = np.random.choice(node_ids, size=virus_length-1, p=probabilities, replace=False).tolist()
            include_node = set(selected_nodes) & check_node_set
            if len(include_node) == 0:
                break
        # ウイルスノードの追加
        check_node_set.add(selected_nodes)
        # 選ばれたノードをウイルスノードリストに追加
        selected_nodes += node
        virus_nodes.append(selected_nodes)
    
    return virus_nodes, spot_dtf


if __name__ == "__main__":
    city = sys.argv[1]  # 対象都市名：hachioji, yokohama
    pref = sys.argv[2]  # 都道府県名
    query = f"{city}shi, {pref}, Japan"  # "Hachiojishi, Tokyo, Japan"
    virus = sys.argv[3]  # ウイルス種類名：bank  
    virus_length = 5  # ウイルス感染によって追加されるランドマークの数
    viral_distance = 500  # ウイルス感染の探索距離
    lbd2 = 1.0  # float(sys.argv[4])  # スポットからの距離に応じてウイルスノードを選択する際に使用する減衰パラメータ
    
    G = ox.load_graphml(filepath=f"data/{city}/road_network.graphml")
    nodes = pd.read_csv(f"data/{city}/nodes.csv", index_col=0)

    tags = {"amenity": virus}  # POIのカテゴリーを指定（観光スポット: "tourism", コンビニ: "shop=convenience"）
    pois = ox.features_from_place(query, tags)  # POIデータを取得
    spot_dtf = pois[pois.geometry.type == 'Point'].copy()  # Pointジオメトリのみを選択
    spot_dtf.loc[:, 'longitude'] = spot_dtf.geometry.x
    spot_dtf.loc[:, 'latitude'] = spot_dtf.geometry.y
    spot_dtf = spot_dtf[['name', 'amenity', 'longitude', 'latitude']]
    spot_dtf["nearest_node"] = ox.distance.nearest_nodes(G, spot_dtf["longitude"], spot_dtf["latitude"])
    spot_dtf = spot_dtf.dropna(subset=['name'])

    # spot_dtfとnodesをmergeする
    merged_dtf = spot_dtf.merge(nodes, left_on="nearest_node", right_on="osmid", suffixes=('', '_nodes'), how='left')
    # mergeの結果を用いて、spot_dtfのstart_distanceカラムを更新
    spot_dtf = merged_dtf[["osmid", "name", "amenity", "longitude", "latitude", "nearest_node"]].copy()
    virus_nodes, spot_dtf = make_virus_nodes(G, spot_dtf, virus_length, viral_distance, lbd2)
    spot_dtf["virus_nodes"] = virus_nodes
    spot_dtf = spot_dtf.reset_index(drop=True)
    spot_dtf.to_csv(f"data/{city}/{virus}_spot_dtf.csv")

    nodes = spot_distance(nodes, spot_dtf)
    nodes = nodes[["osmid", "y", "x", "spot_distance"]]
    nodes.to_csv(f"data/{city}/nodes_{virus}.csv")
