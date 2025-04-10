# fsm追加
import os
import sys

# 標準ライブラリのインポート
import random as rnd  # 乱数を生成するためのライブラリ
import time

# サードパーティライブラリのインポート
import pandas as pd  # データ操作と解析のためのライブラリ
import numpy as np  # 数値計算のためのライブラリ
import networkx as nx  # グラフ理論とネットワークのためのライブラリ
import collections
from geopy.distance import geodesic  # 緯度経度から直線距離を計算するためのツール

# 特定のモジュールのインポート
import osmnx as ox  # OpenStreetMapデータのダウンロードと解析のためのライブラリ
from tslearn.clustering import TimeSeriesKMeans  # 時系列データのクラスタリングのためのツール
from tslearn.utils import to_time_series_dataset  # データを時系列形式に変換するためのツール
import ast  # 文字列をPythonのオブジェクトに変換するためのライブラリ。文字列として表現されたデータ構造を評価するのに使用。

# TextFileWriterクラスの定義
class TextFileWriter:
    def __init__(self, filename):
        # ログファイルのパスを初期化
        self.filename = filename

    def write(self, text):
        # ファイルを開いてテキストを書き込む（上書きモード）
        with open(self.filename, 'w', encoding='utf-8') as file:
            file.write(text)

    def append(self, text):
        # ファイルを開いてテキストを追記する（追記モード）
        with open(self.filename, 'a', encoding='utf-8') as file:
            file.write(text)


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


def clustering(pop_dtf, nodes, cluster_num):
    """
    経路のクラスタリングを行う関数。

    この関数は、生成された複数の経路をDTW（動的時間伸縮法）を用いてクラスタリングし、
    各経路がどのクラスターに属するかを決定します。

    Args:
        pop_dtf (DataFrame): 経路情報を含むデータフレーム。各経路のノード情報が含まれます。
        nodes (DataFrame): 各ノードの座標情報を含むデータフレーム。`osmid`, `y`, `x`列が必要です。
        cluster_num (int): クラスタリングで生成するクラスターの数。

    Returns:
        DataFrame: クラスタリング結果を含むデータフレーム。`cluster`列が追加され、各経路がどのクラスターに属するかを示します。
    """

    dtw_list = []  # 各経路のノード情報を格納するリスト
    points = []  # 各経路の座標を格納するリスト

    # 各経路のノードIDをリスト化
    route_node = pop_dtf["route_node"].tolist()

    # 各経路ごとにノードの座標情報を取得し、データフレームに格納
    for id, route in enumerate(route_node):
        nodes_list = []

        # 経路に含まれる各 'osmid' のノード情報を取得し、リストに追加
        for osmid in route:
            node = nodes[nodes["osmid"] == osmid]
            nodes_list.append(node)

        # リストのデータフレームを結合して一つのデータフレームにする
        nodes_dtf = pd.concat(nodes_list)
        nodes_dtf = nodes_dtf[["osmid", "y", "x"]]

        # 経路ごとにIDを付与
        nodes_dtf = nodes_dtf.reset_index(drop=True)
        nodes_dtf['id'] = id

        # 結果をリストに追加
        dtw_list.append(nodes_dtf)

    # 全ての経路のデータフレームを結合
    dtw_dtf = pd.concat(dtw_list)

    # 各経路の座標データを抽出し、タイムシリーズデータセットに変換
    for num in dtw_dtf["id"].unique():
        points.append(np.array(dtw_dtf[dtw_dtf["id"] == num][["y", "x"]]))

    # タイムシリーズデータセットに変換
    points = to_time_series_dataset(points)

    # DBA-KMeansクラスタリングを実行
    dba_km = TimeSeriesKMeans(n_clusters=cluster_num,
                              n_init=5,
                              metric="dtw",
                              verbose=True,
                              max_iter_barycenter=10,
                              random_state=22)

    # クラスタリングの結果を取得
    pred = dba_km.fit_predict(points)

    # クラスタリング結果を元のデータフレームに追加
    pop_dtf['cluster'] = pred

    return pop_dtf


def evaluate(pop_dtf, spot_dtf, shortest_length, evaluation_value_raito, weight, fp):
    """
    経路の評価を行う関数。

    各経路について、距離の比率と観光スポットを含むかどうかを評価し、
    評価値を計算します。また、最も優れた経路を視覚化し、画像として保存します。

    Args:
        pop_dtf (DataFrame): 経路情報を含むデータフレーム。
        spot_dtf (DataFrame): 観光スポット情報を含むデータフレーム。
        shortest_length (float): 最短経路の長さ。
        evaluation_value_raito (tuple): 距離比率と観光スポットの評価の重みを示すタプル。
        weight (str): 評価に使用する距離のカラム名。
        fp (str): 画像保存時のファイルパス。

    Returns:
        DataFrame: 評価値と追加の評価結果を含む更新されたデータフレーム。
    """

    # 指定されたカラムから経路の長さをリストに変換
    pops_length = pop_dtf[weight].tolist()
    # 経路のノード情報をリストに変換
    pops_route_node = pop_dtf['route_node'].tolist()
    # 観光スポットの最寄りノードをリストに変換
    nearest_node = spot_dtf["nerest_node"].tolist()
    # 観光スポットを含むかどうかのフラグリストを初期化
    include_spot_list = []

    # 各経路の長さを最短経路の長さで割った比率を計算
    length_raito = [shortest_length / length for length in pops_length]

    # 各経路が観光スポットを含むかを判定
    for route_node in pops_route_node:
        # 経路ノードと観光スポットの最寄りノードの共通部分を取得
        include_spot = list(set(route_node) & set(nearest_node))
        # 観光スポットを含む場合は1、含まない場合は0をリストに追加
        if len(include_spot) == 0:
            include_spot_list.append(0)
        else:
            include_spot_list.append(1)
    
    # 各経路の評価値を計算
    evaluation_value = [(x * evaluation_value_raito[0]) + (y * evaluation_value_raito[1]) for (x, y) in zip(length_raito, include_spot_list)]

    # 評価結果をデータフレームに追加
    pop_dtf["length_raito"] = length_raito
    pop_dtf["include_spot"] = include_spot_list
    pop_dtf["evaluation_value"] = evaluation_value

    # 最も優れた経路のインデックスを取得
    fine_pop_index = pop_dtf["evaluation_value"].idxmax()
    fine_pop = pop_dtf.iloc[fine_pop_index]

    # 最も優れた経路のジオデータフレームを生成
    path_gdf = ox.routing.route_to_gdf(G, fine_pop['route_node'])

    # 最も優れた経路の情報を出力
    text.append(f"最も優秀な個体, ノード数：{fine_pop['node_num']}, 総距離：{round(path_gdf['length'].sum() / 1000, 2)}km\n")

    # 最も優れた経路をプロットし、画像として保存
    opts = {
        "route_color": "red",
        "route_linewidth": 5,
        "node_size": 1,
        "bgcolor": "black",
        "node_color": "white",
        "figsize": (16, 8)
    }
    ox.plot_graph_route(G, fine_pop['route_node'], show=False, save=True, filepath=fp, **opts)

    return pop_dtf


def selection(pop_dtf, cluster_num, tournament_select_num, tournament_size):
    """
    トーナメント選択法を用いて個体を選択する関数

    Args:
        pop_dtf (DataFrame): 個体群の情報を含むデータフレーム
        cluster_num (int): クラスタの数
        tournament_select_num (int): トーナメントごとに選択する個体の数
        tournament_size (int): トーナメントに参加する個体の数

    Returns:
        DataFrame: 選択された個体を含む新しい個体群のデータフレーム
    """

    # 選択された個体を格納するリスト
    tournament_pop_indices = []

    # エリート個体を格納するリスト
    elite_pop_indices = []

    for num in range(cluster_num):
        # クラスタ内の個体を取得
        cluster_pop = pop_dtf[pop_dtf['cluster'] == num]
        cluster_pop_num = len(cluster_pop)
        cluster_tournament_pop_num = 0
        text.append(f"第{num}クラスター:{cluster_pop_num}個\n")
        text.append("\n")

        # エリート個体を選択し、インデックスを追加
        elite_pop_index = cluster_pop['evaluation_value'].idxmax()
        elite_pop_indices.append(elite_pop_index)
        
        # エリート個体を削除
        cluster_pop = cluster_pop.drop(elite_pop_index)
        
        # トーナメント選択
        while cluster_tournament_pop_num < cluster_pop_num // 2:
            if len(cluster_pop) < tournament_size:
                break
            
            # トーナメントに参加する個体をランダムに選択
            tournament_indices = rnd.sample(cluster_pop.index.tolist(), tournament_size)
            tournament_pop = cluster_pop.loc[tournament_indices]

            for _ in range(tournament_select_num):
                cluster_tournament_pop_num += 1
                # トーナメント内で最も優れた個体を選択
                tournament_pop_index = tournament_pop['evaluation_value'].idxmax()
                tournament_pop_indices.append(tournament_pop_index)

                # 選択された個体を削除
                tournament_pop = tournament_pop.drop(tournament_pop_index)
                cluster_pop = cluster_pop.drop(tournament_pop_index)

    text.append(f"個数：{len(tournament_pop_indices)}, pop_id：{tournament_pop_indices}\n")
    text.append(f"エリートpop_id{elite_pop_indices}\n")

    # 選択された個体のデータフレームを作成
    tournament_pop_dtf = pop_dtf.loc[tournament_pop_indices]
    tournament_pop_dtf['selection_id'] = 0
    elite_pop_dtf = pop_dtf.loc[elite_pop_indices]
    elite_pop_dtf['selection_id'] = 1
    new_pop_dtf = pd.concat([elite_pop_dtf, tournament_pop_dtf])

    return new_pop_dtf


def crossover(pop_dtf, landmark_dtf, pop_length, landmark_num, crossover_prob):
    """
    二つの親個体を交叉させて新しい個体を生成する関数

    Args:
        pop_dtf (DataFrame): 個体群の情報を含むデータフレーム
        landmark_dtf (DataFrame): ランドマークの情報を含むデータフレーム
        pop_length (int): 経路に含めるランドマークの数
        landmark_num (int): ランドマークの総数
        crossover_prob (float): 交叉の確率

    Returns:
        list: 新しい個体の経路順序リスト
    """
    
    # 親個体をランダムに2つ選択
    cross_pop_indeices = rnd.sample(pop_dtf.index.tolist(), 2)
    cross_pop_dtf = pop_dtf.loc[cross_pop_indeices]
    cross_pop_order = cross_pop_dtf['route_order'].tolist()
    cross_pop_threshold = cross_pop_dtf['threshold'].tolist()

    # 親個体の経路順序を表示
    text.append(f"親1 長さ：{len(cross_pop_order[0])} --> {cross_pop_order[0]}\n")
    text.append(f"親2 長さ：{len(cross_pop_order[1])} --> {cross_pop_order[1]}\n")
    pop_1, pop_2 = cross_pop_order[0][1:pop_length + 1], cross_pop_order[1][1:pop_length + 1]

    # 交叉が実行されるかどうかの確率をチェック
    check_prob = rnd.randint(0, 100)
    if check_prob <= crossover_prob:
        # ランドマークのデータフレームを使用して交叉を実行
        pop_1_dtf, pop_2_dtf = landmark_dtf.loc[pop_1], landmark_dtf.loc[pop_2]
        parents_dtf = pd.concat([pop_1_dtf, pop_2_dtf])
        parents_dtf = parents_dtf.sort_values('projection_value')

        # 新しい個体の経路順序を生成
        new_pop_order_1 = []
        new_pop_order_2 = []

        duplication_list = collections.Counter(parents_dtf.index.to_list())
        duplication_list = [k for k, v in collections.Counter(duplication_list).items() if v > 1]
        for index in duplication_list:
            new_pop_order_1.append(index)
            new_pop_order_2.append(index)
        for index in parents_dtf.index:
            if index in duplication_list:
                continue

            if len(new_pop_order_1) >= pop_length:
                new_pop_order_2.append(index)
            else:
                new_pop_order_1.append(index)

        threshold_mean = np.mean(cross_pop_threshold)

        new_pop_1_dtf = landmark_dtf.iloc[new_pop_order_1]
        new_pop_1_dtf = new_pop_1_dtf.sort_values('start_distance')
        new_pop_order_1 = new_pop_1_dtf.index.tolist()

        new_pop_threshold_1 = threshold_mean

        new_pop_2_dtf = landmark_dtf.iloc[new_pop_order_2]
        new_pop_2_dtf = new_pop_2_dtf.sort_values('start_distance')
        new_pop_order_2 = new_pop_2_dtf.index.tolist()

        new_pop_threshold_2 = threshold_mean

        text.append("交叉が実行されました\n")
    else:
        # 交叉が行われなかった場合、親の経路順序をそのまま使用
        new_pop_order_1 = pop_1
        new_pop_threshold_1 = cross_pop_threshold[0]
        new_pop_order_2 = pop_2
        new_pop_threshold_2 = cross_pop_threshold[1]
        text.append("交叉が行われませんでした\n")

    # スタート地点とゴール地点を追加
    new_pop_order_1.insert(0, 0)
    new_pop_order_1.append(landmark_num + 1)
    new_pop_order_2.insert(0, 0)
    new_pop_order_2.append(landmark_num + 1)

    # 子個体の経路順序を表示
    text.append(f"子1 長さ：{len(new_pop_order_1)} --> {new_pop_order_1}\n")
    text.append(f"子1 免疫力：{new_pop_threshold_1}\n")
    text.append(f"子2 長さ：{len(new_pop_order_2)} --> {new_pop_order_2}\n")
    text.append(f"子2 長さ：{new_pop_threshold_2}\n")

    return [new_pop_order_1, new_pop_order_2], [new_pop_threshold_1, new_pop_threshold_2]


def projection(dtf, start, end):
    """
    指定された始点と終点を基に、データフレーム内の各ランドマークの射影値を計算する関数。

    Args:
        dtf (pandas.DataFrame): 各ランドマークの座標（y, x）を含むデータフレーム
        start (tuple): 射影ベクトルの始点 (x, y)
        end (tuple): 射影ベクトルの終点 (x, y)

    Returns:
        pandas.DataFrame: ランドマークの射影値を含むデータフレーム
    """

    # 始点と終点から射影ベクトルを計算
    vector = np.array([start[1] - end[1], end[0] - start[0]])

    # データフレーム内のランドマーク座標をリストに変換
    xy = dtf[["y", "x"]].values.tolist()
    xy_np = np.array(xy)  # リストをnumpy配列に変換

    # ランドマークの座標を射影ベクトルに射影し、その値を計算
    projection_value = np.dot(xy_np, vector)

    # 射影値をデータフレームの新しい列として追加
    dtf = dtf.copy()
    dtf["projection_value"] = list(projection_value)

    # 更新されたデータフレームを返す
    return dtf


def viral_infection(landmark_dtf, spot_dtf, nodes, route_order, pop_length, virus_length, landmark_num, start, end, lbd, immunity, alpha):
    """
    指定されたランドマークにウイルスのように新しいランドマークを追加し、
    ルート順序を更新する関数。

    Args:
        landmark_dtf (pandas.DataFrame): ランドマークのデータフレーム。少なくとも `osmid` と `spot_distance` 列が必要。
        spot_dtf (pandas.DataFrame): スポットノードのデータフレーム。少なくとも `nerest_node` と `virus_nodes` 列が必要。
        nodes (pandas.DataFrame): ノードのデータフレーム。少なくとも `osmid` 列が必要。
        route_order (list): 現在の経路順序のリスト。
        pop_length (int): 経路の長さ（ランドマークの数）。
        virus_length (int): 追加するランドマークの数。
        landmark_num (int): 現在のランドマーク数。
        start (int): 経路の始点ノードID。
        end (int): 経路の終点ノードID。
        lbd (float): ウイルスランドマーク選択時のランダム性を制御するパラメータ。
        immunity (float): 免疫力（ウイルス感染のしやすさに影響）。
        alpha (float): 免疫力の調整パラメータ。

    Returns:
        tuple: 
            - list: 更新されたルート順序。
            - pandas.DataFrame: 更新されたランドマークのデータフレーム。
    """

    infection_num = 0  # ウイルス感染回数のカウンタ

    # 現在のランドマーク数を取得
    total_landmark_num = len(landmark_dtf)
    # 経路順序から最初の要素（始点）を除外し、指定された長さにトリム
    route_order = route_order[1:pop_length + 1]

    # スポットノードのインデックスをランダムに並び替え
    spot_dtf_idx = rnd.sample(spot_dtf.index.tolist(), len(spot_dtf))

    # route_order に基づいてスポットの距離をリスト化
    spot_distance_list = landmark_dtf.iloc[route_order]["spot_distance"].values.tolist()
    spot_distance_list = np.array(spot_distance_list)  # NumPy 配列に変換
    spot_distance_list = spot_distance_list.reshape(len(spot_dtf), pop_length)  # 行列に変換
    spot_distance_list = spot_distance_list.tolist()  # 再度リストに変換

    # 各スポットに対して最小距離とそのインデックスを取得し、ウイルス感染の判定
    for idx in spot_dtf_idx:
        spot_distance_min = min(spot_distance_list[idx])
        spot_distance_min_idx = spot_distance_list[idx].index(spot_distance_min)  # 最小距離のインデックス
        probabilities = np.exp(-lbd * np.array(spot_distance_list[idx]))  # 確率計算
        probabilities = probabilities[spot_distance_min_idx]

        target_spot = spot_dtf.iloc[idx]  # ターゲットスポットを取得
        target_spot_node = target_spot["nerest_node"]  # ターゲットノードID
        virus_nodes = target_spot["virus_nodes"]  # ウイルスノードリスト

        if probabilities >= immunity:
            check_node = landmark_dtf[landmark_dtf["osmid"] == int(target_spot_node)]
            if not check_node.empty:
               check_node_idx = check_node.index.values[0]
               if check_node_idx in route_order:
                    continue
            text.append("ウイルスが感染しました\n")
            infection_num += 1

            virus_order = []
            
            for node in virus_nodes:
                virus_node = landmark_dtf[landmark_dtf["osmid"] == node]
                if virus_node.empty:
                    # ノードがランドマークに存在しない場合は、新たにランドマークを追加
                    virus_nodes_dtf = nodes[nodes["osmid"] == node]
                    virus_nodes_dtf = projection(virus_nodes_dtf, start, end)  # ランドマークに投影
                    landmark_dtf = pd.concat([landmark_dtf, virus_nodes_dtf])
                    landmark_dtf = landmark_dtf.reset_index(drop=True)
                    
                    # 新しいランドマークのインデックスを追加
                    virus_order.append(total_landmark_num)
                    total_landmark_num += 1
                else:
                    virus_order.append(virus_node.index.values[0])
            text.append(f"ウイルス: {virus_order}\n")

            # 選定されたウイルスランドマークのデータフレームを取得
            virus_dtf = landmark_dtf.iloc[virus_order]

            # ターゲットスポットノードのインデックスを取得
            spot_idx = virus_dtf.index[virus_dtf["osmid"] == int(target_spot_node)].values[0]

            # 経路順序のインデックスを調整
            choose_order_idx = spot_distance_min_idx
            virus_order_idx = virus_order.index(spot_idx)
            delta = pop_length - virus_length
            
            # 経路順序を更新
            if choose_order_idx < virus_order_idx:
                choose_order_idx = virus_order_idx
            elif choose_order_idx - virus_order_idx > delta:
                choose_order_idx = delta + virus_order_idx

            route_order[choose_order_idx] = spot_idx
            for idx in virus_order:
                if idx == spot_idx:
                    virus_order_idx -= 1
                    continue
                elif idx in route_order:
                    continue
                route_order[choose_order_idx - virus_order_idx] = idx
                virus_order_idx -= 1

            # 免疫力の更新
            immunity = 1 / ((1.5 - immunity) ** -1 + np.exp(-alpha * infection_num)) - 0.5 + immunity
        else:
            continue

    # 経路順序の最初と最後に始点と終点を追加
    route_order.insert(0, 0)  # 始点
    route_order.append(landmark_num + 1)  # 終点
    text.append(f"感染回数：{infection_num}, 免疫力：{immunity}, route_order：{route_order}\n")

    return (route_order, immunity), landmark_dtf


# カスタム変換関数
def str_to_list(val):
    return ast.literal_eval(val)


if __name__ == "__main__":
    # 実験の開始時間を記録
    start_time = time.time()

    # ログファイルのファイル名やその他の設定
    experiment = 'pi'  # 実験の種類
    city = sys.argv[1]  #'hachioji'  # 実験を行う都市
    clus = int(sys.argv[2])  # クラスタ数
    #alph = float(sys.argv[3])  # 免疫増加率
    pipi = float(sys.argv[3])  # 適応度の混合比
    fp = f"k{clus:02d}_pi{int(pipi*10):02d}"  # ファイル名の一部

    # ログファイルのパスを設定し、開始ログを書き込む
    os.makedirs(f"work/{city}/{experiment}/{fp}", exist_ok=True)    
    text = TextFileWriter(f'work/{city}/{experiment}/{fp}.log')
    text.write("code start ")  # コードの開始をログに記録
    text.append(f'({time.strftime("%Y/%m/%d %H:%M:%S")})\n')  # 開始時間をログに追加

    # 実験設定の概要をログに記録
    text.append(f"fp: {fp}\n")
    text.append(f"city: {city}\n")
    text.append(f"experiment: {experiment}\n")

    # 出発点と到着点の緯度経度を設定
    start = [35.66490639001417, 139.28782643798056]  # 出発点の緯度経度
    end = [35.62620801031389, 139.3405264254352]  # 到着点の緯度経度
    text.append(f"start:{start}, end:{end}\n")  # 緯度経度情報をログに追加

    # 初期生成時のパラメータを設定
    pop_num = 100  # 個体数（初期世代に含まれる経路の数）
    generation_num = 50  # 世代数（進化の回数）
    generation_counter = 0  # 世代カウンタ

    # 距離減衰のパラメータとランドマークの数を設定
    lbd = 1.0  # 距離減衰係数
    landmark_num = 98  # 選択するランドマークの数
    pop_length = 18  # 経路に含めるランドマークの数
    alpha, beta = 2, 5  # beta関数のパラメータ

    # クラスタリングのパラメータを設定
    cluster_num = 10  # クラスタリングのクラスタ数

    # 評価値の重みを設定
    pi = 0.5
    evaluation_value_raito = (pi, 1-pi) # 距離比率と観光スポット評価の重み

    # 選択のパラメータを設定
    tournament_size = 10  # トーナメントサイズ
    tournament_select_num = 2  # トーナメント選択数
    elite_select_num = 1  # エリート選択数

    # 交叉と突然変異の確率を設定
    crossover_prob = 50  # 交叉の確率（%）
    mutation_prob = 10  # 突然変異の確率（%）

    # ウイルス感染のパラメータを設定
    viral_distance = 500  # ウイルス感染の探索距離
    virus_length = 5  # ウイルス感染によって追加されるランドマークの数
    virus_alpha = 1.0  # 免疫力の調整パラメータ

    text.append("\n初期生成時のパラメータを設定\n")
    text.append(f"pop_num = {pop_num}\n")
    text.append(f"generation_num = {generation_num}\n")

    text.append("\n距離減衰のパラメータとランドマークの数を設定\n")
    text.append(f"lbd = {lbd}\n")
    text.append(f"landmark_num = {landmark_num}\n")
    text.append(f"pop_length = {pop_length}\n")
    text.append(f"alpha, beta = {alpha, beta}\n")

    text.append("\nクラスタリングのパラメータを設定\n")
    text.append(f"cluster_num = {cluster_num}\n")

    text.append("\n評価値の重みを設定\n")
    text.append(f"evaluation_value_raito = {evaluation_value_raito}\n")

    text.append("\n選択のパラメータを設定\n")
    text.append(f"tournament_size = {tournament_size}\n")
    text.append(f"tournament_select_num = {tournament_select_num}\n")
    text.append(f"elite_select_num = {elite_select_num}\n")

    text.append("\n交叉と突然変異の確率を設定\n")
    text.append(f"crossover_prob = {crossover_prob}\n")
    text.append(f"mutation_prob = {mutation_prob}\n")

    text.append("\nウイルス感染のパラメータを設定\n")
    text.append(f"viral_distance = {viral_distance}\n")
    text.append(f"virus_length = {virus_length}\n")
    text.append(f"virus_alpha = {virus_alpha}\n")

    geo_data_fp = f"data/{city}/{city}_road_network.graphml"
    spot_dtf_fp = f"data/{city}/experiment/{city}_bank_spot_dtf.csv"
    nodes_fp = f"data/{city}/experiment/{city}_nodes.csv"
    landmark_dtf_fp = f"data/{city}/experiment/{city}_landmark_dtf.csv"
    pop_dtf_fp = f"data/{city}/experiment/{city}_pop_dtf_{pop_length}.csv"

    # 保存したデータを読み込む
    G = ox.load_graphml(geo_data_fp)
    nodes = pd.read_csv(nodes_fp, index_col=0, converters={'spot_distance': str_to_list})

    spot_dtf = pd.read_csv(spot_dtf_fp, index_col=0, converters={'virus_nodes': str_to_list})
    landmark_dtf = pd.read_csv(landmark_dtf_fp, index_col=0, converters={'spot_distance': str_to_list})
    pop_dtf = pd.read_csv(pop_dtf_fp, index_col=0, converters={'route_node': str_to_list, 'route_order': str_to_list})

    # 指定された開始位置（start）に最も近いノードを探す
    start_node = ox.distance.nearest_nodes(G, start[1], start[0])
    # 終了位置（end）に最も近いノードを探す
    end_node = ox.distance.nearest_nodes(G, end[1], end[0])

    # 最短経路の計算
    shortest_path = nx.dijkstra_path(G, source=start_node, target=end_node, weight='length')

    # 計算された経路を出力
    path_gdf = ox.routing.route_to_gdf(G, shortest_path)
    shortest_length = path_gdf['length'].sum()
    shortest_travel_time = path_gdf['travel_time'].sum()/ 60

    # 初期個体群の評価
    text.append("evaluate start\n")
    pop_dtf = clustering(pop_dtf, nodes, cluster_num)
    pop_dtf = evaluate(pop_dtf, spot_dtf, shortest_length, evaluation_value_raito, 'length', f"work/{city}/{experiment}/{fp}/g{generation_counter:03d}.png")

    spot_dtf.to_csv(f'work/{city}/{experiment}/{fp}/spot.csv')
    landmark_dtf.to_csv(f'work/{city}/{experiment}/{fp}/landmark_dtf_{generation_counter:03d}.csv')
    pop_dtf.to_csv(f'work/{city}/{experiment}/{fp}/pop_dtf_{generation_counter:03d}.csv')
    # 世代ごとの進化処理
    for loop in range(generation_num):
        start_genaration = time.time()
        generation_counter += 1
        text.append(f"\n{generation_counter:03d}世代\n")

        # 個体群の選択
        text.append("selection start\n")
        pop_dtf = selection(pop_dtf, cluster_num, tournament_select_num, tournament_size)
        tournament_pop_dtf = pop_dtf[pop_dtf['selection_id'] == 0]

        # 次世代の個体群を格納するリスト
        next_pops_order = []
        next_pops_threshold = []

        # 交叉と突然変異の処理
        text.append("crossover, mutation start\n")
        while True:
            # 交叉
            pops_order, pops_threshold = crossover(tournament_pop_dtf, landmark_dtf, pop_length, landmark_num, crossover_prob)

            # 突然変異
            check_prob = rnd.randint(0, 100)
            if check_prob <= mutation_prob:
                text.append(f"突然変異が起きました\n")
                text.append(f"突然変異前{pops_threshold[0]}\n")
                threshold = np.random.beta(alpha, beta, size = 1)
                pops_threshold[0] = threshold[0]
                text.append(f"突然変異後：{pops_threshold[0]}\n")

            check_prob = rnd.randint(0, 100)
            if check_prob <= mutation_prob:
                text.append(f"突然変異が起きました\n")
                text.append(f"突然変異前{pops_threshold[1]}\n")
                threshold = np.random.beta(alpha, beta, size = 1)
                pops_threshold[1] = threshold[0]
                text.append(f"突然変異後：{pops_threshold[1]}\n")

            next_pop, landmark_dtf = viral_infection(landmark_dtf, spot_dtf, nodes, pops_order[0], pop_length, virus_length, landmark_num, start, end, lbd, pops_threshold[0], virus_alpha)

            # 新しい個体を次世代に追加
            next_pops_order.append(next_pop[0])
            next_pops_threshold.append(next_pop[1])

            if len(next_pops_order) >= pop_num - (elite_select_num * cluster_num):
                break

            next_pop, landmark_dtf = viral_infection(landmark_dtf, spot_dtf,nodes, pops_order[1], pop_length, virus_length, landmark_num, start, end, lbd, pops_threshold[1], virus_alpha)

            # 新しい個体を次世代に追加
            next_pops_order.append(next_pop[0])
            next_pops_threshold.append(next_pop[1])

            if len(next_pops_order) >= pop_num - (elite_select_num * cluster_num):
                break


        # 次世代の個体群の情報を取得
        next_pop_list = []
        for next_order, next_threshold in zip(next_pops_order, next_pops_threshold):
            pop = pop_info(G, landmark_dtf, pop_length, next_order, end_node, threshold=next_threshold)
            next_pop_list.append(pop)

        next_pop_dtf = pd.DataFrame(next_pop_list, columns=['node_num', 'route_node', 'route_order', 'length', 'travel_time', 'threshold'])

        # エリート選択された個体群を次世代に追加
        elite_pop_dtf = pop_dtf[pop_dtf['selection_id'] == 1]
        elite_route_order = elite_pop_dtf["route_order"].values.tolist()
        new_elite_route_order = []
        for route_order in elite_route_order:
            route_order[pop_length + 1] = len(landmark_dtf)
            new_elite_route_order.append(route_order)

        elite_pop_dtf = elite_pop_dtf.copy()
        elite_pop_dtf['route_order'] = new_elite_route_order
        elite_pop_dtf = elite_pop_dtf[['node_num', 'route_node', 'route_order', 'length', 'travel_time', 'threshold']]

        # エリート個体と次世代個体を結合
        pop_dtf = pd.concat([elite_pop_dtf, next_pop_dtf])
        pop_dtf = pop_dtf.reset_index(drop=True)

        # 次世代個体群の評価
        text.append("evaluate start\n")
        pop_dtf = clustering(pop_dtf, nodes, cluster_num)
        pop_dtf = evaluate(pop_dtf, spot_dtf, shortest_length, evaluation_value_raito, 'length', f"work/{city}/{experiment}/{fp}/g{generation_counter:03d}.png")
        pop_dtf.to_csv(f'work/{city}/{experiment}/{fp}/pop_dtf_{generation_counter:03d}.csv')
        landmark_dtf.to_csv(f'work/{city}/{experiment}/{fp}/landmark_dtf_{generation_counter:03d}.csv')
        end_genertion = time.time()
        text.append(f'{generation_counter:03d}世代目でかかった時間：{end_genertion - start_genaration}\n')

    end_time = time.time()
    text.append(f'({time.strftime("%Y/%m/%d %H:%M:%S")})\n')
    text.append(f'処理時間：{end_time-start_time}\n')
