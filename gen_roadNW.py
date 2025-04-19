import sys
import osmnx as ox
import networkx as nx


city = sys.argv[1]  # 対象都市名：hachioji, yokohama
pref = sys.argv[2]  # 都道府県名：tokyo, kanagawa
query = f"{city.capitalize()}, {pref.capitalize()}, Japan"  # "Hachioji, Tokyo, Japan"
G = ox.graph_from_place(query, network_type = "drive")  #'drive', 'bike', 'walk'
#start = [35.63292692012474, 139.88037284233025]  # 出発点の緯度経度
#G = ox.graph_from_point(start, network_type = "walk", dist = 1200) #'drive', 'bike', 'walk'

# グラフの各エッジ（道路）に速度を追加
G = ox.add_edge_speeds(G)
# グラフの各エッジに旅行時間を追加
G = ox.add_edge_travel_times(G)
# 最大の連結成分を抽出
largest_component = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(largest_component).copy()
ox.save_graphml(G, filepath=f"data/{city}/road_network.graphml")

nodes = ox.graph_to_gdfs(G, nodes=True, edges=False).reset_index()
nodes = nodes[["osmid", "y", "x"]]  # 必要なカラムだけをフィルタリングして取得
nodes.to_csv(f"data/{city}/nodes.csv")
