# APNS2025 for Aohr


## プログラム説明
* with_virus.py
  * 第1引数（city）：対象都市名（例：hachioji，yokohama）
  * 第2引数（frmt）：出発点・到着点の緯度経度が書かれたファイル番号（例：「from-to_01.txt」なら「1」）
  * 第3引数（virus）：経由スポットファイル名（例：「bank_spot_dtf.csv」なら「bank」）
  * 第4引数（pop_num）：個体数（例：100，200）
  * 第5引数（generation_num）：世代数（進化の回数）（例：50，100）
  * 第6引数（lbd）：距離減衰係数（例：0.05，0.1，0.5，1.0，5.0）
  * 第7引数（landmark_num）：ノードからランドマークとして選ぶ数（例：100，200）
  * 第8引数（pop_length）：染色体の長さ（例：10，20，30，40，50）
  * 第9引数（cluster_num）：クラスタ数（例：1，5，10）
  * 第10引数（pi）：適応度における混合比（例：[0.0, 1.0]）
  * 第11引数（virus_alpha）：免疫増加率（例：0.1，0.2，0.5，1.0，2.0，5.0）
  * 第12引数（crossover_prob）：交叉確率（例：「0.5」なら「50」）
  * 第13引数（mutation_prob）：突然変異確率（例：「0.1」なら「10」）
  * 並列BG実行例：`for alpha in 1.0 2.0 5.0 0.5 0.2 0.1; do for k in 1 5 10; do for pi in $(seq 0 0.1 1.0); do python with_virus.py hachioji 1 bank 100 50 1.0 100 20 ${k} ${pi} ${alpha} 50 10 ; done & done; done`
* without_virus.py
