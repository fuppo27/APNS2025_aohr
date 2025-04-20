# APNS2025 for Aohr


## GAプログラム説明
* gen_roadNW.py：対象地域の道路網を取得する
  * 第1引数（city）：対象都市名（例：hachioji, yokohama）
  * 第2引数（pref）：都道府県名（例：tokyo, kanagawa）
  * 出力ファイル
    * road_network.graphml
    * nodes.csv
* gen_landmark.py：始点と終点からの距離確率に基づきランドマークを選択する
  * 第1引数（city）：対象都市名
  * 第2引数（frmt）：出発点・到着点の緯度経度が書かれたファイル（例：from-to_01.txt）
  * 第3引数（lbd1）：距離減衰係数lbd1（例：0.05, 0.10, 0.50, 1.00, 5.00）
  * 第4引数（landmark_num）：選択するランドマークの数（例：100, 200）
  * 出力ファイル
    * landmark_frmt{frmt:02d}_L{landmark_num:03d}_LBD{int(lbd1*100):03d}.csv
    * nodes_frmt{frmt:02d}.csv
* gen_virus.py：スポットウイルスを選択する
  * 第1引数（city）：対象都市名
  * 第2引数（pref）：都道府県名
  * 第3引数（virus）：ウイルス種類名（例：bank）
  * 出力ファイル
    * {virus}.csv
    * nodes_{virus}.csv
* concat_landmark_virus.py：ウイルスノードの並び替え（出発地からの距離でソート）とランドマークノードからの重複除去
  * 第1引数（city）：対象都市名
  * 第2引数（frmt）：出発点・到着点の緯度経度が書かれたファイル
  * 第3引数（virus）：ウイルス種類名
  * 第4引数（lbd1）：距離減衰係数lbd1
  * 第5引数（landmark_num）：ランドマーク数
  * 出力ファイル
    * {virus}_frmt{frmt:02d}.csv
    * landmark_frmt{frmt:02d}_L{landmark_num:03d}_LBD{int(lbd1*100):03d}_{virus}.csv
    * nodes_frmt{frmt:02d}_{virus}.csv
  * BG並列実行例（5並列）：`for lbd in 0.05 0.10 0.50 1.00 5.00 ; do python concat_landmark_virus.py hachioji 1 bank $lbd 100 & done`
* init_pop.py：初期個体生成
  * 第1引数（city）：対象都市名
  * 第2引数（frmt）：出発点・到着点の緯度経度が書かれたファイル
  * 第3引数（lbd1）：距離減衰係数lbd1
  * 第4引数（landmark_num）：ランドマーク数
  * 第5引数（pop_num）：生成したい個体数（例：100, 200）
  * 第6引数（pop_length）：染色体の長さ（例：10, 20, 30, 40, 50）
  * 第7引数（virus）：ウイルス種類名
  * 出力ファイル：pop_frmt{frmt:02d}_L{landmark_num:03d}_LBD{int(lbd1*100):03d}_{virus}_I{pop_num:03d}_ELL{pop_length:02d}.csv
  * BG並列実行例（25並列）：`for lbd in 0.05 0.10 0.50 1.00 5.00; do for ell in 10 20 30 40 50; do python init_pop.py hachioji 1 $lbd 100 100 $ell bank & done; done`
* with_virus.py
  * 第1引数（city）：対象都市名
  * 第2引数（frmt）：出発点・到着点の緯度経度が書かれたファイル番号
  * 第3引数（lbd1）：距離減衰係数lbd1
  * 第4引数（landmark_num）：ランドマーク数
  * 第5引数（pop_num）：個体数
  * 第6引数（pop_length）：染色体の長さ
  * 第7引数（virus）：ウイルス種類名
  * 第8引数（generation_num）：世代数（進化の回数）（例：50，100）
  * 第9引数（cluster_num）：クラスタ数（例：1，5，10）
  * 第10引数（pi）：適応度における混合比（例：[0.0, 1.0]）
  * 第11引数（virus_alpha）：免疫増加率（例：0.1，0.2，0.5，1.0，2.0，5.0）
  * 第12引数（crossover_prob）：交叉確率（例：「0.5」なら「50」）
  * 第13引数（mutation_prob）：突然変異確率（例：「0.1」なら「10」）
  * BG並列実行例（75並列）：
  ```
  for lbd in 0.05 0.10 0.50 1.00 5.00; do 
	for ell in 10 20 30 40 50; do 
		for k in 1 5 10; do 
			for alpha in 0.5 1.0 2.0; do 
				for pi in $(seq 0 0.1 1.0); do 
					python with_virus.py hachioji 1 $lbd 100 100 $ell bank 50 ${k} ${pi} ${alpha} 50 10; 
				done; 
			done &
		done; 
	done; 
  done
  ```
* without_virus.py



## 評価
### 多様性評価：eval_diversity.ipynb
* 
### 最適性評価：eval_optimality.ipynb
### 制約充足性評価：eval_satisfiability.ipynb