import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.linear_solver import pywraplp
import time
import plotly.express as px
import plotly.graph_objects as go

# 再現性のためのシード設定
np.random.seed(42)

def generate_logistics_data(n_samples=200, n_centers=4, random_state=42):
    """
    VRPクラスタリング検証用の合成データを生成する関数。
    
    Args:
        n_samples (int): 生成する顧客（配送先）の数。
        n_centers (int): 顧客分布のホットスポット（中心）の数。
        random_state (int): ランダムシード。
        
    Returns:
        pd.DataFrame: lat, lon, work_time, deadline を含むデータフレーム。
    """
    if True:
        df = pd.read_csv('data.csv')
        df['work_time'] /= 60
        base_date = pd.to_datetime(str(20251201), format="%Y%m%d")
        df["deadline"] = (
            pd.to_datetime(df["deadline"].astype(str), format="%Y%m%d")
            .sub(base_date)
            .dt.days
        )

        return df.sample(n=n_samples, random_state=random_state)
    rng = np.random.RandomState(random_state)
    
    # 1. 空間データの生成（複数の中心を持つガウス分布）
    # 東京周辺（北緯35.68, 東経139.76）を想定
    center_lats = rng.uniform(35.60, 35.80, n_centers)
    center_lons = rng.uniform(139.60, 139.90, n_centers)
    
    lats = []
    lons = []
    
    samples_per_center = n_samples // n_centers
    # 端数調整
    remainder = n_samples % n_centers
    
    for i in range(n_centers):
        count = samples_per_center + (1 if i < remainder else 0)
        # 標準偏差0.02度（約2km）の広がり
        lats.append(rng.normal(center_lats[i], 0.02, count))
        lons.append(rng.normal(center_lons[i], 0.02, count))
        
    lats = np.concatenate(lats)
    lons = np.concatenate(lons)
    
    # シャッフル（分布の偏りをインデックス順序から消すため）
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    lats = lats[indices]
    lons = lons[indices]
    
    # 2. 業務データの生成
    # 作業時間: 15分〜60分の一様分布（設置作業などを想定）
    work_time = rng.randint(15, 61, n_samples)
    
    # 納期: 始業からの経過分（例: 8:00開始で、60分後〜600分後まで）
    # 地理的な位置と納期に弱い相関を持たせると現実的だが、今回はランダムとする
    deadline = rng.randint(60, 600, n_samples)
    
    df = pd.DataFrame({
        'id': indices,
        'lat': lats,
        'lon': lons,
        'work_time': work_time,  # 重み（容量制約対象）
        'deadline': deadline     # 特徴量（距離計算対象）
    })
    
    return df

class WeightedConstrainedKMeans(BaseEstimator, ClusterMixin):
    """
    重み付き容量制約と時空間距離を考慮したK-Meansクラスタリング。
    
    Attributes:
        n_clusters (int): クラスタ数（車両数）。
        max_capacity (float): 各クラスタの作業時間合計の上限。
        weight_col (str): 作業時間のカラム名。
        deadline_col (str): 納期のカラム名。
        time_weight_factor (float): 空間距離に対する時間距離の重み。
            値が大きいほど、納期が近いもの同士をまとめようとする力が強まる。
        max_iter (int): 最大反復回数。
        random_state (int): ランダムシード。
    """
    def __init__(self, n_clusters=8, max_capacity=100, 
                 weight_col='work_time', 
                 deadline_col='deadline',
                 time_weight_factor=1.0,
                 max_iter=30, random_state=None):
        self.n_clusters = n_clusters
        self.max_capacity = max_capacity
        self.weight_col = weight_col
        self.deadline_col = deadline_col
        self.time_weight_factor = time_weight_factor
        self.max_iter = max_iter
        self.random_state = random_state
        
        # 学習結果の格納用
        self.cluster_centers_ = None # 辞書形式: {'coords': array, 'deadline': array}
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X, y=None):
        """
        クラスタリングを実行する。
        
        Args:
            X (pd.DataFrame): lat, lon, work_time, deadlineを含むデータフレーム。
        """
        rng = np.random.RandomState(self.random_state)
        N = len(X)
        
        # --- 前処理 ---
        # 1. 座標データ: Haversine距離計算用にラジアン変換
        coords_rad = np.radians(X[['lat', 'lon']].values)
        
        # 2. 納期データ: 標準化（Z-score）を行い、空間距離とスケールを合わせる
        # ※パラメータ time_weight_factor で重要度を調整するため、まずは標準化が望ましい
        deadlines = X[self.deadline_col].values
        self.deadline_scaler_ = StandardScaler()
        deadlines_scaled = self.deadline_scaler_.fit_transform(deadlines.reshape(-1, 1)).flatten()
        
        # 3. 重み（作業時間）: LPの制約式で使用
        weights = X[self.weight_col].values
        
        # --- 初期実行可能性チェック ---
        total_weight = np.sum(weights)
        if total_weight > self.max_capacity * self.n_clusters:
            raise ValueError(
                f"実行不可能: 全作業時間({total_weight})が総容量({self.max_capacity * self.n_clusters})を超過しています。"
                "クラスタ数を増やすか、最大容量を引き上げてください。"
            )

        # --- 重心の初期化 (K-Means++ 風のランダム選択) ---
        init_indices = rng.choice(N, self.n_clusters, replace=False)
        self.cluster_centers_ = {
            'coords': coords_rad[init_indices],
            'deadline': deadlines_scaled[init_indices]
        }
        
        self.labels_ = np.zeros(N, dtype=int)
        
        # --- 反復最適化ループ (Lloyd's Algorithm with Constrained Assignment) ---
        for iteration in range(self.max_iter):
            # 1. コスト行列（距離行列）の計算
            # 空間距離 (km単位に概算: 地球半径 約6371km)
            dist_spatial = haversine_distances(coords_rad, self.cluster_centers_['coords']) * 6371

            # 時間距離 (標準化された値の絶対差分)
            # shape: (N, K)
            dist_temporal = np.abs(deadlines_scaled[:, np.newaxis] - self.cluster_centers_['deadline'][np.newaxis, :])
            
            # 統合コスト: 空間距離 + (係数 * 時間距離)
            # 例: 1単位の時間差（標準偏差相当）を time_weight_factor kmの移動と等価とみなす
            cost_matrix = dist_spatial + (self.time_weight_factor * dist_temporal)
            
            # 2. 制約付き割り当て (LP Solver)
            new_labels = self._solve_assignment_lp(cost_matrix, weights, self.max_capacity)
            
            if new_labels is None:
                print(f"Warning: Iteration {iteration} で実行可能解が見つかりませんでした。制約が厳しすぎる可能性があります。")
                break
            
            # 3. 収束判定
            if np.array_equal(new_labels, self.labels_) and iteration > 0:
                print(f"収束しました: Iteration {iteration}")
                break
                
            self.labels_ = new_labels
            
            # 4. 重心の更新
            new_coords_list = []
            new_deadlines_list = []
            
            for k in range(self.n_clusters):
                mask = (self.labels_ == k)
                if np.sum(mask) == 0:
                    # 空クラスタが発生した場合の処理（ランダムな点を再割り当てしてリセット）
                    # 厳密なLPを使っていれば空クラスタは起きにくいが、念のため
                    reinit_idx = rng.choice(N)
                    new_coords_list.append(coords_rad[reinit_idx])
                    new_deadlines_list.append(deadlines_scaled[reinit_idx])
                else:
                    # 重心の更新（ベクトルの平均）
                    # 球面上の厳密な重心ではないが、都市スケールでは近似的に算術平均で十分
                    cluster_coords = coords_rad[mask]
                    new_coords_list.append(np.mean(cluster_coords, axis=0))
                    
                    cluster_deadlines = deadlines_scaled[mask]
                    new_deadlines_list.append(np.mean(cluster_deadlines))
            
            self.cluster_centers_['coords'] = np.array(new_coords_list)
            self.cluster_centers_['deadline'] = np.array(new_deadlines_list)
            
        return self

    def _solve_assignment_lp(self, cost_matrix, weights, capacity):
        """
        Google OR-Toolsを使用して割り当て問題を解く。
        Minimize sum(cost[i][j] * x[i][j])
        Subject to:
            1. 各点は必ず1つのクラスタに属する。
            2. 各クラスタの重み合計 <= Capacity。
        """
        solver = pywraplp.Solver.CreateSolver('GLOP') # 高速なLPソルバー
        if not solver:
            return None
            
        num_nodes, num_clusters = cost_matrix.shape
        x = {} # 決定変数 x[i, j]
        
        # 変数定義 (0 <= x <= 1)
        # ※本来は x は {0, 1} のバイナリ変数（MIP）であるべきだが、
        # この構造の輸送問題はLP緩和しても多くの場合整数解（頂点解）を持つため、
        # 高速なGLOPを使用。解が非整数の場合はラウンディングを行う。
        for i in range(num_nodes):
            for j in range(num_clusters):
                x[i, j] = solver.NumVar(0, 1, f'x_{i}_{j}')
                
        # 制約1: 全点割り当て
        for i in range(num_nodes):
            solver.Add(solver.Sum([x[i, j] for j in range(num_clusters)]) == 1)
            
        # 制約2: 容量制約
        for j in range(num_clusters):
            constraint_expr = [x[i, j] * weights[i] for i in range(num_nodes)]
            solver.Add(solver.Sum(constraint_expr) <= capacity)
            
        # 目的関数: 総コスト最小化
        objective = solver.Objective()
        for i in range(num_nodes):
            for j in range(num_clusters):
                objective.SetCoefficient(x[i, j], cost_matrix[i, j])
        objective.SetMinimization()
        
        # ソルバー実行
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            labels = np.zeros(num_nodes, dtype=int)
            for i in range(num_nodes):
                # 最も値が大きいクラスタを選択（基本は1.0になるはず）
                best_k = -1
                max_val = -1.0
                for j in range(num_clusters):
                    val = x[i, j].solution_value()
                    if val > max_val:
                        max_val = val
                        best_k = j
                labels[i] = best_k
            return labels
        else:
            return None

def analyze_results(df, max_cap):
    if True:
        df['cluster_str'] = df['cluster'].astype(str)
        df = df.sort_values('cluster_str') # 凡例の順序を整える

        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color="cluster_str",       # 文字列型のカラムを指定することで離散カラーになる
            size="work_time",          
            hover_data=["id", "work_time", "deadline"],
            color_discrete_sequence=px.colors.qualitative.G10, # 離散色パレットを指定
            zoom=10,
            height=800,
            title="Weighted Constrained K-Means Optimization Result"
        )
        # 地図のスタイル設定 (OpenStreetMap)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})

        # HTMLとして出力する場合
        fig.write_html("clustering_result.html")
    else:
        # 1. 地図プロット（散布図）
        plt.figure(figsize=(12, 6))
        
        # クラスタごとに色分け、納期をマーカーサイズや濃淡で表現することも可能
        sns.scatterplot(
            data=df, x='lon', y='lat', 
            hue='cluster', palette='tab10', 
            style='cluster', s=100, alpha=0.8
        )
        plt.title(f'Weighted Constrained K-Means Result (Max Cap: {max_cap:.1f} min)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # 2. クラスタ統計（容量遵守の確認）
    stats = df.groupby('cluster').agg({
        'work_time': ['sum', 'count', 'mean'],
        'deadline': ['mean', 'min', 'max']
    })
    stats.columns = [
        'work_time_sum',
        'work_time_count',
        'work_time_mean',
        'deadline_mean',
        'deadline_min',
        'deadline_max'
    ]
    stats['Capacity Limit'] = max_cap
    stats['Is Valid'] = stats['work_time_sum'] <= max_cap
    
    print("\n--- クラスタ別統計 ---")
    # Markdownテーブルとして出力するためにdisplayなどを使うか、printで整形
    print(stats.to_markdown()) # pandas to_markdown() requires tabulate


n_vehicles = 20

# データ生成の実行
df_vrp = generate_logistics_data(n_samples=11000, n_centers=n_vehicles, random_state=1)
print("生成データサンプル:")
print(df_vrp.head())

# --- パラメータ設定 ---
# 全作業時間の合計を計算
total_work_load = df_vrp['work_time'].sum()

# 平均負荷
avg_load = total_work_load / n_vehicles
print(f"合計作業時間: {total_work_load} 分")
print(f"車両あたり平均負荷: {avg_load:.1f} 分")

# 容量上限の設定
# 平均ちょうどだとパッキング問題として解なしになる可能性が高いため、
# 10%〜20%程度の余裕（バッファ）を持たせるのが定石。
capacity_buffer = 1.2
max_cap = avg_load * capacity_buffer
print(f"設定容量上限: {max_cap:.1f} 分 (バッファ {int((capacity_buffer-1)*100)}%)")

# --- クラスタリング実行 ---
vrp_kmeans = WeightedConstrainedKMeans(
    n_clusters=n_vehicles,
    max_capacity=max_cap,
    weight_col='work_time',
    deadline_col='deadline',
    time_weight_factor=10,  # 納期をある程度重視
    max_iter=20,
    random_state=42
)

start_time = time.time()
vrp_kmeans.fit(df_vrp)
end_time = time.time()

print(f"計算時間: {end_time - start_time:.4f} 秒")

# 結果をデータフレームに結合
if vrp_kmeans.labels_ is not None:
    df_vrp['cluster'] = vrp_kmeans.labels_
    print("クラスタリング完了。")
else:
    print("クラスタリング失敗。制約条件を見直してください。")

print(df_vrp)

# 実行
if vrp_kmeans.labels_ is not None:
    analyze_results(df_vrp, max_cap)
