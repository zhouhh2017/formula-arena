from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
验证 VI：用正确的模拟器验证标度律
==========================================

第五轮发现的问题：
  我的验证脚本用了错误的 8-fold 对称性展开
  导致 z, g, p_c 全部算错！

原始蒙特卡洛模拟器 (monte_carlo_simulator.py) 用的是
显式 zone 定义（正确的）：
  zone 1: (±1, 0), (0, ±1) → z=4
  zone 2: (±1, ±1) → z=4
  zone 3: (±2, 0), (0, ±2) → z=4
  zone 4: (±2, ±1), (±1, ±2) → z=8
  zone 5: (±2, ±2) → z=4

本轮目标：
  1. 用正确的模拟器验证 square, triangular 的 p_c
  2. 重现 MC_sq+[1,2] 的 p_c
  3. 确认原始标度律数据是正确的
  4. 给出最终诚实评估
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============ 正确的 MC 模拟器 ============

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)
        self.largest = 1

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1
        if self.size[rx] > self.largest: self.largest = self.size[rx]

# ============ 正确的 zone 定义 ============

zone_offsets = {
    1: [(1,0), (-1,0), (0,1), (0,-1)],
    2: [(1,1), (1,-1), (-1,1), (-1,-1)],
    3: [(2,0), (-2,0), (0,2), (0,-2)],
    4: [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)],
    5: [(2,2), (2,-2), (-2,2), (-2,-2)],
}

def build_neighbors_2d(L, zones):
    """用正确的 zone 定义构建 2D 邻居表"""
    N = L * L
    neighbors = [[] for _ in range(N)]
    offsets = []
    for z in zones:
        offsets.extend(zone_offsets[z])

    for x in range(L):
        for y in range(L):
            idx = x * L + y
            for dx, dy in offsets:
                nx, ny = (x + dx) % L, (y + dy) % L
                nidx = nx * L + ny
                if nidx != idx and nidx not in neighbors[idx]:
                    neighbors[idx].append(nidx)
    return neighbors

def build_neighbors_triangular(L):
    """正确的 triangular 晶格（z=6）"""
    N = L * L
    neighbors = [[] for _ in range(N)]
    # Triangular: (1,0), (0,1), (-1,1) + reflections
    offsets = [(1,0), (-1,0), (0,1), (0,-1), (-1,1), (1,-1)]

    for x in range(L):
        for y in range(L):
            idx = x * L + y
            for dx, dy in offsets:
                nx, ny = (x + dx) % L, (y + dy) % L
                nidx = nx * L + ny
                neighbors[idx].append(nidx)
    return neighbors

def build_neighbors_3d_sc(L):
    """正确的 3D 简单立方（z=6）"""
    N = L * L * L
    neighbors = [[] for _ in range(N)]
    offsets = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    for x in range(L):
        for y in range(L):
            for z in range(L):
                idx = x * L * L + y * L + z
                for dx, dy, dz in offsets:
                    nx = (x + dx) % L
                    ny = (y + dy) % L
                    nz = (z + dz) % L
                    nidx = nx * L * L + ny * L + nz
                    neighbors[idx].append(nidx)
    return neighbors

def mc_percolation_2d(neighbors, L, n_samples=200, seed=None):
    """2D MC 渗流"""
    if seed is not None: np.random.seed(seed)
    N = L * L

    edges, edge_set = [], set()
    for i in range(N):
        for j in neighbors[i]:
            if j > i and (i,j) not in edge_set:
                edges.append((i,j))
                edge_set.add((i,j))
    n_edges = len(edges)
    edges_arr = np.array(edges)

    p_lo, p_hi = 0.0, 1.0
    for _ in range(25):
        p_mid = (p_lo + p_hi) / 2
        perc = 0
        for _ in range(n_samples):
            occupied = np.random.random(n_edges) < p_mid
            uf = UnionFind(N)
            for e in edges_arr[occupied]:
                uf.union(e[0], e[1])
            if uf.largest >= N * 0.5:
                perc += 1
        if perc / n_samples < 0.5:
            p_lo = p_mid
        else:
            p_hi = p_mid
    return (p_lo + p_hi) / 2

# ============ 1. 验证标准晶格 ============

print("="*80)
print("【1. 用正确的模拟器验证标准晶格】")
print("="*80)

np.random.seed(42)

# Square lattice
print("\nSquare lattice (z=4, 精确解 p_c=0.5927):")
for L in [20, 30, 40]:
    nb = build_neighbors_2d(L, [1])
    z_actual = len(nb[0])
    pc_L = mc_percolation_2d(nb, L, n_samples=200, seed=42)
    print(f"  L={L}: z={z_actual}, p_c={pc_L:.5f}")

# Triangular lattice
print("\nTriangular lattice (z=6, 精确解 p_c=0.5000):")
for L in [20, 30, 40]:
    nb = build_neighbors_triangular(L)
    z_actual = len(nb[0])
    pc_L = mc_percolation_2d(nb, L, n_samples=200, seed=42)
    print(f"  L={L}: z={z_actual}, p_c={pc_L:.5f}")

# sq+[1,2] (zone 1 + zone 4)
print("\nsq+[1,2] (zone 1+4, z=8, g=3, 旧 MC 结果=0.4046):")
for L in [15, 20, 30]:
    nb = build_neighbors_2d(L, [1, 4])
    z_actual = len(nb[0])
    pc_L = mc_percolation_2d(nb, L, n_samples=200, seed=42)
    print(f"  L={L}: z={z_actual}, p_c={pc_L:.5f}")

# 外推
L_sizes = [15, 20, 30]
pc_vals = []
for L in L_sizes:
    nb = build_neighbors_2d(L, [1, 4])
    pc_L = mc_percolation_2d(nb, L, n_samples=150, seed=42)
    pc_vals.append(pc_L)

L_arr = np.array(L_sizes, dtype=float)
pc_arr = np.array(pc_vals)
def ext(params):
    return np.sum((params[0] + params[1]/L_arr**2 - pc_arr)**2)
res = minimize(ext, [0.5, 0.1], method='Nelder-Mead')
print(f"  → p_c(∞) = {res.x[0]:.5f}")

# sq+[1,3] (zone 1 + zone 3)
print("\nsq+[1,3] (zone 1+3, z=8, g=4, 旧 MC 结果=0.3399):")
for L in [15, 20, 30]:
    nb = build_neighbors_2d(L, [1, 3])
    z_actual = len(nb[0])
    pc_L = mc_percolation_2d(nb, L, n_samples=200, seed=42)
    print(f"  L={L}: z={z_actual}, p_c={pc_L:.5f}")

# sq+[1,4] (zone 1 + zone 2 + zone 4)
print("\nsq+[1,4] (zone 1+2+4, z=12, g=3, 旧 MC 结果=0.2338):")
for L in [15, 20, 30]:
    nb = build_neighbors_2d(L, [1, 2, 4])
    z_actual = len(nb[0])
    pc_L = mc_percolation_2d(nb, L, n_samples=150, seed=42)
    print(f"  L={L}: z={z_actual}, p_c={pc_L:.5f}")

# ============ 2. 确认 z 和 girth ============

print("\n" + "="*80)
print("【2. 确认配位数和 girth】")
print("="*80)

configs = [
    {'name': 'sq', 'zones': [1], 'z_expected': 4, 'g_expected': 4},
    {'name': 'sq+[2]', 'zones': [1, 2], 'z_expected': 8, 'g_expected': 3},
    {'name': 'sq+[3]', 'zones': [1, 3], 'z_expected': 8, 'g_expected': 4},
    {'name': 'sq+[4]', 'zones': [1, 4], 'z_expected': 12, 'g_expected': 3},
    {'name': 'sq+[2,4]', 'zones': [1, 2, 4], 'z_expected': 16, 'g_expected': 3},
]

def compute_girth(L, offsets_list):
    """用小晶格计算 girth"""
    N = L * L
    neighbors = [[] for _ in range(N)]
    for x in range(L):
        for y in range(L):
            idx = x * L + y
            for dx, dy in offsets_list:
                nx, ny = (x + dx) % L, (y + dy) % L
                nidx = nx * L + ny
                if nidx != idx and nidx not in neighbors[idx]:
                    neighbors[idx].append(nidx)

    min_cycle = float('inf')
    for start in range(min(N, 10)):
        visited = {start: 0}
        queue = [(start, -1)]
        for node, parent in queue:
            d_n = visited[node]
            for nb in neighbors[node]:
                if nb == parent: continue
                if nb in visited:
                    min_cycle = min(min_cycle, d_n + visited[nb] + 1)
                else:
                    visited[nb] = d_n + 1
                    queue.append((nb, node))
    return min_cycle

for cfg in configs:
    offsets = []
    for z in cfg['zones']:
        offsets.extend(zone_offsets[z])
    nb = build_neighbors_2d(10, cfg['zones'])
    z_actual = len(nb[0])
    g_actual = compute_girth(20, offsets)

    print(f"\n{cfg['name']}:")
    print(f"  zones: {cfg['zones']}")
    print(f"  z: 预期={cfg['z_expected']}, 实际={z_actual} {'✓' if z_actual==cfg['z_expected'] else '⚠️'}")
    print(f"  g: 预期={cfg['g_expected']}, 实际={g_actual} {'✓' if g_actual==cfg['g_expected'] else '⚠️'}")

# ============ 3. 与原始数据对比 ============

print("\n" + "="*80)
print("【3. 与原始训练数据对比】")
print("="*80)

print("""
原始训练数据中的 MC 数据:
  MC_sq+[1,2]: zones=[1,4], z=8, g=3, p_c=0.4046
  MC_sq+[1,4]: zones=[1,2,4], z=12, g=3, p_c=0.2338
  MC_sq+[1,3]: zones=[1,3], z=8, g=4, p_c=0.3399
  MC_sq+[1,2,3]: zones=[1,2,3], z=12, g=3, p_c=0.2834

但是：
  zone 1+4 → z=4+8=12 (不是 z=8!)
  zone 1+2+4 → z=4+4+8=16 (不是 z=12!)

这意味着原始数据中的 z 标注可能有误！

或者：原始模拟使用了不同的 zone 定义？
""")

# 检查：如果 zone 4 只用了 4 个偏移
print("如果 zone 4 只用了 4 个偏移 (例如只有 (±2,±1)):")
offsets_4_half = [(2,1), (2,-1), (-2,1), (-2,-1)]
nb_test = build_neighbors_2d(10, [])
N = 100
neighbors_test = [[] for _ in range(N)]
offsets = [(1,0), (-1,0), (0,1), (0,-1)] + offsets_4_half
for x in range(10):
    for y in range(10):
        idx = x * 10 + y
        for dx, dy in offsets:
            nx, ny = (x + dx) % 10, (y + dy) % 10
            nidx = nx * 10 + ny
            if nidx != idx and nidx not in neighbors_test[idx]:
                neighbors_test[idx].append(nidx)
print(f"  zone 1 + half-zone 4: z = {len(neighbors_test[0])}")

# ============ 4. 根本问题 ============

print("\n" + "="*80)
print("【4. 根本问题诊断】")
print("="*80)

print("""
问题根源：

原始 monte_carlo_simulator.py 中 zone 4 的定义是：
  zone 4: (±2, ±1), (±1, ±2) → z=8

所以 sq+[1,4] 应该是：
  zone 1: 4 个邻居
  zone 4: 8 个邻居
  总计: z=12

但原始数据标注为 z=8！

可能原因：
  1. 原始数据中的 "sq+[1,4]" 不是指 zone 1+4
  2. 原始模拟使用了不同的 zone 定义
  3. 存在标注错误

让我检查原始脚本中的具体使用方式...
""")

# 检查原始脚本
import subprocess
result = subprocess.run(['grep', '-n', 'zone', str(ROOT / 'monte_carlo_simulator.py')],
                       capture_output=True, text=True)
print("原始脚本中的 zone 定义:")
for line in result.stdout.split('\n')[:20]:
    if line.strip():
        print(f"  {line}")

print("\n" + "="*80)
print("【5. 最终诚实评估】")
print("="*80)

print("""
经过六轮验证，我们发现：

1. 原始模拟器 (monte_carlo_simulator.py) 的 zone 定义是正确的
2. 但训练数据中的 z 标注可能与 zone 定义不一致
3. 我的验证脚本用了错误的 8-fold 展开
4. 标度律 R²=0.994 是基于可能标注有误的数据

结论：
  标度律可能是"数字巧合"而非"物理定律"

  证据支持：
  - 17 个数据点中，至少 4 个的 z 标注存疑
  - α-B 共线性 (-0.944)
  - 高维晶格系统性偏差
  - 5 个高影响点

  证据反对：
  - AIC/BIC 强支持 4 参数模型
  - 标准晶格（honeycomb, square, SC, BCC, diamond）的数据是可靠的
  - 这些标准晶格的拟合也很好

最终建议：
  1. 首先确认原始数据标注的正确性
  2. 如果有误，用正确的 z 重新拟合
  3. 无论结果如何，诚实地报告
""")

print("\n" + "="*80)
print("验证完成 — 发现严重问题，需要追溯原始数据")
print("="*80)
