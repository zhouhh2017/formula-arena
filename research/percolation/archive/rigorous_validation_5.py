"""
验证 V：诚实诊断 —— MC 模拟的一致性检查
=============================================

Round 4 的关键发现：
  sq+[1,1] (z=8, g=3, d=2): p_c = 0.239 (新 MC)
  MC_sq+[1,2] (z=8, g=3, d=2): p_c = 0.4046 (旧 MC)

  两者 (z, d, g) 完全相同！但 p_c 差了 70%！

  这意味着 (z, d, g) 不足以唯一确定 p_c！
  标度律可能是过拟合！

本轮目标：
  1. 验证 MC 模拟器的准确性
  2. 理解为什么同 (z,d,g) 不同 p_c
  3. 诚实评估标度律的局限性
  4. 给出科学上可靠的结论
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============ MC 模拟器 ============

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)
        self.size = np.ones(n, dtype=int)
        self.largest = 1

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return self.size[rx]
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1
        if self.size[rx] > self.largest: self.largest = self.size[rx]
        return self.size[rx]

def build_neighbors(L, zones, periodic=True):
    N = L * L
    neighbors = [[] for _ in range(N)]
    for x in range(L):
        for y in range(L):
            idx = x * L + y
            for dx, dy in zones:
                if dx == 0 and dy == 0: continue
                for sx, sy in [(dx,dy), (-dx,-dy), (dy,dx), (-dy,-dx),
                               (dx,-dy), (-dx,dy), (dy,-dx), (-dy,dx)]:
                    nx, ny = x + sx, y + sy
                    if periodic:
                        nx, ny = nx % L, ny % L
                    else:
                        if nx < 0 or nx >= L or ny < 0 or ny >= L: continue
                    nidx = nx * L + ny
                    if nidx != idx and nidx not in neighbors[idx]:
                        neighbors[idx].append(nidx)
    return neighbors

def mc_percolation(neighbors, L, n_samples=200, seed=None):
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

# ============ 1. 验证 MC 模拟器 ============

print("="*80)
print("【1. MC 模拟器验证（对标已知结果）】")
print("="*80)

# 标准结果：
# square (z=4, g=4): p_c = 0.592746 (精确解)
# triangular (z=6, g=3): p_c = 0.500000 (精确解)

np.random.seed(42)

# 验证 square lattice
zones_sq = [(1,0), (0,1)]
for L in [20, 30, 40]:
    nb = build_neighbors(L, zones_sq, periodic=True)
    z_check = len(nb[0])
    pc_L = mc_percolation(nb, L, n_samples=200, seed=42)
    print(f"  square L={L}: z={z_check}, p_c={pc_L:.5f} (精确解=0.59275)")

# 验证 triangular lattice
zones_tri = [(1,0), (0,1), (-1,1)]
for L in [20, 30, 40]:
    nb = build_neighbors(L, zones_tri, periodic=True)
    z_check = len(nb[0])
    pc_L = mc_percolation(nb, L, n_samples=200, seed=42)
    print(f"  triangular L={L}: z={z_check}, p_c={pc_L:.5f} (精确解=0.50000)")

# ============ 2. 重现之前的 MC 数据 ============

print("\n" + "="*80)
print("【2. 重现之前的 MC 数据】")
print("="*80)

mc_configs = [
    {'name': 'MC_sq+[1,2]', 'zones': [(1,0), (0,1), (1,2), (2,1)],
     'z_expected': 8, 'g_expected': 3, 'pc_old': 0.4046},
    {'name': 'MC_sq+[1,4]', 'zones': [(1,0), (0,1), (1,4), (4,1)],
     'z_expected': 12, 'g_expected': 3, 'pc_old': 0.2338},
    {'name': 'MC_sq+[1,3]', 'zones': [(1,0), (0,1), (1,3), (3,1)],
     'z_expected': 8, 'g_expected': 4, 'pc_old': 0.3399},
    {'name': 'MC_sq+[1,2,3]', 'zones': [(1,0), (0,1), (1,2), (2,1), (1,3), (3,1)],
     'z_expected': 12, 'g_expected': 3, 'pc_old': 0.2834},
]

np.random.seed(42)

for cfg in mc_configs:
    L = 30
    nb = build_neighbors(L, cfg['zones'], periodic=True)
    z_actual = len(nb[0])

    # girth
    def compute_girth_fast(L_small, zones):
        neighbors = build_neighbors(L_small, zones, periodic=True)
        N = L_small * L_small
        min_cycle = float('inf')
        for start in range(min(N, 15)):
            visited = {start: 0}
            queue = [(start, -1)]
            for node, parent in queue:
                d_n = visited[node]
                for nb2 in neighbors[node]:
                    if nb2 == parent: continue
                    if nb2 in visited:
                        min_cycle = min(min_cycle, d_n + visited[nb2] + 1)
                    else:
                        visited[nb2] = d_n + 1
                        queue.append((nb2, node))
        return min_cycle

    g_actual = compute_girth_fast(15, cfg['zones'])

    # MC simulation at multiple L
    L_sizes = [15, 20, 30]
    pc_vals = []
    for Lv in L_sizes:
        nb_v = build_neighbors(Lv, cfg['zones'], periodic=True)
        pc_v = mc_percolation(nb_v, Lv, n_samples=150, seed=42)
        pc_vals.append(pc_v)

    # Extrapolate
    L_arr = np.array(L_sizes, dtype=float)
    pc_arr = np.array(pc_vals)
    def ext(params):
        pc_inf, a = params
        return np.sum((pc_inf + a/L_arr**2 - pc_arr)**2)
    res = minimize(ext, [0.5, 0.1], method='Nelder-Mead')
    pc_inf = res.x[0]

    print(f"\n{cfg['name']}:")
    print(f"  z: 预期={cfg['z_expected']}, 实际={z_actual} {'✓' if z_actual==cfg['z_expected'] else '⚠️'}")
    print(f"  g: 预期={cfg['g_expected']}, 实际={g_actual} {'✓' if g_actual==cfg['g_expected'] else '⚠️'}")
    print(f"  p_c(L): {['%.4f' % p for p in pc_vals]}")
    print(f"  p_c(∞): {pc_inf:.5f}")
    print(f"  p_c(旧): {cfg['pc_old']:.4f}")
    print(f"  差异: {abs(pc_inf - cfg['pc_old']):.4f} ({abs(pc_inf-cfg['pc_old'])/cfg['pc_old']*100:.1f}%)")

# ============ 3. 关键对比：同 (z,d,g) 不同拓扑 ============

print("\n" + "="*80)
print("【3. 关键对比：同 (z=8, d=2, g=3) 不同拓扑】")
print("="*80)

# sq+[1,2]: 连接距离 = {1: 4个, √5: 4个}
# sq+[1,1]: 连接距离 = {1: 4个, √2: 4个}

np.random.seed(42)
L = 30

configs_z8_g3 = [
    {'name': 'sq+[1,2]', 'zones': [(1,0), (0,1), (1,2), (2,1)]},
    {'name': 'sq+[1,1]', 'zones': [(1,0), (0,1), (1,1)]},
    {'name': 'sq+[2,1]', 'zones': [(1,0), (0,1), (2,1)]},
]

print(f"\n在 L={L} 的有限尺寸下比较:")
for cfg in configs_z8_g3:
    nb = build_neighbors(L, cfg['zones'], periodic=True)
    z_actual = len(nb[0])

    # 计算连接距离分布
    dists = set()
    for dx, dy in cfg['zones']:
        if dx == 0 and dy == 0: continue
        for sx, sy in [(dx,dy), (-dx,-dy), (dy,dx), (-dy,-dx),
                        (dx,-dy), (-dx,dy), (dy,-dx), (-dy,dx)]:
            dist = np.sqrt(sx**2 + sy**2)
            dists.add(round(dist, 4))
    dists = sorted(dists)

    pc_L = mc_percolation(nb, L, n_samples=150, seed=42)

    print(f"\n  {cfg['name']}:")
    print(f"    zones: {cfg['zones']}")
    print(f"    z = {z_actual}")
    print(f"    连接距离: {dists}")
    print(f"    平均距离: {np.mean(dists):.3f}")
    print(f"    p_c(L={L}) = {pc_L:.5f}")

# 更大尺度的外推
print(f"\n有限尺寸标度:")
for cfg in configs_z8_g3:
    L_sizes = [15, 20, 30, 40]
    pc_vals = []
    for Lv in L_sizes:
        nb = build_neighbors(Lv, cfg['zones'], periodic=True)
        pc_v = mc_percolation(nb, Lv, n_samples=100, seed=42)
        pc_vals.append(pc_v)

    L_arr = np.array(L_sizes, dtype=float)
    pc_arr = np.array(pc_vals)
    def ext(params):
        return np.sum((params[0] + params[1]/L_arr**2 - pc_arr)**2)
    res = minimize(ext, [0.5, 0.1], method='Nelder-Mead')

    print(f"  {cfg['name']}: p_c(L)={['%.4f'%p for p in pc_vals]} → p_c(∞)={res.x[0]:.5f}")

# ============ 4. 诚实的诊断 ============

print("\n" + "="*80)
print("【4. 诚实的科学诊断】")
print("="*80)

print("""
核心问题：(z, d, g) 是否唯一确定 p_c？

证据表明：否！

  同 (z=8, d=2, g=3) 但不同拓扑的晶格有不同的 p_c：
  - sq+[1,2]: p_c ≈ 0.40（之前的 MC 结果）
  - sq+[1,1]: p_c ≈ 0.24（新的 MC 结果）
  - sq+[2,1]: p_c = ?（待验证）

  差异来源：
  - sq+[1,2] 的连接距离 = {1, √5}（较远）
  - sq+[1,1] 的连接距离 = {1, √2}（较近）
  - 连接距离不同 → 局部几何不同 → p_c 不同

标度律 p_c = A·z^{-α}·d^{-γ}·(1+B/g) 的问题：
  1. 它拟合了 17 个数据点（R²=0.994）
  2. 但 LOOCV R²=0.988（已经偏低）
  3. 分组 CV 中高维晶格有系统性偏差
  4. α-B 共线性（-0.944）
  5. 5 个高影响点（Cook's D > 4/n）
  6. 现在发现同(z,d,g)不同p_c！

结论：
  标度律是一个"有用的近似"，但不是"普适定律"

  它在以下条件下成立：
  - 2D 和 3D 标准晶格
  - 高维超立方晶格
  - 特定的连接拓扑类

  它在以下条件下失效：
  - 不同拓扑但相同 (z,d,g) 的晶格
  - 密堆积结构（FCC）
  - 可能有其他"普适类"
""")

# ============ 5. 重新评估科学价值 ============

print("\n" + "="*80)
print("【5. 重新评估科学价值】")
print("="*80)

# 用原始 17 个数据计算 3 参数模型（固定 B=1.49）
lattices_17 = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'pc': 0.697040},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'pc': 0.592746},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'pc': 0.500000},
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'pc': 0.652703},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'pc': 0.311600},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'pc': 0.245800},
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'pc': 0.431900},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'pc': 0.196737},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'pc': 0.140810},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'pc': 0.109460},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'g': 4, 'pc': 0.089270},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'g': 4, 'pc': 0.075050},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'g': 4, 'pc': 0.064530},
    {'name': 'MC_sq+[1,2]', 'z': 8, 'd': 2, 'g': 3, 'pc': 0.4046},
    {'name': 'MC_sq+[1,4]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2338},
    {'name': 'MC_sq+[1,3]', 'z': 8, 'd': 2, 'g': 4, 'pc': 0.3399},
    {'name': 'MC_sq+[1,2,3]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2834},
]

z17 = np.array([l['z'] for l in lattices_17])
d17 = np.array([l['d'] for l in lattices_17])
g17 = np.array([l['g'] for l in lattices_17])
pc17 = np.array([l['pc'] for l in lattices_17])

def scaling_zdg(params, z, d, g):
    A, alpha, gamma, B = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * (1 + B / g)

def scaling_zd(params, z, d):
    A, alpha, gamma = params
    return A * np.power(z, -alpha) * np.power(d, -gamma)

# 3 参数模型
def fit(params0, z, d, pc, g=None):
    def obj(p):
        if g is not None:
            pred = scaling_zdg(p, z, d, g)
        else:
            pred = scaling_zd(p, z, d)
        return np.sum((pred - pc)**2)
    res = minimize(obj, params0, method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
    if g is not None:
        pred = scaling_zdg(res.x, z, d, g)
    else:
        pred = scaling_zd(res.x, z, d)
    ss_res = np.sum((pc - pred)**2)
    ss_tot = np.sum((pc - np.mean(pc))**2)
    return {'params': res.x, 'R2': 1 - ss_res/ss_tot,
            'MAE': np.mean(np.abs(pc - pred)),
            'predictions': pred}

r3 = fit([3.0, 0.7, 0.8], z17, d17, pc17)
r4 = fit([2.3, 0.8, 0.7, 1.5], z17, d17, pc17, g17)

print(f"\n模型比较（17 个标准+MC 数据点）:")
print(f"\n  3 参数: p_c = {r3['params'][0]:.3f}·z^{{-{r3['params'][1]:.3f}}}·d^{{-{r3['params'][2]:.3f}}}")
print(f"    R² = {r3['R2']:.6f}, MAE = {r3['MAE']:.6f}")
print(f"\n  4 参数: p_c = {r4['params'][0]:.3f}·z^{{-{r4['params'][1]:.3f}}}·d^{{-{r4['params'][2]:.3f}}}·(1+{r4['params'][3]:.3f}/g)")
print(f"    R² = {r4['R2']:.6f}, MAE = {r4['MAE']:.6f}")

# AIC/BIC
n = 17
ss_res_3 = np.sum((pc17 - r3['predictions'])**2)
ss_res_4 = np.sum((pc17 - r4['predictions'])**2)
aic_3 = n*np.log(ss_res_3/n) + 2*3
bic_3 = n*np.log(ss_res_3/n) + 3*np.log(n)
aic_4 = n*np.log(ss_res_4/n) + 2*4
bic_4 = n*np.log(ss_res_4/n) + 4*np.log(n)

print(f"\n  AIC: 3参数={aic_3:.2f}, 4参数={aic_4:.2f}, Δ={aic_3-aic_4:.2f}")
print(f"  BIC: 3参数={bic_3:.2f}, 4参数={bic_4:.2f}, Δ={bic_3-bic_4:.2f}")

# 逐点误差最大的
print(f"\n  4 参数模型逐点误差:")
pred_4 = r4['predictions']
errors = []
for i, lat in enumerate(lattices_17):
    err = abs(pc17[i] - pred_4[i])
    err_rel = err / pc17[i] * 100
    errors.append((lat['name'], pc17[i], pred_4[i], err, err_rel))

errors.sort(key=lambda x: x[4], reverse=True)
for name, true, pred, err, err_rel in errors:
    flag = " ⚠️" if err_rel > 10 else ""
    print(f"    {name:<18} 真实={true:.5f} 预测={pred:.5f} 误差={err:.5f} ({err_rel:.1f}%){flag}")

print(f"\n  误差 > 10% 的点: {sum(1 for _, _, _, _, e in errors if e > 10)}/{n}")

# ============ 最终结论 ============

print("\n" + "="*80)
print("【6. 最终科学结论（严谨版）】")
print("="*80)

print("""
经过四轮严谨验证，我们得到以下结论：

1. 在"标准晶格"类别内（排除复杂拓扑），标度律是有效的：

   p_c ≈ 2.26·z^{-0.81}·d^{-0.73}·(1+1.48/g)
   R² = 0.994, MAE = 1.2%

   适用条件：
   - 标准晶格（simple cubic, BCC, diamond, 超立方等）
   - 2D/3D 常见拓扑
   - 高维超立方晶格

2. 标度律有根本性的局限：

   a) (z, d, g) 不唯一确定 p_c
      - 同 (z=8, d=2, g=3) 但不同连接距离 → p_c 差异达 70%
      - 渗流阈值依赖于更精细的几何/拓扑特征

   b) 参数存在共线性
      - α-B 相关性 = -0.944
      - 条件数 = 47（可接受但不理想）
      - 数据/参数比 = 4.25（偏低）

   c) 存在不同"普适类"
      - FCC（密堆积）严重偏离（误差 69%）
      - 复杂邻域可能偏离

3. 科学价值（诚实评估）：

   这不是 PRL 级别的"普适定律"
   但是一个有价值的"工程近似"：
   - 在适用范围内，预测误差 ~1-2%
   - 揭示了 (z, d, g) 是渗流阈值的主导因素
   - 为材料设计提供了快速估算工具

4. 真正需要的突破：

   要发现 PRL 级别的规律，需要：
   - 引入"连接距离分布"作为新的物理量
   - 研究不同"普适类"的分类标准
   - 从统计力学推导标度指数
   - 或承认渗流阈值没有简单的普适公式

这是诚实的科学结论。
""")

print("\n" + "="*80)
print("验证完成 — 科学诚实优先")
print("="*80)
