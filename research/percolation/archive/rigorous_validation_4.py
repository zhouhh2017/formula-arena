"""
验证 IV：用 Monte Carlo 模拟打破共线性
==========================================

关键发现（第三轮）：
  1. 固定 B=1.49 后，3 参数模型 AIC 更优
  2. 高维晶格 d≥7 有 +5~9% 系统性偏差
  3. 5 个高影响点（Cook's D > 4/n）

目标：用 MC 模拟生成"打破共线性"的新数据
  - 需要一个 高 z 但 g≠4 的晶格
  - 需要一个 高维但 g≠4 的晶格

方案：正方晶格 + 复杂邻域
  (1) sq + [1,1] (diagonal): z=8, g=3 (三角形!)
  (2) sq + [1,2] (L-shape): z=8, g=3 or 4
  (3) sq + [2,0] (skip-1): z=8, g=4 or 6
  (4) triangular + [2,0]: z=12, g=?
  (5) honeycomb + [1,1]: z=6, g=?

计划：
  1. 构建候选晶格的几何结构
  2. MC 模拟计算 p_c（有限尺寸标度）
  3. 用新数据重新拟合
  4. 检验共线性是否被打破
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============ Monte Carlo 渗流模拟器 ============

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
        if rx == ry:
            return self.size[rx]
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        if self.size[rx] > self.largest:
            self.largest = self.size[rx]
        return self.size[rx]

def build_neighbors(L, zones, periodic=True):
    """构建邻域表"""
    N = L * L
    neighbors = [[] for _ in range(N)]

    for x in range(L):
        for y in range(L):
            idx = x * L + y
            for dx, dy in zones:
                if dx == 0 and dy == 0:
                    continue
                # 8-fold symmetry
                for sx, sy in [(dx,dy), (-dx,-dy), (dy,dx), (-dy,-dx),
                               (dx,-dy), (-dx,dy), (dy,-dx), (-dy,dx)]:
                    nx, ny = x + sx, y + sy
                    if periodic:
                        nx, ny = nx % L, ny % L
                    else:
                        if nx < 0 or nx >= L or ny < 0 or ny >= L:
                            continue
                    nidx = nx * L + ny
                    if nidx != idx and nidx not in neighbors[idx]:
                        neighbors[idx].append(nidx)

    return neighbors

def mc_percolation(neighbors, L, n_samples=200):
    """MC 模拟计算渗流阈值"""
    N = L * L

    # 收集所有边
    edges = []
    edge_set = set()
    for i in range(N):
        for j in neighbors[i]:
            if j > i:
                edge_key = (i, j)
                if edge_key not in edge_set:
                    edges.append(edge_key)
                    edge_set.add(edge_key)

    n_edges = len(edges)
    edges = np.array(edges)

    # 二分搜索 p_c
    p_lo, p_hi = 0.0, 1.0

    for _ in range(30):  # 精度 ~1e-9
        p_mid = (p_lo + p_hi) / 2

        # 多次采样
        percolating = 0
        for _ in range(n_samples):
            # 随机占据
            occupied = np.random.random(n_edges) < p_mid
            active_edges = edges[occupied]

            uf = UnionFind(N)
            for e in active_edges:
                uf.union(e[0], e[1])

            if uf.largest >= N * 0.5:  # 渗流判据
                percolating += 1

        prob = percolating / n_samples

        if prob < 0.5:
            p_lo = p_mid
        else:
            p_hi = p_mid

    return (p_lo + p_hi) / 2

def finite_size_scaling(neighbors_func, L_sizes, zones, n_samples=100):
    """有限尺寸标度提取 p_c(∞)"""
    pc_values = []

    for L in L_sizes:
        neighbors = neighbors_func(L, zones)
        pc_L = mc_percolation(neighbors, L, n_samples)
        pc_values.append(pc_L)

    # 外推 L→∞: p_c(L) = p_c(∞) + a/L^2
    L_arr = np.array(L_sizes, dtype=float)
    pc_arr = np.array(pc_values)

    def extrapolate(params):
        pc_inf, a = params
        pred = pc_inf + a / L_arr**2
        return np.sum((pred - pc_arr)**2)

    result = minimize(extrapolate, [0.5, 0.1], method='Nelder-Mead')
    pc_inf = result.x[0]

    return pc_inf, pc_values

# ============ 候选晶格 ============

print("="*80)
print("【候选晶格设计】")
print("="*80)

candidates = [
    {
        'name': 'sq+[1,1]',
        'zones': [(1,0), (0,1), (1,1)],
        'z_expected': 8,
        'g_expected': 3,  # 有三角形: 0-(1,0)-(1,1)-0
        'reason': '高 z + 低 girth（打破 z-g 共线性）'
    },
    {
        'name': 'sq+[0,2]',
        'zones': [(1,0), (0,1), (0,2)],
        'z_expected': 8,
        'g_expected': 4,
        'reason': '对照：z=8, g=4（与已有数据比较）'
    },
    {
        'name': 'sq+[1,1]+[0,2]',
        'zones': [(1,0), (0,1), (1,1), (0,2)],
        'z_expected': 12,
        'g_expected': 3,
        'reason': '高 z + 低 girth（强打破共线性）'
    },
    {
        'name': 'tri+[2,0]',
        'zones': [(1,0), (0,1), (-1,1), (2,0)],
        'z_expected': 8,
        'g_expected': 3,
        'reason': 'triangular 基 + 扩展邻域'
    },
    {
        'name': 'sq+[0,3]',
        'zones': [(1,0), (0,1), (0,3)],
        'z_expected': 8,
        'g_expected': 6,  # 最小环：0-(1,0)-(1,1)-(0,1)-(0,2)-(0,3)-...
        'reason': '高 z + 高 girth（打破共线性）'
    },
]

for c in candidates:
    print(f"\n{c['name']}:")
    print(f"  zones: {c['zones']}")
    print(f"  z = {c['z_expected']}, g = {c['g_expected']}")
    print(f"  理由: {c['reason']}")

# ============ 快速验证 girth ============

print("\n" + "="*80)
print("【验证 girth（最短环长度）】")
print("="*80)

def compute_girth(L_small, zones):
    """用小晶格计算 girth"""
    neighbors = build_neighbors(L_small, zones, periodic=True)
    N = L_small * L_small

    # BFS 找最短环
    min_cycle = float('inf')

    for start in range(min(N, 20)):  # 只检查前 20 个顶点
        # BFS
        visited = {start: 0}
        queue = [(start, -1)]  # (node, parent)
        dist = 0

        for node, parent in queue:
            d = visited[node]
            for nb in neighbors[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    cycle_len = d + visited[nb] + 1
                    min_cycle = min(min_cycle, cycle_len)
                else:
                    visited[nb] = d + 1
                    queue.append((nb, node))

    return min_cycle

for c in candidates:
    g = compute_girth(10, c['zones'])
    c['g_computed'] = g

    # 验证 z
    neighbors = build_neighbors(10, c['zones'], periodic=True)
    z_actual = len(neighbors[0])  # 中心顶点的邻居数

    print(f"\n{c['name']}:")
    print(f"  z (预期={c['z_expected']}, 实际={z_actual}) {'✓' if z_actual == c['z_expected'] else '⚠️'}")
    print(f"  g (预期={c['g_expected']}, 实际={g}) {'✓' if g == c['g_expected'] else '⚠️'}")

# ============ MC 模拟（快速版本）============

print("\n" + "="*80)
print("【Monte Carlo 模拟（快速版，L=10-40）】")
print("="*80)

np.random.seed(42)

for c in candidates:
    zones = c['zones']
    name = c['name']
    z = c.get('z_actual', c['z_expected'])

    # 验证 z
    neighbors_test = build_neighbors(10, zones)
    z_actual = len(neighbors_test[0])
    c['z_actual'] = z_actual

    # 有限尺寸标度
    L_sizes = [10, 15, 20, 30, 40]
    pc_values = []

    print(f"\n{name} (z={z_actual}, g={c.get('g_computed', '?')}):")

    for L in L_sizes:
        neighbors = build_neighbors(L, zones, periodic=True)
        pc_L = mc_percolation(neighbors, L, n_samples=100)
        pc_values.append(pc_L)
        print(f"  L={L}: p_c = {pc_L:.5f}")

    # 外推
    L_arr = np.array(L_sizes, dtype=float)
    pc_arr = np.array(pc_values)

    def extrapolate(params):
        pc_inf, a = params
        pred = pc_inf + a / L_arr**2
        return np.sum((pred - pc_arr)**2)

    result = minimize(extrapolate, [0.5, 0.1], method='Nelder-Mead',
                     options={'maxiter': 10000})
    pc_inf = result.x[0]
    c['pc_mc'] = pc_inf

    print(f"  → p_c(∞) = {pc_inf:.5f}")

# ============ 用新数据重新拟合 ============

print("\n" + "="*80)
print("【用新数据重新拟合标度律】")
print("="*80)

# 原始数据（排除 FCC）
original_data = [
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
]

# 添加 MC 模拟数据
mc_data = []
for c in candidates:
    if 'pc_mc' in c and c['z_actual'] not in [d['z'] for d in original_data if d['g'] == c.get('g_computed', 999)]:
        mc_data.append({
            'name': c['name'],
            'z': c['z_actual'],
            'd': 2,
            'g': c.get('g_computed', 4),
            'pc': c['pc_mc']
        })

# 只添加打破共线性的数据
print("\n添加的新数据:")
for d in mc_data:
    print(f"  {d['name']}: z={d['z']}, d={d['d']}, g={d['g']}, p_c={d['pc']:.5f}")

all_data = original_data + mc_data
n_all = len(all_data)

z_all = np.array([d['z'] for d in all_data])
d_all = np.array([d['d'] for d in all_data])
g_all = np.array([d['g'] for d in all_data])
pc_all = np.array([d['pc'] for d in all_data])

# ============ 重新拟合 ============

def scaling_zdg(params, z, d, g):
    A, alpha, gamma, B = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * (1 + B / g)

def robust_fit(formula, params0, z, d, pc, g=None, max_restarts=5):
    best_result, best_loss = None, np.inf
    starts = [params0]
    np.random.seed(0)
    for _ in range(max_restarts - 1):
        starts.append(params0 * (1 + np.random.uniform(-0.3, 0.3, len(params0))))
    for start in starts:
        def objective(p):
            pred = formula(p, z, d, g) if g is not None else formula(p, z, d)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)): return 1e10
            return np.sum((pred - pc)**2)
        result = minimize(objective, start, method='Nelder-Mead',
                         options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_loss:
            pred = formula(result.x, z, d, g) if g is not None else formula(result.x, z, d)
            if not np.any(np.isnan(pred)):
                best_loss = result.fun
                best_result = result
    if best_result is None: return None
    pred = formula(best_result.x, z, d, g) if g is not None else formula(best_result.x, z, d)
    if np.any(np.isnan(pred)): return None
    ss_res = np.sum((pc - pred)**2)
    ss_tot = np.sum((pc - np.mean(pc))**2)
    k = len(best_result.x)
    return {'params': best_result.x, 'R2': 1 - ss_res/ss_tot,
            'MAE': np.mean(np.abs(pc - pred)),
            'AIC': n_all*np.log(ss_res/n_all) + 2*k, 'BIC': n_all*np.log(ss_res/n_all) + k*np.log(n_all)}

print(f"\n原始数据: {len(original_data)} 个点")
print(f"新增数据: {len(mc_data)} 个点")
print(f"总计: {n_all} 个点")

# 拟合 4 参数模型
result_4p = robust_fit(scaling_zdg, [2.3, 0.8, 0.7, 1.5], z_all, d_all, pc_all, g_all)

if result_4p:
    print(f"\n4 参数模型: p_c = A·z^{{-α}}·d^{{-γ}}·(1+B/g)")
    print(f"  R² = {result_4p['R2']:.6f}, MAE = {result_4p['MAE']:.6f}")
    print(f"  A = {result_4p['params'][0]:.4f}")
    print(f"  α = {result_4p['params'][1]:.4f}")
    print(f"  γ = {result_4p['params'][2]:.4f}")
    print(f"  B = {result_4p['params'][3]:.4f}")

    # 条件数
    eps = 1e-6
    J = np.zeros((n_all, 4))
    for j in range(4):
        p_plus = result_4p['params'].copy(); p_plus[j] += eps
        p_minus = result_4p['params'].copy(); p_minus[j] -= eps
        J[:, j] = (scaling_zdg(p_plus, z_all, d_all, g_all) - scaling_zdg(p_minus, z_all, d_all, g_all)) / (2*eps)
    for j in range(4):
        J[:, j] *= result_4p['params'][j]
    S = np.linalg.svd(J, compute_uv=False)
    cond = S[0] / S[-1]
    corr = np.corrcoef(J.T)
    max_corr = np.max(np.abs(corr[np.triu_indices(4, k=1)]))

    print(f"  条件数 = {cond:.1f}")
    print(f"  最大参数相关性 = {max_corr:.3f}")

    # 对比原始结果
    print(f"\n对比原始结果（17 个点）:")
    print(f"  原始: R²=0.9937, MAE=0.0121, cond=47, max_corr=0.944")
    print(f"  新增: R²={result_4p['R2']:.4f}, MAE={result_4p['MAE']:.4f}, cond={cond:.1f}, max_corr={max_corr:.3f}")

    if max_corr < 0.9:
        print(f"\n  ✓ 共线性被打破！(max_corr {0.944:.3f} → {max_corr:.3f})")
    else:
        print(f"\n  ⚠️ 共线性仍有改善空间 (max_corr {max_corr:.3f})")

# ============ 最终报告 ============

print("\n" + "="*80)
print("【最终验证报告】")
print("="*80)

print("""
四轮验证总结:
  Round 1: R²=0.994, LOOCV R²=0.988, 但 B 的 Bootstrap std 异常
  Round 2: AIC/BIC 强支持 4 参数, 条件数=47, α-B 相关性=-0.944
  Round 3: 发现 z-g 不独立是共线性根源; 固定 B 后模型更优
  Round 4: 用 MC 模拟生成新数据打破共线性

最终评估:
""")

if result_4p:
    if max_corr < 0.9 and result_4p['R2'] > 0.99:
        print("  ★★★ PRL 级别确认（共线性已解决）")
    elif result_4p['R2'] > 0.99:
        print("  ★★☆ 预测可靠，但参数解释仍受限")
    else:
        print("  ★☆☆ 需要更多数据")
else:
    print("  拟合失败")

print("\n" + "="*80)
print("验证完成")
print("="*80)
