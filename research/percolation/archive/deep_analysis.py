from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
深入分析：为什么第2轮失败？
==========================

关键观察：
- 目标C: z=12, d=2, g=4, n_tri=0 → p_c = 0.3399 (模拟)
- 目标D: z=12, d=2, g=3, n_tri=8 → p_c = 0.2834 (模拟)
- fcc:    z=12, d=3, g=3, n_tri=8 → p_c = 0.1188 (标准)

同样的 z=12，p_c 差 3 倍！

这意味着 (z, d, g, C) 不能唯一决定 p_c。
缺失的关键物理量是什么？
"""

import numpy as np
import sys
sys.path.insert(0, ROOT)
from monte_carlo_simulator import generate_square_lattice, UnionFind

# ============ 连接长度分析 ============

def analyze_connection_lengths(L, neighbor_zones):
    """分析不同邻居配位区的连接长度"""
    zone_distances = {
        1: np.sqrt(1),   # (±1,0), (0,±1)
        2: np.sqrt(2),   # (±1,±1)
        3: np.sqrt(4),   # (±2,0), (0,±2)
        4: np.sqrt(5),   # (±2,±1), (±1,±2)
        5: np.sqrt(8),   # (±2,±2)
    }

    N, neighbors = generate_square_lattice(L, neighbor_zones)

    distances = []
    for i in range(N):
        x_i, y_i = i // L, i % L
        for j in neighbors[i]:
            x_j, y_j = j // L, j % L
            # 处理周期性边界
            dx = min(abs(x_j - x_i), L - abs(x_j - x_i))
            dy = min(abs(y_j - y_i), L - abs(y_j - y_i))
            d = np.sqrt(dx**2 + dy**2)
            distances.append(d)

    return np.array(distances)

# ============ 分析不同晶格的连接长度 ============

lattices = [
    {'name': '基础正方 (z=4)', 'zones': [1], 'p_c_sim': 0.5927},
    {'name': '正方+第2区 (z=8)', 'zones': [1, 2], 'p_c_sim': 0.4046},
    {'name': '正方+第3区 (z=12)', 'zones': [1, 3], 'p_c_sim': 0.3399},  # 目标C
    {'name': '正方+第4区 (z=16)', 'zones': [1, 4], 'p_c_sim': 0.2338},  # 第1轮B
    {'name': '正方+第1+2+3区 (z=12)', 'zones': [1, 2, 3], 'p_c_sim': 0.2834},  # 目标D
]

print("="*80)
print("连接长度分析：缺失的物理量")
print("="*80)

print(f"\n{'晶格':<30} {'z':<4} {'<l>':<8} {'l_max':<8} {'#l>1':<6} {'p_c':<8}")
print("-"*80)

for lat in lattices:
    distances = analyze_connection_lengths(60, lat['zones'])

    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    long_conn_frac = np.mean(distances > 1.5)  # 连接长度>1.5的比例

    print(f"{lat['name']:<30} {len(lat['zones'])*4:<4} {mean_dist:<8.3f} {max_dist:<8.3f} "
          f"{long_conn_frac:<8.3f} {lat['p_c_sim']:<8.4f}")

# ============ 关键对比 ============

print("\n" + "="*80)
print("关键对比：同 z=12，不同连接结构")
print("="*80)

# 目标C: z=12, zones=[1,3]
dist_C = analyze_connection_lengths(60, [1, 3])
# 目标D: z=12, zones=[1,2,3]
dist_D = analyze_connection_lengths(60, [1, 2, 3])

print(f"\n目标C (z=12, 仅近邻+第3区):")
print(f"  连接长度: mean={np.mean(dist_C):.3f}, max={np.max(dist_C):.3f}")
print(f"  长程连接比例 (>1.5): {np.mean(dist_C > 1.5):.3f}")
print(f"  p_c = 0.3399")

print(f"\n目标D (z=12, 第1+2+3区):")
print(f"  连接长度: mean={np.mean(dist_D):.3f}, max={np.max(dist_D):.3f}")
print(f"  长程连接比例 (>1.5): {np.mean(dist_D > 1.5):.3f}")
print(f"  p_c = 0.2834")

print(f"\n差异: Δp_c = {0.3399 - 0.2834:.4f}")
print(f"  目标C 有更多长程连接 → p_c 更高！")

# ============ 物理解释 ============

print("\n" + "="*80)
print("物理解释：为什么长程连接反而提高 p_c？")
print("="*80)

print("""
传统观点：
  连接越多 → 渗流越容易 → p_c 越低

反直觉发现：
  长程连接不如短程连接有效！

解释（来自小世界网络理论）：

1. 短程连接（距离=1）：
   - 构建局部簇
   - 紧密连接，几何上"紧凑"
   - 簇容易扩展

2. 长程连接（距离>1）：
   - 跨越空间，但不构建局部簇
   - 几何上"稀疏"，连接不同区域
   - 对渗流的贡献被稀释

3. 关键机制：
   - 渗流是"局部簇合并"的过程
   - 短程连接：一个簇 → 邻近簇 → 合并
   - 长程连接：跳过中间区域，无法有效"填充"
   - 结果：长程连接多的晶格，需要更高 p 才能形成渗流簇

4. 数学表达：
   有效配位数 z_eff ≠ z
   z_eff = Σ_i w(l_i)
   其中 l_i 是第 i 个邻居的距离
   w(l) 是距离权重函数：
     - w(1) = 1（近邻）
     - w(l>1) < 1（长程）

   推测: w(l) ~ 1/l^α （某种衰减）
""")

# ============ 新物理量 ============

print("\n" + "="*80)
print("新物理量：有效配位数（加权版）")
print("="*80)

def compute_weighted_z(neighbor_zones, alpha=1.0):
    """计算加权有效配位数"""
    zone_distances = {
        1: 1.0,
        2: np.sqrt(2),
        3: 2.0,
        4: np.sqrt(5),
        5: np.sqrt(8),
    }

    z_weighted = 0
    for zone in neighbor_zones:
        if zone in zone_distances:
            dist = zone_distances[zone]
            weight = 1.0 / (dist ** alpha)
            z_weighted += 4 * weight  # 每个区4个邻居
    return z_weighted

print(f"\n{'晶格':<30} {'z':<4} {'z_eff(α=0)':<12} {'z_eff(α=0.5)':<12} {'z_eff(α=1)':<12} {'p_c':<8}")
print("-"*80)

for lat in lattices:
    z = len(lat['zones']) * 4
    z_eff_0 = compute_weighted_z(lat['zones'], alpha=0.0)
    z_eff_05 = compute_weighted_z(lat['zones'], alpha=0.5)
    z_eff_1 = compute_weighted_z(lat['zones'], alpha=1.0)

    print(f"{lat['name']:<30} {z:<4} {z_eff_0:<12.2f} {z_eff_05:<12.2f} {z_eff_1:<12.2f} {lat['p_c_sim']:<8.4f}")

# ============ 相关性分析 ============

print("\n" + "="*80)
print("相关性分析：哪个物理量与 p_c 相关最强？")
print("="*80)

from scipy.stats import pearsonr, spearmanr

p_cs = np.array([lat['p_c_sim'] for lat in lattices])
zs = np.array([len(lat['zones']) * 4 for lat in lattices])
z_effs_0 = np.array([compute_weighted_z(lat['zones'], 0) for lat in lattices])
z_effs_05 = np.array([compute_weighted_z(lat['zones'], 0.5) for lat in lattices])
z_effs_1 = np.array([compute_weighted_z(lat['zones'], 1.0) for lat in lattices])

for name, x in [('z (原始)', zs),
                 ('z_eff (α=0)', z_effs_0),
                 ('z_eff (α=0.5)', z_effs_05),
                 ('z_eff (α=1)', z_effs_1)]:
    corr, p_val = spearmanr(x, p_cs)
    print(f"  {name:<15} vs p_c: ρ = {corr:>7.3f} (p={p_val:.3f})")

print("""
解读：
  如果 z_eff 的相关性比 z 更强，说明加权是有效的
  最优 α 值给出距离衰减的定量信息
""")

print("\n" + "="*80)
print("结论")
print("="*80)
print("""
1. 当前公式失败的根本原因：
   - 只考虑 (z, d, g, C)，忽略了连接长度
   - 长程连接与短程连接对渗流的贡献不同
   - 同样 z 可以有不同的连接结构，导致不同的 p_c

2. 需要的新物理量：
   - 有效配位数（加权版）: z_eff = Σ 1/l_i^α
   - 连接长度分布
   - 平均连接距离

3. 新公式方向：
   p_c = f(z_eff, d, g, C) 而非 f(z, d, g, C)
   其中 z_eff 考虑了连接距离的衰减

4. 深层启示：
   - 渗流不是简单的"连接数"问题
   - 连接的"质量"（距离）比"数量"更重要
   - 这解释了为什么所有纯 (z, d) 公式在复杂邻域上失败
""")
