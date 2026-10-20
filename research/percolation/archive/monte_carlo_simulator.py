"""
蒙特卡洛渗流模拟
================

实现：
1. 2D 正方晶格（z=4）+ 复杂邻域
2. 3D 简单立方（z=6）
3. Union-Find 簇检测
4. 有限尺寸标度提取 p_c
5. 二分法搜索 p_c

物理量：
- z: 配位数
- d: 维度
- g: girth（最短环长度）
- n_tri: 每个顶点的三角形数
"""

import numpy as np
from typing import List, Tuple, Set
import matplotlib.pyplot as plt

# ============ Union-Find 数据结构 ============

class UnionFind:
    """并查集，用于检测连通簇"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n  # 簇大小
        self.n = n

    def find(self, x):
        """查找根节点（带路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """合并两个簇（按秩合并）"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        self.size[px] += self.size[py]

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_largest_cluster_size(self):
        """获取最大簇的大小"""
        roots = set(self.find(x) for x in range(self.n))
        return max(self.size[r] for r in roots) if roots else 0

    def get_cluster_sizes(self):
        """获取所有簇的大小"""
        from collections import Counter
        roots = [self.find(x) for x in range(self.n)]
        return Counter(roots)

# ============ 晶格生成 ============

def generate_square_lattice(L: int, neighbor_zones: List[int] = [1]) -> Tuple[int, List[List[int]]]:
    """
    生成 2D 正方晶格（带复杂邻域）

    参数：
    - L: 系统尺寸 (L x L)
    - neighbor_zones: 邻居配位区列表
      - zone 1: (±1, 0), (0, ±1) → z=4
      - zone 2: (±1, ±1) → z=4 (对角线)
      - zone 3: (±2, 0), (0, ±2) → z=4
      - zone 4: (±2, ±1), (±1, ±2) → z=8
      - zone 5: (±2, ±2) → z=4

    返回：
    - N: 总格点数
    - neighbors: 每个格点的邻居列表
    """
    N = L * L
    neighbors = [[] for _ in range(N)]

    # 定义每个配位区的偏移
    zone_offsets = {
        1: [(1,0), (-1,0), (0,1), (0,-1)],
        2: [(1,1), (1,-1), (-1,1), (-1,-1)],
        3: [(2,0), (-2,0), (0,2), (0,-2)],
        4: [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)],
        5: [(2,2), (2,-2), (-2,2), (-2,-2)],
    }

    # 收集所有偏移
    offsets = []
    for zone in neighbor_zones:
        if zone in zone_offsets:
            offsets.extend(zone_offsets[zone])

    # 构建邻居列表（周期性边界）
    for x in range(L):
        for y in range(L):
            idx = x * L + y
            for dx, dy in offsets:
                nx, ny = (x + dx) % L, (y + dy) % L
                n_idx = nx * L + ny
                neighbors[idx].append(n_idx)

    return N, neighbors

def generate_simple_cubic_lattice(L: int) -> Tuple[int, List[List[int]]]:
    """
    生成 3D 简单立方晶格（z=6）

    参数：
    - L: 系统尺寸 (L x L x L)

    返回：
    - N: 总格点数
    - neighbors: 每个格点的邻居列表
    """
    N = L * L * L
    neighbors = [[] for _ in range(N)]

    # 6个近邻
    offsets = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    for x in range(L):
        for y in range(L):
            for z in range(L):
                idx = x * L * L + y * L + z
                for dx, dy, dz in offsets:
                    nx, ny, nz = (x + dx) % L, (y + dy) % L, (z + dz) % L
                    n_idx = nx * L * L + ny * L + nz
                    neighbors[idx].append(n_idx)

    return N, neighbors

# ============ 渗流模拟 ============

def percolation_simulation(N: int, neighbors: List[List[int]], p: float, n_samples: int = 100) -> float:
    """
    单次渗流模拟

    参数：
    - N: 总格点数
    - neighbors: 邻居列表
    - p: 占据概率
    - n_samples: 样本数

    返回：
    - P_inf: 渗流概率（最大簇占比）
    """
    P_inf_total = 0.0

    for _ in range(n_samples):
        # 随机占据
        occupied = np.random.random(N) < p

        # 初始化 Union-Find
        uf = UnionFind(N)

        # 连接占据的邻居
        for i in range(N):
            if occupied[i]:
                for j in neighbors[i]:
                    if occupied[j] and i < j:  # 避免重复
                        uf.union(i, j)

        # 计算最大簇
        max_cluster = uf.get_largest_cluster_size()
        n_occupied = np.sum(occupied)

        if n_occupied > 0:
            P_inf_total += max_cluster / n_occupied

    return P_inf_total / n_samples

def find_percolation_threshold(N: int, neighbors: List[List[int]],
                                p_range: Tuple[float, float] = (0.1, 0.9),
                                tolerance: float = 0.01,
                                n_samples: int = 100,
                                P_inf_threshold: float = 0.5) -> float:
    """
    用二分法搜索 p_c

    参数：
    - N: 总格点数
    - neighbors: 邻居列表
    - p_range: p 的搜索范围
    - tolerance: 收敛精度
    - n_samples: 每个 p 的样本数
    - P_inf_threshold: 渗流判据（最大簇占比）

    返回：
    - p_c: 渗流阈值
    """
    p_low, p_high = p_range

    while p_high - p_low > tolerance:
        p_mid = (p_low + p_high) / 2
        P_inf = percolation_simulation(N, neighbors, p_mid, n_samples)

        if P_inf < P_inf_threshold:
            p_low = p_mid
        else:
            p_high = p_mid

    return (p_low + p_high) / 2

# ============ 有限尺寸标度 ============

def finite_size_scaling(lattice_type: str, L_values: List[int],
                         neighbor_zones: List[int] = [1],
                         p_guess: float = 0.5,
                         n_samples: int = 100) -> Tuple[float, float]:
    """
    用有限尺寸标度提取 p_c

    参数：
    - lattice_type: '2d_square' 或 '3d_cubic'
    - L_values: 系统尺寸列表
    - neighbor_zones: 邻居配位区（2D）
    - p_guess: p_c 的初始猜测
    - n_samples: 样本数

    返回：
    - p_c: 外推到 L→∞ 的渗流阈值
    - error: 估计误差
    """
    p_c_values = []

    for L in L_values:
        print(f"  L = {L}...", end=' ')

        if lattice_type == '2d_square':
            N, neighbors = generate_square_lattice(L, neighbor_zones)
        elif lattice_type == '3d_cubic':
            N, neighbors = generate_simple_cubic_lattice(L)
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")

        # 用二分法找 p_c
        p_c = find_percolation_threshold(N, neighbors,
                                          p_range=(max(0.01, p_guess - 0.3), min(0.99, p_guess + 0.3)),
                                          tolerance=0.005,
                                          n_samples=n_samples)

        p_c_values.append(p_c)
        print(f"p_c = {p_c:.4f}")

    # 外推到 L→∞
    # 假设 p_c(L) = p_c(∞) + a / L^b
    # 简单线性外推：p_c(∞) ≈ p_c(L_max) - (p_c(L_max) - p_c(L_min)) / (L_max - L_min)
    if len(p_c_values) >= 2:
        L_arr = np.array(L_values)
        p_c_arr = np.array(p_c_values)

        # 线性拟合 p_c vs 1/L
        from scipy.optimize import curve_fit

        def scaling_law(L, p_inf, a):
            return p_inf + a / L

        try:
            popt, pcov = curve_fit(scaling_law, L_arr, p_c_arr, p0=(p_guess, 0.1))
            p_c_inf = popt[0]
            error = np.sqrt(pcov[0, 0])
        except:
            # 如果拟合失败，用最后两个点外推
            p_c_inf = p_c_values[-1]
            error = abs(p_c_values[-1] - p_c_values[-2])
    else:
        p_c_inf = p_c_values[0] if p_c_values else 0.0
        error = 0.0

    return p_c_inf, error

# ============ 主程序 ============

def main():
    print("="*80)
    print("蒙特卡洛渗流模拟")
    print("="*80)

    # 测试：2D 正方晶格 (z=4, d=2, g=4)
    # 已知 p_c ≈ 0.5927

    print("\n【测试】2D 正方晶格 (z=4)")
    print("已知 p_c ≈ 0.592746")

    L_values = [20, 30, 40, 50]
    p_c_sim, error = finite_size_scaling('2d_square', L_values, neighbor_zones=[1],
                                          p_guess=0.59, n_samples=50)

    print(f"\n模拟结果: p_c = {p_c_sim:.4f} ± {error:.4f}")
    print(f"理论值:   p_c = 0.5927")
    print(f"相对误差: {abs(p_c_sim - 0.5927) / 0.5927:.2%}")

    # 验证公式预测
    print("\n【验证】用基础幂律公式预测")
    z, d = 4, 2
    A, alpha, beta = 3.96, 0.90, 0.83
    p_c_formula = A / (z**alpha * d**beta)
    print(f"公式预测: p_c = {p_c_formula:.4f}")
    print(f"模拟结果: p_c = {p_c_sim:.4f}")
    print(f"预测误差: {abs(p_c_formula - p_c_sim):.4f}")

    print("\n" + "="*80)
    print("下一步：预测未知晶格")
    print("="*80)
    print("脚本: verify_predictions.py")

if __name__ == '__main__':
    main()
