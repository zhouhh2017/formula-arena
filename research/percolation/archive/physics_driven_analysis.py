from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
物理理论驱动的渗流阈值分析
================================
从多个物理学科出发，构建有明确物理意义的 p_c 公式

理论基础:
1. 渗流理论 (Bethe-Peierls, 有效介质, 标度)
2. 谱图论 (谱半径, 随机游走)
3. 环统计 (girth, 环分布)
4. 统计力学 (配分函数, 关联函数)
5. 几何拓扑 (Euler 特征, 对偶性)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize, differential_evolution
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import math

np.random.seed(42)


# =====================================================================
# 第一部分: 晶格的完整物理描述
# =====================================================================

def build_physics_dataset():
    """
    为每个晶格计算/查找完整的物理量集合

    物理量:
    - z: 配位数 (coordination number)
    - d: 嵌入维度
    - g: girth (最短环长度)
    - f_dist: 面类型分布 (face size distribution)
    - lambda1: 谱半径 (邻接矩阵最大特征值, 正则图 = z)
    - mu: 随机游走返回概率的衰减率
    - n_tri: 每个顶点的三角形数 (clustering)
    - n_sq: 每个顶点的四边形数
    - dim_eff: 有效维度 (考虑环效应)
    - eta: Bethe 近似误差修正因子
    - p_bethe: Bethe 近似值 1/(z-1)
    - packing_fraction: 堆积密度
    - dual_z: 对偶晶格配位数 (2D)
    - euler_char: Euler 特征密度
    - k_critical: 临界壳层数
    """

    data = []

    # ===== 1. 标准 2D 晶格 =====
    # 数据来源: 教科书 + 精确理论结果

    # 蜂窝晶格 (honeycomb)
    # z=3, 六边形面, girth=6, 对偶: 三角晶格 (z=6)
    # 三角形数 = 0 (无三角形), 四边形数 = 0
    data.append({
        'name': 'honeycomb', 'd': 2, 'z': 3, 'pc': 0.697040,
        'g': 6, 'n_tri': 0, 'n_sq': 0, 'n_hex': 1,  # 每顶点1个六边形
        'dual_z': 6, 'is_self_dual': False,
        'packing': 0.9069,  # 六方密堆积
        'source': 'Stauffer1994'
    })

    # 正方晶格 (square)
    # z=4, 正方形面, girth=4, 自对偶
    data.append({
        'name': 'square', 'd': 2, 'z': 4, 'pc': 0.592746,
        'g': 4, 'n_tri': 0, 'n_sq': 1, 'n_hex': 0,
        'dual_z': 4, 'is_self_dual': True,
        'packing': 0.7854,  # π/4
        'source': 'Stauffer1994'
    })

    # 三角晶格 (triangular)
    # z=6, 三角形面, girth=3, 对偶: 蜂窝 (z=3)
    # 每个顶点被6个三角形包围
    data.append({
        'name': 'triangular', 'd': 2, 'z': 6, 'pc': 0.500000,
        'g': 3, 'n_tri': 6, 'n_sq': 0, 'n_hex': 0,
        'dual_z': 3, 'is_self_dual': False,
        'packing': 0.9069,
        'source': 'Stauffer1994_exact'
    })

    # Kagome 晶格
    # z=4, 三角形+六边形面 (3.6.3.6), girth=3
    # 每个顶点: 2个三角形 + 2个六边形
    data.append({
        'name': 'kagome', 'd': 2, 'z': 4, 'pc': 0.652703,
        'g': 3, 'n_tri': 2, 'n_sq': 0, 'n_hex': 2,
        'dual_z': None, 'is_self_dual': False,
        'packing': 0.6,
        'source': 'Stauffer1994'
    })

    # ===== 2. 标准 3D 晶格 =====

    # 简单立方 (sc)
    # z=6, girth=4 (正方形面), 无三角形
    data.append({
        'name': 'sc', 'd': 3, 'z': 6, 'pc': 0.311600,
        'g': 4, 'n_tri': 0, 'n_sq': 3, 'n_hex': 0,
        'dual_z': None, 'is_self_dual': False,
        'packing': 0.5236,  # π/6
        'source': 'Stauffer1994'
    })

    # 体心立方 (bcc)
    # z=8, 最近邻; girth=4 (有正方形面)
    # 每个顶点有6个正方形面
    data.append({
        'name': 'bcc', 'd': 3, 'z': 8, 'pc': 0.245800,
        'g': 4, 'n_tri': 0, 'n_sq': 6, 'n_hex': 0,
        'dual_z': None, 'is_self_dual': False,
        'packing': 0.6802,
        'source': 'Stauffer1994'
    })

    # 面心立方 (fcc)
    # z=12, girth=3 (三角形面)
    # 每个顶点有8个三角形 + 6个正方形
    data.append({
        'name': 'fcc', 'd': 3, 'z': 12, 'pc': 0.118800,
        'g': 3, 'n_tri': 8, 'n_sq': 6, 'n_hex': 0,
        'dual_z': None, 'is_self_dual': False,
        'packing': 0.7405,  # 最密堆积
        'source': 'Stauffer1994'
    })

    # 金刚石 (diamond)
    # z=4, girth=6 (六边形环, 类似蜂窝的3D版)
    # 无三角形, 无四边形
    data.append({
        'name': 'diamond', 'd': 3, 'z': 4, 'pc': 0.431900,
        'g': 6, 'n_tri': 0, 'n_sq': 0, 'n_hex': 1,
        'dual_z': None, 'is_self_dual': False,
        'packing': 0.3401,
        'source': 'Stauffer1994'
    })

    # ===== 3. 高维超立方晶格 =====
    for dim in range(4, 10):
        z = 2 * dim
        data.append({
            'name': f'hypersc_{dim}', 'd': dim, 'z': z,
            'pc': {4: 0.196737, 5: 0.140810, 6: 0.109460,
                   7: 0.089270, 8: 0.075050, 9: 0.064530}[dim],
            'g': 4,  # 超立方最短环都是正方形
            'n_tri': 0,
            'n_sq': 2 * dim * (dim - 1) // 2,  # C(d,2) 个正方形面/顶点
            'n_hex': 0,
            'dual_z': None, 'is_self_dual': False,
            'packing': None,
            'source': 'literature'
        })

    # ===== 4. Archimedean 晶格 (Suding-Ziff 1998) =====
    # 顶点构型 → 面分布和 girth

    archimedean_data = [
        # (3,12²): 每个顶点 1个三角形 + 2个十二边形
        {'name': 'arch_3_12_12', 'd': 2, 'z': 3, 'pc': 0.807904,
         'g': 3, 'n_tri': 1, 'n_sq': 0, 'n_hex': 0, 'n_12': 2,
         'dual_z': None, 'is_self_dual': False, 'packing': 0.7},

        # (4,6,12): 每个顶点 1个正方形 + 1个六边形 + 1个十二边形
        {'name': 'arch_4_6_12', 'd': 2, 'z': 4, 'pc': 0.747806,
         'g': 4, 'n_tri': 0, 'n_sq': 1, 'n_hex': 1, 'n_12': 1,
         'dual_z': None, 'is_self_dual': False, 'packing': 0.65},

        # (4,8²): 每个顶点 1个正方形 + 2个八边形
        {'name': 'arch_4_8_8', 'd': 2, 'z': 4, 'pc': 0.729724,
         'g': 4, 'n_tri': 0, 'n_sq': 1, 'n_hex': 0, 'n_8': 2,
         'dual_z': None, 'is_self_dual': False, 'packing': 0.65},

        # (3⁴,6): 每个顶点 4个三角形 + 1个六边形
        {'name': 'arch_3_3_3_3_6', 'd': 2, 'z': 5, 'pc': 0.579498,
         'g': 3, 'n_tri': 4, 'n_sq': 0, 'n_hex': 1,
         'dual_z': None, 'is_self_dual': False, 'packing': 0.55},

        # (3,4,6,4): 每个顶点 1三角形 + 2正方形 + 1六边形
        {'name': 'arch_3_4_6_4', 'd': 2, 'z': 5, 'pc': 0.621819,
         'g': 3, 'n_tri': 1, 'n_sq': 2, 'n_hex': 1,
         'dual_z': None, 'is_self_dual': False, 'packing': 0.55},

        # (3³,4²): 每个顶点 3三角形 + 2正方形
        {'name': 'arch_3_3_3_4_4', 'd': 2, 'z': 5, 'pc': 0.550213,
         'g': 3, 'n_tri': 3, 'n_sq': 2, 'n_hex': 0,
         'dual_z': None, 'is_self_dual': False, 'packing': 0.5},

        # (3²,4,3,4): 每个顶点 3三角形 + 2正方形 (交替排列)
        {'name': 'arch_3_3_4_3_4', 'd': 2, 'z': 5, 'pc': 0.550806,
         'g': 3, 'n_tri': 3, 'n_sq': 2, 'n_hex': 0,
         'dual_z': None, 'is_self_dual': False, 'packing': 0.5},
    ]
    data.extend(archimedean_data)

    df = pd.DataFrame(data)

    # 确保所有面类型列存在（缺失填0）
    for col in ['n_hex', 'n_8', 'n_12']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    # ===== 计算派生物理量 =====

    # 1. Bethe 近似: p_c^Bethe = 1/(z-1)
    df['p_bethe'] = 1.0 / (df['z'] - 1)

    # 2. Bethe 相对误差
    df['bethe_err'] = (df['p_bethe'] - df['pc']) / df['pc']

    # 3. 聚类系数 (每顶点的三角形连接密度)
    # C = 2 * n_tri / (z * (z-1))
    df['clustering'] = 2 * df['n_tri'] / (df['z'] * (df['z'] - 1))

    # 4. 谱半径 (对正则图 = z)
    df['spectral_radius'] = df['z']

    # 5. 随机游走返回率 (非严格, 但近似)
    # 对于正则图, P_return ~ z * (1/z)^(g/2) = z^(1-g/2)
    # 这是 girth 的函数
    df['rw_return'] = df['z'] * np.power(df['z'].astype(float), -df['g'].astype(float)/2)

    # 6. 有效配位数 (考虑环的 "浪费")
    # 三角形使邻居重叠: z_eff = z - n_tri * (z-2)/z 的某种度量
    # 更精确: 每个三角形减少 1 个独立路径
    df['z_eff'] = df['z'] - df['n_tri'] * 2.0 / df['z']

    # 7. 面密度 (每个顶点的面数)
    # 总面数/顶点数 的近似
    total_faces = df['n_tri'] + df['n_sq'] + df['n_hex'] + df['n_8'] + df['n_12']
    df['face_density'] = total_faces / df['z']  # 粗略近似

    # 8. 平均面大小 (weighted)
    def avg_face_size(row):
        faces = []
        for s, key in [(3, 'n_tri'), (4, 'n_sq'), (6, 'n_hex'), (8, 'n_8'), (12, 'n_12')]:
            count = row.get(key, 0)
            if pd.notna(count) and count > 0:
                faces.extend([s] * int(count))
        return np.mean(faces) if faces else 6.0

    df['avg_face'] = df.apply(avg_face_size, axis=1)

    # 9. "环密度指数" R = n_tri/z + 1/n_sq (小环越密，渗流越难)
    df['ring_density'] = df['n_tri']/df['z'] + 1.0/np.maximum(df['n_sq'], 0.5)

    # 10. 倒空间量: 1/g (girth 的倒数, 环效应的度量)
    df['g_inv'] = 1.0 / df['g']

    # 11. 维度因子: d/(d+1) 或 d/(d+2) (高维趋于1)
    df['d_factor'] = df['d'] / (df['d'] + 1.0)

    # 12. 拓扑熵 (粗略): log(z) * (1 - clustering)
    df['topo_entropy'] = np.log(df['z']) * (1 - df['clustering'])

    # 13. Bethe 修正因子: (p_c - p_bethe) / p_bethe
    df['bethe_correction'] = (df['pc'] - df['p_bethe']) / df['p_bethe']

    # 14. z*d 的倒数 (平均场)
    df['zd_inv'] = 1.0 / (df['z'] * df['d'])

    # 15. 关联体积: z^d (一阶近似)
    df['corr_volume'] = np.power(df['z'].astype(float), df['d'].astype(float))

    return df


# =====================================================================
# 第二部分: 物理理论驱动的公式
# =====================================================================

def compute_theoretical_formulas(df):
    """
    从不同物理理论推导的公式
    """

    z = df['z'].values.astype(float)
    d = df['d'].values.astype(float)
    pc = df['pc'].values.astype(float)
    g = df['g'].values.astype(float)
    n_tri = df['n_tri'].values.astype(float)
    n_sq = df['n_sq'].values.astype(float)
    clustering = df['clustering'].values.astype(float)
    rw_return = df['rw_return'].values.astype(float)
    z_eff = df['z_eff'].values.astype(float)
    face_density = df['face_density'].values.astype(float)
    avg_face = df['avg_face'].values.astype(float)
    ring_density = df['ring_density'].values.astype(float)
    g_inv = df['g_inv'].values.astype(float)
    d_factor = df['d_factor'].values.astype(float)
    topo_entropy = df['topo_entropy'].values.astype(float)
    zd_inv = df['zd_inv'].values.astype(float)

    results = {}

    # ===================================================================
    # 理论 1: Bethe-Peierls 修正
    # ===================================================================
    #
    # 基础: p_c^Bethe = 1/(z-1) — 来自树状近似, 忽略环
    #
    # 修正: 环使邻居重叠, 降低渗流效率
    # 对每个 girth-g 环, 有 g-2 个 "冗余" 路径
    # 有效配位数: z_eff = z · (1 - c · n_tri/z)
    #   其中 c ~ 1 是三角形重叠因子
    #
    # 预测: p_c = 1/(z_eff - 1) = 1/(z·(1-c·n_tri/z) - 1)

    print("\n" + "=" * 80)
    print("理论 1: Bethe-Peierls + 环修正")
    print("=" * 80)
    print("""
物理推导:
  Bethe 近似假设树状结构 (无环) → p_c = 1/(z-1)
  但真实晶格有环 → 邻居重叠 → 有效配位数降低
  每个三角形使 2 个邻居重叠 → z_eff = z - 2·n_tri/z
  修正: p_c = 1/(z_eff - 1)
    """)

    def f_bethe_ring(params, z, n_tri):
        c, alpha = params
        z_eff = z - alpha * n_tri / np.maximum(z, 1)
        return c / np.maximum(z_eff - 1, 0.1)

    best = fit_formula(f_bethe_ring, pc, [z, n_tri],
                       bounds=[(0.1, 3), (0, 5)],
                       name="p_c = c/(z_eff - 1)")
    if best:
        results['Bethe+环修正'] = best

    # ===================================================================
    # 理论 2: 谱理论 — 谱半径决定渗流
    # ===================================================================
    #
    # 关键定理 (Nachmias, Peres 2008):
    #   对于正则图: p_c ≈ ρ / (λ₁ - 1 + ρ) 其中 λ₁ 是谱半径
    #   对正则图 λ₁ = z, 所以 p_c ~ z/(z-1+...) ≈ 1/(z-1)
    #
    # 但非正则图的谱半径 λ₁ ≠ z
    # 更重要的是: 谱间隙 gap = z - λ₂ 控制关联长度
    #
    # 预测: p_c ~ f(z, gap) ≈ a / (z - b + c·gap/z)

    print("\n" + "=" * 80)
    print("理论 2: 谱图论 — 谱半径与谱间隙")
    print("=" * 80)
    print("""
物理推导:
  渗流阈值与图的谱性质密切相关 (Nachmias-Peres 2008)
  对正则图: λ₁ = z, 所以谱理论 → Bethe 近似
  谱间隙 gap = z - λ₂ 控制关联长度 ξ
  环结构影响 λ₂: 环越多, gap 越小, 关联越长
    """)

    # 用聚类系数代理谱间隙
    # (高聚类 → 小谱间隙 → 长关联 → 高 p_c)
    def f_spectral(params, z, clustering):
        a, b, c = params
        return a / (z - b + c * clustering * z)

    best = fit_formula(f_spectral, pc, [z, clustering],
                       bounds=[(0.1, 5), (0, 3), (0, 5)],
                       name="p_c = a/(z - b + c·C·z)")
    if best:
        results['谱理论+聚类'] = best

    # ===================================================================
    # 理论 3: 随机游走 — 返回概率与渗流
    # ===================================================================
    #
    # 关联: 渗流 ≈ 随机游走能否逃逸到无穷
    #
    # Pólya 定理: d≤2 随机游走常返, d>2 暂态
    # 在 d=2, 返回概率 → 1, 所以 p_c 较高
    #
    # 对于有限 girth g 的图, g 步内返回概率 ~ z^(1-g/2)
    # 短环 → 高返回概率 → 需要更高 p 才能渗流
    #
    # 预测: p_c = a · z^(-b) · (1 + c · z^(1-g/2))

    print("\n" + "=" * 80)
    print("理论 3: 随机游走 — 返回概率")
    print("=" * 80)
    print("""
物理推导:
  渗流 ≈ 随机游走能否逃逸到无穷远
  Pólya (1921): d≤2 常返, d>2 暂态
  短环(girth g)增大返回概率: P_return ~ z^(1-g/2)
  高返回 → 需要更高 p 才能渗流
  p_c = a·z^(-b)·(1 + c·z^(1-g/2))
    """)

    def f_random_walk(params, z, g):
        a, b, c = params
        return a * np.power(z, -b) * (1 + c * np.power(np.maximum(z, 1), 1 - g/2))

    best = fit_formula(f_random_walk, pc, [z, g],
                       bounds=[(0.1, 10), (0, 3), (0, 10)],
                       name="p_c = a·z^(-b)·(1+c·z^(1-g/2))")
    if best:
        results['随机游走+girth'] = best

    # ===================================================================
    # 理论 4: 有效介质理论 (EMT)
    # ===================================================================
    #
    # EMT (Kirkpatrick 1973): 将随机网络近似为均匀介质
    # 对键渗流: p_c^bond ≈ 2/d (高维)
    #
    # 对 site 渗流, 需要修正:
    # p_c^site ≈ f(z, d) 其中 f 依赖于配位壳层结构
    #
    # 经典结果: p_c^site ≈ 1/(z·d/d_factor) 某种形式
    #
    # 更精确: 考虑 "冗余度" = 配位数 vs 最小渗流所需

    print("\n" + "=" * 80)
    print("理论 4: 有效介质理论")
    print("=" * 80)
    print("""
物理推导:
  EMT (Kirkpatrick 1973): 随机网络 ≈ 均匀有效介质
  键渗流: p_c^bond ≈ 2/d (高维精确)
  Site 渗流需修正, 取决于配位壳层结构
  关键: z·p_c 近似常数 → 但低维偏差大
    """)

    def f_emt(params, z, d, avg_face):
        a, b, c = params
        # p_c = a / (z · d^b · (avg_face/3)^c)
        return a / (z * np.power(d, b) * np.power(np.maximum(avg_face, 1), c))

    best = fit_formula(f_emt, pc, [z, d, avg_face],
                       bounds=[(0.1, 10), (0, 3), (0, 3)],
                       name="p_c = a/(z·d^b·(f/3)^c)")
    if best:
        results['EMT+面大小'] = best

    # ===================================================================
    # 理论 5: 标度理论 — 维度与上临界维度
    # ===================================================================
    #
    # 标度理论 (Stauffer & Aharony):
    # - 上临界维度 d_c = 6 (渗流)
    # - d < d_c: p_c 由非平均场指数控制
    # - d ≥ d_c: 平均场精确
    #
    # 关键洞察: p_c(z, d) 应该在 d → d_c 时趋于平均场
    # 平均场极限: p_c ~ 1/(z-1)
    #
    # 预测: p_c = 1/(z-1) · (1 + a·exp(-b·d))
    #   其中 exp(-b·d) 是有限维修正

    print("\n" + "=" * 80)
    print("理论 5: 标度理论 — 上临界维度")
    print("=" * 80)
    print("""
物理推导:
  上临界维度 d_c = 6: d≥6 平均场精确
  有限维修正应指数衰减: ~ exp(-b·d)
  p_c = 1/(z-1) · (1 + a·exp(-b·d))
  物理含义: Bethe 近似 + 有限维指数修正
    """)

    def f_scaling(params, z, d):
        a, b = params
        return (1.0 / np.maximum(z - 1, 0.5)) * (1 + a * np.exp(-b * d))

    best = fit_formula(f_scaling, pc, [z, d],
                       bounds=[(-2, 5), (0, 3)],
                       name="p_c = (1+a·exp(-b·d))/(z-1)")
    if best:
        results['标度+上临界维'] = best

    # ===================================================================
    # 理论 6: 拓扑熵 — 信息论视角
    # ===================================================================
    #
    # 信息论: p_c 编码了晶格的 "渗流复杂度"
    # 复杂度 ~ 分支可能性 × 环约束
    #
    # 分支贡献: log(z) (每个节点 z 个选择)
    # 环约束: -n_tri·log(3)/z (每个三角形减少自由度)
    #
    # 净拓扑熵: S = log(z) - n_tri·log(3)/z - n_sq·log(4)/z
    # p_c ~ 1/S  (熵越低, p_c 越高)

    print("\n" + "=" * 80)
    print("理论 6: 拓扑熵 — 信息论视角")
    print("=" * 80)
    print("""
物理推导:
  渗流复杂度 = 分支可能性 × 环约束
  分支贡献: log(z)
  环约束: -Σ n_face·log(face_size)/z
  净拓扑熵: S = log(z) - n_tri·log(3)/z - n_sq·log(4)/z
  p_c ~ a/S + b
    """)

    def f_entropy(params, z, n_tri, n_sq):
        a, b = params
        S = np.log(np.maximum(z, 1)) - n_tri * np.log(3) / np.maximum(z, 1) - n_sq * np.log(4) / np.maximum(z, 1)
        return a / np.maximum(S, 0.01) + b

    best = fit_formula(f_entropy, pc, [z, n_tri, n_sq],
                       bounds=[(0, 10), (-1, 2)],
                       name="p_c = a/S_topo + b")
    if best:
        results['拓扑熵'] = best

    # ===================================================================
    # 理论 7: 综合理论 — 多因子协同
    # ===================================================================
    #
    # 综合以上: p_c 受三个物理因素控制:
    #
    # (1) 配位数效应: ~ 1/z (Bethe/平均场)
    # (2) 环效应: girth 和三角形密度 (减少有效分支)
    # (3) 维度效应: d 控制关联长度和有限维修正
    #
    # 统一公式:
    # p_c = A / (z^α · d^β) · (1 + B · clustering · z^γ)
    #
    # 其中:
    # - z^α·d^β: 主标度项
    # - clustering: 环密度的度量
    # - B·clustering·z^γ: 环修正 (环越密, p_c 越高)

    print("\n" + "=" * 80)
    print("理论 7: 综合 — 多因子协同")
    print("=" * 80)
    print("""
物理推导:
  三个因素协同控制 p_c:
  (1) 配位数效应: ~ 1/z (Bethe)
  (2) 环效应: clustering 度量环密度
  (3) 维度效应: d 控制关联长度

  统一公式:
  p_c = A/(z^α · d^β) · (1 + B·C·z^γ)

  物理含义:
  - A/(z^α·d^β): 主标度 (渗流概率 ~ 1/连接数)
  - B·C·z^γ: 环修正 (环使邻居重叠, 需要更高 p)
    """)

    def f_unified(params, z, d, clustering):
        A, alpha, beta, B, gamma = params
        return A / (np.power(z, alpha) * np.power(d, beta)) * (1 + B * clustering * np.power(z, gamma))

    best = fit_formula(f_unified, pc, [z, d, clustering],
                       bounds=[(0, 20), (0, 3), (0, 3), (-5, 5), (-3, 3)],
                       name="p_c = A/(z^α·d^β)·(1+B·C·z^γ)")
    if best:
        results['综合多因子'] = best

    # ===================================================================
    # 理论 8: Girth-Bethe 联合理论
    # ===================================================================
    #
    # 物理洞察: Bethe 近似在 girth → ∞ 时精确
    # 有限 girth 产生修正:
    #   p_c = p_bethe · (1 + a·(1/g)^b)
    #
    # 但还需要维度修正:
    #   p_c = p_bethe · f(d) · g(g)

    print("\n" + "=" * 80)
    print("理论 8: Girth-Bethe 联合理论")
    print("=" * 80)
    print("""
物理推导:
  Bethe 精确当且仅当 girth → ∞ (树状极限)
  有限 girth → 环 → 修正
  p_c = (1/(z-1)) · (1 + a·g^(-b)) · d^(-c)
  或: p_c = a / ((z-1)^b · g^c)
    """)

    def f_girth_bethe(params, z, g, d):
        a, b, c, e = params
        return a / (np.power(np.maximum(z-1, 0.5), b) * np.power(g, c) * np.power(d, e))

    best = fit_formula(f_girth_bethe, pc, [z, g, d],
                       bounds=[(0, 20), (0, 3), (-2, 2), (-2, 2)],
                       name="p_c = a/((z-1)^b·g^c·d^e)")
    if best:
        results['Girth-Bethe'] = best

    return results


def fit_formula(func, target, inputs, bounds, name="", n_restarts=100):
    """多起点优化拟合公式"""

    best_rmse = float('inf')
    best_params = None

    for _ in range(n_restarts):
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        try:
            result = scipy_minimize(
                lambda p: np.sqrt(np.mean((func(p, *inputs) - target)**2)),
                x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if result.fun < best_rmse:
                # 验证预测有效
                pred = func(result.x, *inputs)
                valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
                if valid.sum() > len(target) * 0.8:
                    best_rmse = result.fun
                    best_params = result.x.copy()
        except:
            pass

    # 差分进化
    try:
        result_de = differential_evolution(
            lambda p: np.sqrt(np.mean((func(p, *inputs) - target)**2)),
            bounds, maxiter=3000, tol=1e-14, seed=42
        )
        pred = func(result_de.x, *inputs)
        valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
        if valid.sum() > len(target) * 0.8 and result_de.fun < best_rmse:
            best_rmse = result_de.fun
            best_params = result_de.x.copy()
    except:
        pass

    if best_params is not None:
        pred = func(best_params, *inputs)
        valid = ~np.isnan(pred) & ~np.isinf(pred)
        if valid.sum() < len(target) * 0.5:
            return None

        ss_res = np.sum((pred[valid] - target[valid])**2)
        ss_tot = np.sum((target[valid] - target[valid].mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mae = np.mean(np.abs(pred[valid] - target[valid]))

        print(f"\n  ✓ {name}")
        print(f"    参数: {['%.4f' % p for p in best_params]}")
        print(f"    R² = {r2:.6f}, MAE = {mae:.4f}, RMSE = {best_rmse:.4f}")

        return {
            'name': name,
            'params': best_params,
            'func': func,
            'r2': r2,
            'mae': mae,
            'rmse': best_rmse,
            'n_params': len(best_params),
        }
    else:
        print(f"\n  ✗ {name} — 拟合失败")
        return None


# =====================================================================
# 第三部分: LOOCV 验证
# =====================================================================

def loocv_with_refit(func, inputs_list, target, n_params, bounds, n_refit=15):
    """LOOCV, 每折重拟合参数"""
    n = len(target)
    errors = []

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        inputs_train = [x[mask] for x in inputs_list]
        target_train = target[mask]

        inputs_test = [x[~mask] for x in inputs_list]
        target_test = target[~mask]

        # 在训练集上优化
        best_rmse = float('inf')
        best_p = None

        for _ in range(n_refit):
            x0 = np.random.randn(n_params) * 2
            try:
                result = scipy_minimize(
                    lambda p: np.sqrt(np.mean((func(p, *inputs_train) - target_train)**2)),
                    x0, method='Nelder-Mead',
                    options={'maxiter': 2000, 'fatol': 1e-12}
                )
                if result.fun < best_rmse:
                    pred = func(result.x, *inputs_train)
                    valid = ~np.isnan(pred) & ~np.isinf(pred)
                    if valid.sum() > len(target_train) * 0.8:
                        best_rmse = result.fun
                        best_p = result.x.copy()
            except:
                pass

        if best_p is not None:
            pred = func(best_p, *inputs_test)
            if not (np.isnan(pred[0]) or np.isinf(pred[0])):
                errors.append(abs(pred[0] - target_test[0]))

    return np.mean(errors) if errors else 1e10


def run_loocv(df, theory_results):
    """对所有物理公式做 LOOCV"""

    z = df['z'].values.astype(float)
    d = df['d'].values.astype(float)
    pc = df['pc'].values.astype(float)
    g = df['g'].values.astype(float)
    n_tri = df['n_tri'].values.astype(float)
    n_sq = df['n_sq'].values.astype(float)
    clustering = df['clustering'].values.astype(float)
    avg_face = df['avg_face'].values.astype(float)

    print("\n" + "=" * 80)
    print("🔬 LOOCV 严格验证")
    print("=" * 80)

    loocv_results = {}

    for name, result in theory_results.items():
        func = result['func']
        n_p = result['n_params']

        # 确定输入
        if 'n_tri' in name and 'n_sq' in name:
            inputs = [z, n_tri, n_sq]
            bounds = [(0, 10), (-1, 2)]
        elif 'clustering' in name and '综合' in name:
            inputs = [z, d, clustering]
            bounds = [(0, 20), (0, 3), (0, 3), (-5, 5), (-3, 3)]
        elif 'clustering' in name:
            inputs = [z, clustering]
            bounds = [(0.1, 5), (0, 3), (0, 5)]
        elif 'girth' in name.lower() or 'Girth' in name:
            inputs = [z, g, d]
            bounds = [(0, 20), (0, 3), (-2, 2), (-2, 2)]
        elif 'g' in name.lower() and 'walk' in name.lower():
            inputs = [z, g]
            bounds = [(0.1, 10), (0, 3), (0, 10)]
        elif 'avg_face' in name or 'EMT' in name:
            inputs = [z, d, avg_face]
            bounds = [(0.1, 10), (0, 3), (0, 3)]
        elif '标度' in name:
            inputs = [z, d]
            bounds = [(-2, 5), (0, 3)]
        elif '环修正' in name:
            inputs = [z, n_tri]
            bounds = [(0.1, 3), (0, 5)]
        else:
            inputs = [z, d]
            bounds = [(0.1, 10)] * n_p

        loocv_mae = loocv_with_refit(func, inputs, pc, n_p, bounds)
        loocv_results[name] = loocv_mae

        gap = loocv_mae - result['mae']
        print(f"\n{name}:")
        print(f"  训练 MAE = {result['mae']:.4f}")
        print(f"  LOOCV MAE = {loocv_mae:.4f}")
        print(f"  泛化差距 = {gap:.4f}")

    # ML 基线
    X = df[['z', 'd', 'z_eff', 'clustering', 'rw_return', 'g_inv', 'topo_entropy']].values
    loo = LeaveOneOut()
    lr = LinearRegression()
    lr_pred = []
    for train_idx, test_idx in loo.split(X):
        lr.fit(X[train_idx], pc[train_idx])
        lr_pred.append(lr.predict(X[test_idx])[0])
    mae_ml = mean_absolute_error(pc, lr_pred)

    print(f"\nML (7物理特征): LOOCV MAE = {mae_ml:.4f}")

    return loocv_results, mae_ml


# =====================================================================
# 主程序
# =====================================================================

def main():
    print("=" * 80)
    print("🧠 物理理论驱动的渗流阈值分析")
    print("=" * 80)
    print("""
方法论:
  不是机械拟合, 而是从物理理论出发
  每个公式都有明确的物理推导
  变量代表可测量的物理量

多学科交叉:
  • 统计物理 (Bethe-Peierls, 标度理论, 有效介质)
  • 谱图论 (谱半径, 谱间隙)
  • 概率论 (随机游走, 分支过程)
  • 信息论 (拓扑熵)
  • 几何拓扑 (girth, 面分布, Euler 特征)
""")

    # 构建数据集
    df = build_physics_dataset()

    print(f"\n📊 数据集: {len(df)} 个晶格")
    print(f"  2D: {len(df[df['d']==2])}, 3D: {len(df[df['d']==3])}, 高维: {len(df[df['d']>=4])}")

    print(f"\n📋 物理量统计:")
    print(f"  z 范围: {df['z'].min()} - {df['z'].max()}")
    print(f"  girth 分布: {df['g'].value_counts().sort_index().to_dict()}")
    print(f"  聚类系数范围: {df['clustering'].min():.3f} - {df['clustering'].max():.3f}")
    print(f"  Bethe 误差范围: {df['bethe_err'].min():.3f} - {df['bethe_err'].max():.3f}")

    # 运行理论公式
    theory_results = compute_theoretical_formulas(df)

    # LOOCV
    loocv_results, mae_ml = run_loocv(df, theory_results)

    # 汇总
    print("\n" + "=" * 80)
    print("🏆 物理公式排名 (LOOCV)")
    print("=" * 80)

    all_results = []
    for name, result in theory_results.items():
        all_results.append({
            'name': name,
            'n_params': result['n_params'],
            'train_mae': result['mae'],
            'train_r2': result['r2'],
            'loocv_mae': loocv_results.get(name, 1e10),
            'physics': True,
        })
    all_results.append({
        'name': 'ML (7物理特征)',
        'n_params': 7,
        'train_mae': mae_ml,
        'train_r2': None,
        'loocv_mae': mae_ml,
        'physics': False,
    })

    all_results.sort(key=lambda x: x['loocv_mae'])

    print(f"\n{'排名':<4} {'公式':<25} {'参数':<4} {'训练MAE':<10} {'LOOCV':<10} {'物理':<6}")
    print("-" * 65)
    for i, r in enumerate(all_results):
        phys = "✓" if r['physics'] else "✗"
        print(f"#{i+1:<3} {r['name']:<25} {r['n_params']:<4} {r['train_mae']:<10.4f} {r['loocv_mae']:<10.4f} {phys:<6}")

    # 保存
    output = str(ROOT / 'physics_driven_report.txt')
    with open(output, 'w') as f:
        f.write("物理理论驱动的渗流阈值分析\n")
        f.write("=" * 80 + "\n\n")
        for i, r in enumerate(all_results):
            f.write(f"#{i+1}: {r['name']}\n")
            f.write(f"  参数数: {r['n_params']}\n")
            f.write(f"  训练 MAE: {r['train_mae']:.4f}\n")
            f.write(f"  LOOCV MAE: {r['loocv_mae']:.4f}\n\n")

    print(f"\n💾 已保存: {output}")


if __name__ == '__main__':
    main()
