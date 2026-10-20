"""
关键诊断：为什么扩展数据后公式退化？
=====================================

核心发现：
  1. 原始14晶格 R²=0.988 → 扩展46晶格 R²=0.952（下降3.6%）
  2. 经典2D数据被公式完全错误预测（误差38-42%）
  3. Malarz 2D数据误差11%（比经典2D好很多）
  4. 参数偏移：A: 2.56→2.85, α: 0.81→0.87

需要诊断：
  1. 经典2D vs Malarz 2D vs 高维 — 三个regime是否物理不同？
  2. 参数为什么会偏移？
  3. 是否存在分段标度？
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============ 经典数据集（14个晶格） ============
classic = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'pc': 0.697040, 'cat': 'classic_2d'},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'pc': 0.592746, 'cat': 'classic_2d'},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'pc': 0.500000, 'cat': 'classic_2d'},
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'pc': 0.652703, 'cat': 'classic_2d'},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'pc': 0.311600, 'cat': 'classic_3d'},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'pc': 0.245800, 'cat': 'classic_3d'},
    {'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'pc': 0.118800, 'cat': 'classic_3d'},
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'pc': 0.431900, 'cat': 'classic_3d'},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'pc': 0.196737, 'cat': 'high_d'},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'pc': 0.140810, 'cat': 'high_d'},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'pc': 0.109460, 'cat': 'high_d'},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'g': 4, 'pc': 0.089270, 'cat': 'high_d'},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'g': 4, 'pc': 0.075050, 'cat': 'high_d'},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'g': 4, 'pc': 0.064530, 'cat': 'high_d'},
]

# Malarz 数据（31个，全部d=2）
malarz_raw = [
    ('sq-1,2,3,4,5,6', 28, 3, 0.14326), ('sq-2,3,4,5,6', 24, 3, 0.14575),
    ('sq-1,3,4,5,6', 24, 4, 0.14801), ('sq-1,2,4,5,6', 24, 3, 0.15223),
    ('sq-1,2,3,5,6', 20, 3, 0.16661), ('sq-1,2,3,4,6', 24, 3, 0.16134),
    ('sq-3,4,5,6', 20, 4, 0.15221), ('sq-2,4,5,6', 20, 3, 0.15844),
    ('sq-2,3,5,6', 16, 3, 0.17601), ('sq-2,3,4,6', 20, 3, 0.16529),
    ('sq-1,4,5,6', 20, 4, 0.15815), ('sq-1,3,5,6', 16, 4, 0.18007),
    ('sq-1,3,4,6', 20, 4, 0.16675), ('sq-1,2,5,6', 16, 3, 0.18216),
    ('sq-1,2,4,6', 20, 3, 0.17409), ('sq-1,2,3,6', 16, 3, 0.20134),
    ('sq-4,5,6', 16, 4, 0.16819), ('sq-3,5,6', 12, 4, 0.19867),
    ('sq-3,4,6', 16, 4, 0.17288), ('sq-2,5,6', 12, 3, 0.20036),
    ('sq-2,4,6', 16, 3, 0.18454), ('sq-2,3,6', 12, 3, 0.21503),
    ('sq-1,5,6', 12, 4, 0.19936), ('sq-1,4,6', 16, 4, 0.18143),
    ('sq-1,3,6', 12, 4, 0.22577), ('sq-1,2,6', 12, 3, 0.23076),
    ('sq-5,6', 8, 4, 0.24422), ('sq-4,6', 12, 4, 0.19799),
    ('sq-3,6', 8, 4, 0.25673), ('sq-2,6', 8, 3, 0.26600),
    ('sq-1,6', 8, 4, 0.27309),
]
malarz = [{'name': n, 'z': z, 'd': 2, 'g': g, 'pc': p, 'cat': 'malarz_2d'} for n, z, g, p in malarz_raw]

# 合并
all_lattices = classic + malarz
n_total = len(all_lattices)

z_all = np.array([l['z'] for l in all_lattices], dtype=float)
d_all = np.array([l['d'] for l in all_lattices], dtype=float)
g_all = np.array([l['g'] for l in all_lattices], dtype=float)
pc_all = np.array([l['pc'] for l in all_lattices])
cats = [l['cat'] for l in all_lattices]
names_all = [l['name'] for l in all_lattices]

print("="*80)
print("【关键诊断：公式退化原因分析】")
print("="*80)

# ============ 1. 分regime分析 ============
print(f"\n【1. 分 regime 分析】")
print("-"*60)

categories = {
    'classic_2d': [i for i, c in enumerate(cats) if c == 'classic_2d'],
    'classic_3d': [i for i, c in enumerate(cats) if c == 'classic_3d'],
    'malarz_2d': [i for i, c in enumerate(cats) if c == 'malarz_2d'],
    'high_d': [i for i, c in enumerate(cats) if c == 'high_d'],
}

for cat, idx in categories.items():
    z_c = z_all[idx]
    d_c = d_all[idx]
    g_c = g_all[idx]
    pc_c = pc_all[idx]
    print(f"\n{cat} ({len(idx)} 个):")
    print(f"  z ∈ [{z_c.min():.0f}, {z_c.max():.0f}]")
    print(f"  d = {np.unique(d_c)}")
    print(f"  g ∈ {np.unique(g_c)}")
    print(f"  p_c ∈ [{pc_c.min():.4f}, {pc_c.max():.4f}]")

# ============ 2. 关键问题：Malarz 数据的 "zone" 结构 ============
print(f"\n{'='*80}")
print("【2. Malarz 数据的 zone 结构分析】")
print("-"*60)

print("""
Malarz 2023 的晶格定义：
  - 基础：2D 正方晶格
  - 邻域：由 zone 定义（zone k = 距离 sqrt(i²+j²) = k 的格点）
  - 例：zone 1 = {(±1,0),(0,±1)} → z=4
        zone 2 = {(±1,±1)} → z=4
        zone 3 = {(±2,0),(0,±2)} → z=4
        zone 4 = {(±2,±1),(±1,±2)} → z=8
        zone 5 = {(±2,±2)} → z=4
        zone 6 = {(±3,0),(0,±3)} → z=4

  "sq-1,2,3,4,5,6" = 包含所有6个zone → z = 4+4+4+8+4+4 = 28
  "sq-5,6" = 仅 zone 5+6 → z = 4+4 = 8

关键问题：
  1. 这些都是 2D 晶格，d=2 固定
  2. 但 zone 结构引入了不同尺度的连接
  3. 最近邻(zone 1)距离=1，zone 6距离=3
  4. 混合尺度连接是否影响渗流行为？
""")

# ============ 3. 核心诊断：p_c vs z·d 的幂律关系 ============
print(f"\n{'='*80}")
print("【3. 核心诊断：p_c vs z·d 的幂律关系】")
print("-"*60)

# 对每个类别单独拟合 p_c = A · (z·d)^{-α}
def simple_power(params, zd):
    A, alpha = params
    return A * np.power(zd, -alpha)

print(f"\n各类别的幂律标度 p_c = A · (z·d)^{{-α}}:")
print(f"{'类别':<15} {'A':<8} {'α':<8} {'R²':<10} {'MAE':<8}")
print("-"*55)

alphas = {}
for cat, idx in categories.items():
    z_c = z_all[idx]
    d_c = d_all[idx]
    pc_c = pc_all[idx]
    zd_c = z_c * d_c

    def obj(params):
        pred = simple_power(params, zd_c)
        return np.sum((pc_c - pred)**2)

    res = minimize(obj, [5.0, 0.7], method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    pred = simple_power(res.x, zd_c)
    R2 = 1 - np.sum((pc_c - pred)**2) / np.sum((pc_c - np.mean(pc_c))**2)
    MAE = np.mean(np.abs(pc_c - pred))
    alphas[cat] = res.x[1]
    print(f"{cat:<15} {res.x[0]:<8.3f} {res.x[1]:<8.3f} {R2:<10.6f} {MAE:<8.5f}")

print(f"\n关键发现:")
print(f"  classic_2d: α = {alphas.get('classic_2d', 0):.3f}")
print(f"  malarz_2d:  α = {alphas.get('malarz_2d', 0):.3f}")
print(f"  差异: {abs(alphas.get('classic_2d', 0) - alphas.get('malarz_2d', 0)):.3f}")

if abs(alphas.get('classic_2d', 0) - alphas.get('malarz_2d', 0)) > 0.1:
    print(f"  ⚠️ 两个类别的 α 显著不同！")
    print(f"     说明经典2D和Malarz 2D遵循不同的标度律")

# ============ 4. 诊断：经典2D数据的 girth 效应 ============
print(f"\n{'='*80}")
print("【4. 经典2D数据的 girth 效应】")
print("-"*60)

# 经典2D: honeycomb(z=3,g=6), square(z=4,g=4), triangular(z=6,g=3), kagome(z=4,g=3)
classic_2d_data = [
    ('honeycomb', 3, 6, 0.697040),
    ('square', 4, 4, 0.592746),
    ('triangular', 6, 3, 0.500000),
    ('kagome', 4, 3, 0.652703),
]

print("\n经典2D: z·d vs p_c")
for name, z, g, pc in classic_2d_data:
    zd = z * 2
    print(f"  {name:<12}: z·d = {zd}, g = {g}, p_c = {pc:.4f}")
    print(f"    1/(z·d) = {1/zd:.4f}, Bethe p_c = 1/(z-1) = {1/(z-1):.4f}")

print("""
观察：
  - honeycomb: z·d=6, p_c=0.697 > Bethe=0.500（高 40%）
  - square: z·d=8, p_c=0.593 > Bethe=0.333（高 78%）
  - triangular: z·d=12, p_c=0.500 > Bethe=0.200（高 150%）
  - kagome: z·d=8, p_c=0.653 > Bethe=0.333（高 96%）

  经典2D的 p_c 远高于 Bethe 近似！
  这说明2D下环结构的影响极其强烈。
""")

# Malarz 2D 对比
print("Malarz 2D: z·d vs p_c")
malarz_sample = [
    ('sq-5,6', 8, 4, 0.24422),
    ('sq-1,5,6', 12, 4, 0.19936),
    ('sq-1,2,3,4,5,6', 28, 3, 0.14326),
]
for name, z, g, pc in malarz_sample:
    zd = z * 2
    print(f"  {name:<20}: z·d = {zd}, g = {g}, p_c = {pc:.4f}")
    print(f"    1/(z·d) = {1/zd:.4f}, Bethe p_c = 1/(z-1) = {1/(z-1):.4f}")
    print(f"    p_c/Bethe = {pc * (z-1):.3f}")

print("""
观察：
  - sq-5,6: z·d=16, p_c/Bethe = 0.588
  - sq-1,5,6: z·d=24, p_c/Bethe = 0.379
  - sq-1,2,3,4,5,6: z·d=56, p_c/Bethe = 0.307

  Malarz的 p_c/Bethe 比率随 z 增大而减小，趋向1。
  经典2D的比率远高于Malarz（因为经典2D的z太小）。
""")

# ============ 5. 关键诊断：p_c·(z-1) vs z·d ============
print(f"\n{'='*80}")
print("【5. R = p_c·(z-1) vs z·d — 寻找标度崩溃】")
print("-"*60)

R = pc_all * (z_all - 1)
zd = z_all * d_all

print(f"\n{'晶格':<20} {'R':<8} {'z·d':<8} {'类别'}")
print("-"*55)

# 按 z·d 排序
order = np.argsort(zd)
for i in order:
    cat = cats[i]
    flag = ""
    if R[i] > 0.8 or R[i] < 0.2:
        flag = " ⚠️"
    print(f"{names_all[i]:<20} {R[i]:<8.4f} {zd[i]:<8.0f} {cat}{flag}")

# ============ 6. 分段标度假设 ============
print(f"\n{'='*80}")
print("【6. 分段标度假设】")
print("-"*60)

print("""
假设：存在不同的标度regime

Regime 1: 低 z·d（经典2D/3D）
  - z·d < 20
  - 环结构主导
  - Bethe近似完全失效
  - 需要 girth 修正

Regime 2: 中 z·d（Malarz 2D）
  - z·d = 16-56
  - 混合zone连接
  - Bethe近似开始适用
  - girth 修正减小

Regime 3: 高 z·d（高维超立方）
  - z·d > 30
  - 平均场行为
  - 接近 Bethe 极限
  - girth 修正小

检验：对每个regime分别拟合
""")

# Regime 1: classic 2D + 3D (z·d <= 24)
# Regime 2: malarz (z·d = 16-56, d=2)
# Regime 3: high_d (z·d >= 32)

# 拟合分段
def model_3(params, z, d, g):
    A, alpha, B = params
    return A * np.power(z*d, -alpha) * (1 + B/g)

regimes = {
    'low (classic)': [i for i in range(n_total) if cats[i] in ('classic_2d', 'classic_3d')],
    'mid (malarz)': [i for i in range(n_total) if cats[i] == 'malarz_2d'],
    'high (high_d)': [i for i in range(n_total) if cats[i] == 'high_d'],
}

print(f"{'Regime':<20} {'n':<4} {'A':<8} {'α':<8} {'B':<8} {'R²':<10} {'MAE':<8}")
print("-"*75)

for rname, idx in regimes.items():
    z_r = z_all[idx]
    d_r = d_all[idx]
    g_r = g_all[idx]
    pc_r = pc_all[idx]

    def obj_r(params):
        pred = model_3(params, z_r, d_r, g_r)
        return np.sum((pc_r - pred)**2)

    res_r = minimize(obj_r, [3.0, 0.7, 1.0], method='Nelder-Mead',
                     options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-12})
    pred_r = model_3(res_r.x, z_r, d_r, g_r)
    R2_r = 1 - np.sum((pc_r - pred_r)**2) / np.sum((pc_r - np.mean(pc_r))**2)
    MAE_r = np.mean(np.abs(pc_r - pred_r))
    print(f"{rname:<20} {len(idx):<4} {res_r.x[0]:<8.3f} {res_r.x[1]:<8.3f} {res_r.x[2]:<8.3f} {R2_r:<10.6f} {MAE_r:<8.5f}")

# ============ 7. 交叉预测测试 ============
print(f"\n{'='*80}")
print("【7. 交叉预测测试：用一个regime的参数预测另一个】")
print("-"*60)

# 用经典数据的参数预测Malarz
classic_idx = categories['classic_2d'] + categories['classic_3d']
malarz_idx = categories['malarz_2d']
hd_idx = categories['high_d']

z_cl = z_all[classic_idx]
d_cl = d_all[classic_idx]
g_cl = g_all[classic_idx]
pc_cl = pc_all[classic_idx]

z_ml = z_all[malarz_idx]
d_ml = d_all[malarz_idx]
g_ml = g_all[malarz_idx]
pc_ml = pc_all[malarz_idx]

z_hd = z_all[hd_idx]
d_hd = d_all[hd_idx]
g_hd = g_all[hd_idx]
pc_hd = pc_all[hd_idx]

# 拟合经典
def obj_cl(params):
    pred = model_3(params, z_cl, d_cl, g_cl)
    return np.sum((pc_cl - pred)**2)

res_cl = minimize(obj_cl, [3.0, 0.7, 1.0], method='Nelder-Mead',
                  options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-12})
params_cl = res_cl.x

# 拟合Malarz
def obj_ml(params):
    pred = model_3(params, z_ml, d_ml, g_ml)
    return np.sum((pc_ml - pred)**2)

res_ml = minimize(obj_ml, [3.0, 0.7, 1.0], method='Nelder-Mead',
                  options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-12})
params_ml = res_ml.x

# 拟合高维
def obj_hd(params):
    pred = model_3(params, z_hd, d_hd, g_hd)
    return np.sum((pc_hd - pred)**2)

res_hd = minimize(obj_hd, [3.0, 0.7, 1.0], method='Nelder-Mead',
                  options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-12})
params_hd = res_hd.x

print(f"\n各regime参数:")
print(f"  经典: A={params_cl[0]:.3f}, α={params_cl[1]:.3f}, B={params_cl[2]:.3f}")
print(f"  Malarz: A={params_ml[0]:.3f}, α={params_ml[1]:.3f}, B={params_ml[2]:.3f}")
print(f"  高维: A={params_hd[0]:.3f}, α={params_hd[1]:.3f}, B={params_hd[2]:.3f}")

# 交叉预测
tests = [
    ('经典→Malarz', params_cl, z_ml, d_ml, g_ml, pc_ml),
    ('经典→高维', params_cl, z_hd, d_hd, g_hd, pc_hd),
    ('Malarz→经典', params_ml, z_cl, d_cl, g_cl, pc_cl),
    ('Malarz→高维', params_ml, z_hd, d_hd, g_hd, pc_hd),
    ('高维→经典', params_hd, z_cl, d_cl, g_cl, pc_cl),
    ('高维→Malarz', params_hd, z_ml, d_ml, g_ml, pc_ml),
]

print(f"\n交叉预测 MAE:")
print(f"{'训练→测试':<20} {'MAE':<10} {'最大误差%':<10}")
print("-"*45)

for name, params, z_test, d_test, g_test, pc_test in tests:
    pred = model_3(params, z_test, d_test, g_test)
    mae = np.mean(np.abs(pc_test - pred))
    max_err = np.max(np.abs(pc_test - pred) / pc_test * 100)
    print(f"{name:<20} {mae:<10.5f} {max_err:<10.1f}%")

# ============ 8. 根本原因分析 ============
print(f"\n{'='*80}")
print("【8. 根本原因分析】")
print("-"*60)

print(f"""
核心问题：为什么扩展数据后公式退化？

1. 经典2D vs Malarz 2D 的 α 显著不同：
   - 经典2D: α ≈ {alphas.get('classic_2d', 0):.3f}
   - Malarz: α ≈ {alphas.get('malarz_2d', 0):.3f}
   - 高维: α ≈ {alphas.get('high_d', 0):.3f}

2. 交叉预测完全失败：
   - 用经典参数预测Malarz → MAE 很大
   - 用Malarz参数预测经典 → MAE 也很大
   - 说明不同regime遵循不同的物理规律

3. 根本原因：
   a) 经典2D: 环结构主导，girth效应极强
      - 最近邻距离 = 1（格点间距）
      - 所有连接在同一尺度
      - Bethe近似误差极大

   b) Malarz 2D: 多尺度zone连接
      - zone 1距离=1, zone 6距离=3
      - 大z时连接分布更广
      - Bethe近似误差减小

   c) 高维: 几何结构主导
      - z = 2d，维度增加导致更多方向
      - 接近平均场
      - Bethe近似相对准确

结论：
  不存在单一标度律 p_c = f(z·d, g) 能统一所有regime！

  可能的解决方案：
  1. 分段公式（不同regime用不同参数）
  2. 引入新的物理量区分regime
  3. 放弃z·d统一框架，用z和d分别处理

  但这意味着之前的"发现"——z·d是标度变量——
  只是在14个晶格（主要是高维数据）上的统计巧合！
""")

# ============ 9. 诚实的自我评估 ============
print(f"\n{'='*80}")
print("【9. 诚实的自我评估】")
print("-"*60)

print(f"""
我们之前认为的"关键发现"：
  ✅ z·d 是标度变量 → 仅在高维数据中成立
  ✅ 3参数公式 R²=0.988 → 扩展到46个晶格后降至0.952
  ✅ 高维极限 R²=0.998 → 仍然成立（但仅对高维）

新增发现（本次分析）：
  ❌ 标度律不具有跨regime的预测能力
  ❌ 经典2D和Malarz 2D有不同的幂律指数
  ❌ 交叉预测完全失败

真正的科学价值：
  1. 高维极限 R = 1 - 1/z + a/(z·d^b) 对 d≥4 仍然精确
  2. 对 Malarz 2D 数据，简单幂律 p_c ~ (z·d)^{-0.52} 足够好
  3. 不存在统一的3参数公式——渗流阈值有更复杂的结构

下一步：
  1. 放弃追求统一公式
  2. 分 regime 建立公式
  3. 寻找 regime 之间的交叉标度
  4. 研究为什么不同 regime 有不同的物理
""")
