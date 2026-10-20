"""
新角度 I：从"归一化渗流阈值"出发
====================================

核心洞察：
  Bethe 晶格（树，无环）：p_c = 1/(z-1)
  真实晶格：p_c < 1/(z-1)（环促进渗流）

  定义：R = p_c · (z-1)  — "归一化阈值"
    - 树：R = 1
    - 真实晶格：R < 1（偏离程度由环决定）

也许 R 有一个只依赖环结构的普适公式？

策略：
  1. 计算所有晶格的 R = p_c · (z-1)
  2. 寻找 R 与什么物理量相关
  3. 如果 R 有简洁公式 → PRL 级别发现！
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============ 完整数据（含 FCC）============

lattices = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'n_tri': 0, 'pc': 0.697040, 'src': 'exact'},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'n_tri': 0, 'pc': 0.592746, 'src': 'exact'},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'n_tri': 6, 'pc': 0.500000, 'src': 'exact'},
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'n_tri': 2, 'pc': 0.652703, 'src': 'literature'},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'n_tri': 0, 'pc': 0.311600, 'src': 'literature'},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'n_tri': 0, 'pc': 0.245800, 'src': 'literature'},
    {'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'n_tri': 24, 'pc': 0.118800, 'src': 'literature'},
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'n_tri': 0, 'pc': 0.431900, 'src': 'literature'},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'n_tri': 0, 'pc': 0.196737, 'src': 'literature'},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'n_tri': 0, 'pc': 0.140810, 'src': 'literature'},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'n_tri': 0, 'pc': 0.109460, 'src': 'literature'},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'g': 4, 'n_tri': 0, 'pc': 0.089270, 'src': 'literature'},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'g': 4, 'n_tri': 0, 'pc': 0.075050, 'src': 'literature'},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'g': 4, 'n_tri': 0, 'pc': 0.064530, 'src': 'literature'},
]

n = len(lattices)

# ============ 1. 计算归一化阈值 R = p_c · (z-1) ============

print("="*80)
print("【1. 归一化阈值 R = p_c · (z-1)】")
print("="*80)

print(f"\n{'晶格':<15} {'z':<3} {'d':<2} {'p_c':<10} {'Bethe=1/(z-1)':<14} {'R=p_c·(z-1)':<12} {'n_tri':<5} {'g':<2}")
print("-"*80)

for lat in lattices:
    bethe = 1.0 / (lat['z'] - 1)
    R = lat['pc'] * (lat['z'] - 1)
    lat['R'] = R
    lat['bethe'] = bethe
    print(f"{lat['name']:<15} {lat['z']:<3} {lat['d']:<2} {lat['pc']:<10.5f} {bethe:<14.5f} {R:<12.4f} {lat['n_tri']:<5} {lat['g']:<2}")

R_values = np.array([lat['R'] for lat in lattices])
print(f"\nR 统计:")
print(f"  min = {R_values.min():.4f} (fcc)")
print(f"  max = {R_values.max():.4f} (honeycomb)")
print(f"  mean = {R_values.mean():.4f}")

# ============ 2. 计算候选物理量 ============

print("\n" + "="*80)
print("【2. 计算候选物理量】")
print("="*80)

for lat in lattices:
    z = lat['z']
    d = lat['d']
    g = lat['g']
    n_tri = lat['n_tri']

    # 聚类系数
    lat['C'] = 2 * n_tri / (z * (z-1)) if z > 1 else 0

    # 三角形密度（每邻居）
    lat['rho_tri'] = n_tri / z

    # 环密度代理（短环/总可能环）
    # 对于 g=3: rho = n_tri / C(z,2)
    # 对于 g=4: 需要四边形数，但我们没有数据
    # 使用 1/g 作为代理
    lat['ring_proxy'] = 1.0 / g

    # "拓扑紧凑度" = (z-1) / g
    # 物理意义：每个邻居方向上有多少短环
    lat['compactness'] = (z - 1) / g

    # "Bethe 偏离因子" = 1 - R
    lat['deviation'] = 1 - lat['R']

    # 谱隙代理（用 z 和 d）
    # 对于正则图，谱隙 ∝ 1/z（对高维）
    lat['spectral_proxy'] = 1.0 / z

    # 关联长度代理
    # Nachmias-Peres (2008): p_c ~ 1/z + O(1/z^{4/3}) for high d
    lat['np_proxy'] = 1.0 / z

    # "有效维度"代理
    # d_eff = d · (1 - 1/g) — 环减少有效维度
    lat['d_eff'] = d * (1 - 1.0/g)

print("\n新物理量:")
print(f"{'晶格':<15} {'R':<8} {'C':<6} {'ρ_tri':<6} {'1/g':<6} {'(z-1)/g':<9} {'1-R':<8} {'d_eff':<6}")
print("-"*80)
for lat in lattices:
    print(f"{lat['name']:<15} {lat['R']:<8.4f} {lat['C']:<6.3f} {lat['rho_tri']:<6.3f} "
          f"{lat['ring_proxy']:<6.3f} {lat['compactness']:<9.3f} {lat['deviation']:<8.4f} {lat['d_eff']:<6.2f}")

# ============ 3. R 与各物理量的相关性 ============

print("\n" + "="*80)
print("【3. R 与各物理量的相关性】")
print("="*80)

# R 与什么相关？
features = {
    'z': np.array([lat['z'] for lat in lattices]),
    'd': np.array([lat['d'] for lat in lattices]),
    'g': np.array([lat['g'] for lat in lattices]),
    'C': np.array([lat['C'] for lat in lattices]),
    'rho_tri': np.array([lat['rho_tri'] for lat in lattices]),
    '1/g': np.array([lat['ring_proxy'] for lat in lattices]),
    '(z-1)/g': np.array([lat['compactness'] for lat in lattices]),
    '1/z': np.array([lat['spectral_proxy'] for lat in lattices]),
    'd_eff': np.array([lat['d_eff'] for lat in lattices]),
    'log(z)': np.log(np.array([lat['z'] for lat in lattices])),
    'log(d)': np.log(np.array([lat['d'] for lat in lattices])),
}

R_arr = np.array([lat['R'] for lat in lattices])

print(f"\nR 与各物理量的 Pearson 相关系数:")
print(f"{'物理量':<15} {'r':<10} {'p-value':<10} {'强度':<10}")
print("-"*50)

corrs = {}
for name, feat in features.items():
    r, p = pearsonr(R_arr, feat)
    corrs[name] = (r, p)
    strength = "强" if abs(r) > 0.7 else "中" if abs(r) > 0.4 else "弱"
    print(f"{name:<15} {r:<+10.4f} {p:<10.4f} {strength:<10}")

# 排序
print(f"\n最强相关（|r|排序）:")
for name, (r, p) in sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True):
    print(f"  R vs {name:<12}: r = {r:+.4f}, p = {p:.4f}")

# ============ 4. 寻找 R 的公式 ============

print("\n" + "="*80)
print("【4. 寻找 R = f(?) 的公式】")
print("="*80)

# 思路：R 只依赖于"环结构"
# 最简单的假设：R 只依赖于聚类系数 C
# 或者：R 依赖于 "短环密度"

# 4.1 R vs C 的线性关系
C_arr = features['C']
slope_C, intercept_C = np.polyfit(C_arr, R_arr, 1)
pred_C = slope_C * C_arr + intercept_C
ss_res_C = np.sum((R_arr - pred_C)**2)
ss_tot_R = np.sum((R_arr - np.mean(R_arr))**2)
R2_C = 1 - ss_res_C / ss_tot_R
print(f"\n4.1 R = {slope_C:.4f}·C + {intercept_C:.4f}")
print(f"    R² = {R2_C:.4f}")

# 4.2 R vs rho_tri
rho_tri_arr = features['rho_tri']
slope_rt, intercept_rt = np.polyfit(rho_tri_arr, R_arr, 1)
pred_rt = slope_rt * rho_tri_arr + intercept_rt
R2_rt = 1 - np.sum((R_arr - pred_rt)**2) / ss_tot_R
print(f"\n4.2 R = {slope_rt:.4f}·ρ_tri + {intercept_rt:.4f}")
print(f"    R² = {R2_rt:.4f}")

# 4.3 R vs 1/g
inv_g_arr = features['1/g']
slope_ig, intercept_ig = np.polyfit(inv_g_arr, R_arr, 1)
pred_ig = slope_ig * inv_g_arr + intercept_ig
R2_ig = 1 - np.sum((R_arr - pred_ig)**2) / ss_tot_R
print(f"\n4.3 R = {slope_ig:.4f}·(1/g) + {intercept_ig:.4f}")
print(f"    R² = {R2_ig:.4f}")

# 4.4 R vs (z-1)/g
compact_arr = features['(z-1)/g']
slope_co, intercept_co = np.polyfit(compact_arr, R_arr, 1)
pred_co = slope_co * compact_arr + intercept_co
R2_co = 1 - np.sum((R_arr - pred_co)**2) / ss_tot_R
print(f"\n4.4 R = {slope_co:.4f}·(z-1)/g + {intercept_co:.4f}")
print(f"    R² = {R2_co:.4f}")

# 4.5 R vs C + 1/g（线性组合）
from numpy.linalg import lstsq
X = np.column_stack([C_arr, inv_g_arr, np.ones(n)])
beta, _, _, _ = lstsq(X, R_arr, rcond=None)
pred_combo = X @ beta
R2_combo = 1 - np.sum((R_arr - pred_combo)**2) / ss_tot_R
print(f"\n4.5 R = {beta[0]:.4f}·C + {beta[1]:.4f}·(1/g) + {beta[2]:.4f}")
print(f"    R² = {R2_combo:.4f}")

# 4.6 R vs C + rho_tri + 1/g（多元）
X3 = np.column_stack([C_arr, rho_tri_arr, inv_g_arr, np.ones(n)])
beta3, _, _, _ = lstsq(X3, R_arr, rcond=None)
pred_3 = X3 @ beta3
R2_3 = 1 - np.sum((R_arr - pred_3)**2) / ss_tot_R
print(f"\n4.6 R = {beta3[0]:.4f}·C + {beta3[1]:.4f}·ρ_tri + {beta3[2]:.4f}·(1/g) + {beta3[3]:.4f}")
print(f"    R² = {R2_3:.4f}")

# ============ 5. 更好的物理量：Bethe 偏离因子 ============

print("\n" + "="*80)
print("【5. 分析 Bethe 偏离因子 (1-R)】")
print("="*80)

dev_arr = 1 - R_arr
print(f"\n(1-R) 表示'环的渗流促进效应'：")
print(f"{'晶格':<15} {'R':<8} {'1-R':<8} {'C':<6} {'n_tri':<5} 分析")
print("-"*70)
for lat in lattices:
    analysis = ""
    if lat['R'] < 0.5:
        analysis = "极强促进"
    elif lat['R'] < 0.7:
        analysis = "强促进"
    elif lat['R'] < 0.85:
        analysis = "中等促进"
    elif lat['R'] < 0.95:
        analysis = "弱促进"
    else:
        analysis = "几乎无环"
    print(f"{lat['name']:<15} {lat['R']:<8.4f} {1-lat['R']:<8.4f} {lat['C']:<6.3f} {lat['n_tri']:<5} {analysis}")

# 关键洞察：1-R 与 C 的关系？
print(f"\n(1-R) vs C 的散点:")
for lat in lattices:
    bar = "█" * int((1 - lat['R']) * 50)
    print(f"  {lat['name']:<15} 1-R={1-lat['R']:.4f} C={lat['C']:.3f} {bar}")

# ============ 6. 关键测试：高维极限 ============

print("\n" + "="*80)
print("【6. 高维极限行为（d→∞）】")
print("="*80)

print("\n理论预期（平均场）:")
print("  d→∞ 时，p_c ~ 1/z → R = p_c·(z-1) ~ (z-1)/z = 1 - 1/z")

print(f"\n实际高维数据:")
for lat in lattices:
    if lat['d'] >= 4:
        R_expected = 1 - 1.0/lat['z']
        print(f"  {lat['name']}: d={lat['d']}, z={lat['z']}, R={lat['R']:.4f}, "
              f"理论R≈{R_expected:.4f}, 差异={lat['R']-R_expected:+.4f}")

# 高维时 R 的趋势
high_d = [lat for lat in lattices if lat['d'] >= 4]
if high_d:
    z_high = np.array([lat['z'] for lat in high_d])
    R_high = np.array([lat['R'] for lat in high_d])
    theory_high = 1 - 1.0/z_high

    print(f"\n高维 R vs 理论 (1-1/z) 的偏差:")
    print(f"  平均偏差 = {np.mean(R_high - theory_high):.4f}")
    print(f"  标准差 = {np.std(R_high - theory_high):.4f}")

    # 偏差与什么相关？
    d_high = np.array([lat['d'] for lat in high_d])
    corr_d, p_d = pearsonr(d_high, R_high - theory_high)
    print(f"\n  偏差与 d 的相关: r={corr_d:.4f}, p={p_d:.4f}")

    # 偏差是否随 1/d 衰减？
    if corr_d < -0.5:
        print(f"  ✓ 偏差随 d 增大而减小（趋于平均场）")

    # 拟合 R = 1 - 1/z - a/(z·d^b)
    def high_d_model(params, z, d):
        a, b = params
        return 1 - 1.0/z - a / (z * np.power(d, b))

    def obj(params):
        pred = high_d_model(params, z_high, d_high)
        return np.sum((R_high - pred)**2)

    res = minimize(obj, [1.0, 1.0], method='Nelder-Mead')
    a_fit, b_fit = res.x
    pred_fit = high_d_model(res.x, z_high, d_high)
    ss_res = np.sum((R_high - pred_fit)**2)
    ss_tot = np.sum((R_high - np.mean(R_high))**2)
    R2_high = 1 - ss_res / ss_tot

    print(f"\n  高维模型: R = 1 - 1/z - {a_fit:.4f}/(z·d^{b_fit:.4f})")
    print(f"  R² = {R2_high:.4f}")

# ============ 7. 统一框架探索 ============

print("\n" + "="*80)
print("【7. 统一框架：R 的通用公式】")
print("="*80)

print("""
假设：R = 1 - Δ(z, d, g, ...)

其中 Δ 是"环促进效应"，满足：
  - Δ → 0 当 g → ∞（无环）
  - Δ → 1 - 1/z 当 d → ∞（平均场）
  - Δ 依赖于聚类结构

候选形式：
  A: Δ = a · C^b
  B: Δ = a · (1/g)^b
  C: Δ = a · C / z^b
  D: Δ = a · n_tri / (z · d^b)
  E: Δ = a / (z · g^b)
""")

z_all = np.array([lat['z'] for lat in lattices])
d_all = np.array([lat['d'] for lat in lattices])
g_all = np.array([lat['g'] for lat in lattices])
C_all = np.array([lat['C'] for lat in lattices])
ntri_all = np.array([lat['n_tri'] for lat in lattices])
R_all = R_arr
dev_all = 1 - R_all

def fit_dev(formula_func, params0):
    def obj(params):
        pred = formula_func(params, z_all, d_all, g_all, C_all, ntri_all)
        return np.sum((dev_all - pred)**2)
    res = minimize(obj, params0, method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
    pred = formula_func(res.x, z_all, d_all, g_all, C_all, ntri_all)
    ss_res = np.sum((dev_all - pred)**2)
    R2 = 1 - ss_res / np.sum((dev_all - np.mean(dev_all))**2)
    return res.x, R2, pred

# 候选模型
def dev_A(params, z, d, g, C, ntri):
    a, b = params
    return a * np.power(np.maximum(C, 1e-10), b)

def dev_B(params, z, d, g, C, ntri):
    a, b = params
    return a * np.power(1.0/g, b)

def dev_C(params, z, d, g, C, ntri):
    a, b = params
    return a * C / np.power(z, b)

def dev_D(params, z, d, g, C, ntri):
    a, b = params
    return a * ntri / (z * np.power(d, b))

def dev_E(params, z, d, g, C, ntri):
    a, b = params
    return a / (z * np.power(g, b))

def dev_F(params, z, d, g, C, ntri):
    a, b, c = params
    return a * C / np.power(z, b) + c / (z * np.power(d, 1.0))

models = [
    ('A: Δ = a·C^b', dev_A, [0.5, 0.5]),
    ('B: Δ = a·(1/g)^b', dev_B, [0.5, 0.5]),
    ('C: Δ = a·C/z^b', dev_C, [0.5, 0.5]),
    ('D: Δ = a·n_tri/(z·d^b)', dev_D, [0.5, 0.5]),
    ('E: Δ = a/(z·g^b)', dev_E, [0.5, 0.5]),
    ('F: 组合', dev_F, [0.5, 0.5, 0.5]),
]

print(f"\n模型拟合结果:")
print(f"{'模型':<25} {'R²':<10} {'参数':<20}")
print("-"*60)

best_R2 = -np.inf
best_model = None

for name, func, init in models:
    try:
        params, R2, pred = fit_dev(func, init)
        param_str = ", ".join(f"{p:.4f}" for p in params)
        print(f"{name:<25} {R2:<10.6f} {param_str:<20}")
        if R2 > best_R2:
            best_R2 = R2
            best_model = (name, func, params, pred)
    except Exception as e:
        print(f"{name:<25} 失败: {e}")

if best_model:
    name, func, params, pred = best_model
    print(f"\n最佳模型: {name}")
    print(f"R² = {best_R2:.6f}")
    print(f"参数: {params}")

    # 转换为 p_c 公式
    print(f"\n等价 p_c 公式:")
    print(f"  p_c = (1-R) / (z-1)")
    print(f"  R = 1 - {name.split(': ')[1]}")

    # 逐点误差
    print(f"\n逐点比较:")
    print(f"{'晶格':<15} {'p_c真实':<10} {'p_c预测':<10} {'误差':<8} {'相对误差':<10}")
    print("-"*60)

    R_pred = 1 - pred
    pc_pred = R_pred / (z_all - 1)
    pc_true = np.array([lat['pc'] for lat in lattices])

    for i, lat in enumerate(lattices):
        err = abs(pc_true[i] - pc_pred[i])
        err_rel = err / pc_true[i] * 100
        flag = " ⚠️" if err_rel > 20 else ""
        print(f"{lat['name']:<15} {pc_true[i]:<10.5f} {pc_pred[i]:<10.5f} {err:<8.5f} {err_rel:<10.2f}%{flag}")

# ============ 8. 物理意义解读 ============

print("\n" + "="*80)
print("【8. 物理意义解读】")
print("="*80)

print("""
如果 Δ = f(环结构) 有简洁形式：

物理意义：
  p_c = (1 - Δ) / (z-1)
      = 1/(z-1) · (1 - Δ)

  - 1/(z-1) 是 Bethe 近似（树图）
  - (1 - Δ) 是"环修正因子"
  - Δ 量化了环对渗流的促进效应

如果 Δ 只依赖于某个简洁的物理量，那就是普适定律！

关键问题：
  - FCC 是否符合这个框架？
  - 高维极限 Δ → 1/z？
  - Δ 是否与"分形维数"或"谱隙"有关？
""")

print("\n" + "="*80)
print("探索完成")
print("="*80)
