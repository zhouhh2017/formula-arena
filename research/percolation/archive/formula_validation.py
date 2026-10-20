from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
最终公式验证与物理分析
========================
  1. 对已知文献晶格做预测验证
  2. 残差结构分析
  3. 公式物理推导
  4. 与高维极限公式的对比
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

# ============================================================
# 加载数据
# ============================================================
df = pd.read_csv(f'{OUT}/complete_dataset_v2.csv')
df = df[df['d'] > 0].copy().reset_index(drop=True)
n = len(df)
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values

print("=" * 75)
print("【最终公式验证与物理分析】")
print("=" * 75)

# ============================================================
# 核心公式
# ============================================================
# Formula A: 1参数 Bethe·(1+a·z^0.5·d^-1.5·g^-1)
a_A = 4.026
def formula_A(z, d, g):
    return 1/(z-1) * (1 + a_A * np.sqrt(z) / (np.power(d, 1.5) * g))

# Formula B: RM7 (4参数)
p_rm7 = np.array([3.385, -0.515, 1.488, 0.894])
def formula_B(z, d, g):
    a, b, c, e = p_rm7
    return 1/(z-1) * (1 + a / (np.power(z, b) * np.power(d, c) * np.power(g, e)))

# Formula C: 高维极限 (之前发现的)
# R = 1 - 1/z + 7.19/(z·d^0.42)
def formula_C(z, d, g):
    return 1 - 1/z + 7.19 / (z * np.power(d, 0.42))

# ============================================================
# Part 1: 对已知文献晶格做预测
# ============================================================
print(f"\n{'='*75}")
print("【1. 对已知文献晶格的预测（不在训练集中）】")
print(f"{'='*75}")

# 这些是文献中的高精度值，但不在我们的训练数据中
validation_lattices = [
    # 2D Archimedean (已知精确/高精度值)
    # 这些如果MC模拟出来可以验证
    {'name': 'moore_3_6', 'z': 3, 'd': 2, 'g': 6, 'pc': None, 'note': 'Moore bound graph'},
    # 3D lattices (文献值)
    {'name': 'hcp', 'z': 12, 'd': 3, 'g': 3, 'pc': 0.1479, 'note': 'hcp site, Malarz 2022'},
    {'name': 'sc_nn2', 'z': 18, 'd': 3, 'g': 3, 'pc': None, 'note': 'sc+2nd nn'},
    # 4D
    {'name': '4d_fcc', 'z': 12, 'd': 4, 'g': 3, 'pc': None, 'note': '4D close-packed'},
]

print(f"\n用公式A预测已知晶格:")
print(f"{'晶格':<15} {'z':>3} {'d':>2} {'g':>2} {'p_c文献':>8} {'p_c_A':>8} {'p_c_B':>8}")
print("-" * 55)

for lat in validation_lattices:
    if lat['pc'] is not None:
        pc_a = formula_A(lat['z'], lat['d'], lat['g'])
        pc_b = formula_B(lat['z'], lat['d'], lat['g'])
        err_a = abs(lat['pc'] - pc_a) / lat['pc'] * 100
        print(f"{lat['name']:<15} {lat['z']:>3} {lat['d']:>2} {lat['g']:>2} "
              f"{lat['pc']:>8.4f} {pc_a:>8.4f} {pc_b:>8.4f}  (A误差={err_a:.1f}%)")

# ============================================================
# Part 2: 残差结构分析
# ============================================================
print(f"\n{'='*75}")
print("【2. 残差结构分析】")
print(f"{'='*75}")

pred_A = formula_A(z, d, g)
residuals = pc - pred_A
rel_errors = residuals / pc * 100

# 按类别分析
categories = {
    'd=2 (Malarz)': d == 2,
    'd=2 (经典)': (d == 2) & (np.isin(names, ['honeycomb', 'square', 'triangular', 'kagome'])),
    'd=3': d == 3,
    'd≥4': d >= 4,
}

print(f"\n残差分布:")
print(f"{'类别':<20} {'n':>3} {'均值%':>8} {'std%':>8} {'max%':>8}")
print("-" * 55)

for cat_name, mask in categories.items():
    if cat_name == 'd=2 (经典)':
        mask = np.isin(names, ['honeycomb', 'square', 'triangular', 'kagome'])
    err_cat = rel_errors[mask]
    print(f"{cat_name:<20} {np.sum(mask):>3} {np.mean(err_cat):>+8.2f} {np.std(err_cat):>8.2f} {np.max(np.abs(err_cat)):>8.2f}")

# 残差 vs z·d
zd = z * d
print(f"\n残差 vs z·d:")
for i in range(4):
    lo = i * 15
    hi = (i+1) * 15
    mask = (zd >= lo) & (zd < hi)
    if np.sum(mask) > 0:
        print(f"  z·d ∈ [{lo},{hi}): n={np.sum(mask)}, mean_err={np.mean(rel_errors[mask]):>+.2f}%, std={np.std(rel_errors[mask]):.2f}%")

# 残差 vs girth
print(f"\n残差 vs girth:")
for g_val in [3, 4, 6]:
    mask = g == g_val
    print(f"  g={g_val}: n={np.sum(mask)}, mean_err={np.mean(rel_errors[mask]):>+.2f}%, std={np.std(rel_errors[mask]):.2f}%")

# 残差 vs d
print(f"\n残差 vs d:")
for d_val in sorted(np.unique(d)):
    mask = d == d_val
    print(f"  d={int(d_val)}: n={np.sum(mask)}, mean_err={np.mean(rel_errors[mask]):>+.2f}%, std={np.std(rel_errors[mask]):.2f}%")

# ============================================================
# Part 3: 公式物理推导
# ============================================================
print(f"\n{'='*75}")
print("【3. 公式物理推导尝试】")
print(f"{'='*75}")

print(f"""
最佳公式:
  p_c = 1/(z-1) · (1 + 4.026 · √z / (d^{{3/2}} · g))

等价形式:
  R = p_c · (z-1) = 1 + 4.026 · √z / (d^{{3/2}} · g)
  R - 1 = 4.026 · √z / (d^{{3/2}} · g)

物理解读:

  1. Bethe 基线:
     p_c^Bethe = 1/(z-1)
     对应于树状图（无环）的渗流阈值

  2. 修正项的物理含义:
     ΔR = R - 1 ∝ √z / (d^{{3/2}} · g)

     a) z 依赖性 (∝ √z):
        - 在固定 d 下，增加 z 导致更大的 Bethe 偏离
        - √z 可能与随机游走的回转半径有关
        - 或者与配位数的涨落有关

     b) d 依赖性 (∝ d^{{-3/2}}):
        - 维度的 3/2 次方 — 与 d 维空间的态密度有关？
        - 在 d 维中，随机游走返回原点的概率 ∝ d^{{-1}}
        - d^{{3/2}} 可能与 d 维 Green 函数的奇异性有关

     c) g 依赖性 (∝ 1/g):
        - 更短的环（小 g）→ 更大的修正
        - 1/g 可能与环的"权重"成正比
        - 每个环对 Bethe 偏离的贡献 ~ 1/g

  3. 与已知理论的联系:
     - Nachmias-Peres: p_c = 1/z + 5/(2z^{{4/3}})
       → 对应于大 z 极限，指数 4/3 ≠ 3/2
     - 我们的公式在 d=2, z=28 时给出 R ≈ 3.87
       而 NP 预测 R ≈ 1 + 5/(2·28^{{1/3}}) ≈ 1.85
       → 说明 NP 在有限 d 下不适用

  4. 无量纲分析:
     √z / (d^{{3/2}} · g) 可以写为:
     (z/d^3)^{{1/2}} · d^0 / g = ρ^{{1/2}} / g
     其中 ρ = z/d^3 是"有效密度"

     或者: z^{{1/2}} · d^{{-3/2}} · g^{{-1}}
     = (z/d) · d^{{-1/2}} · (1/d) · g^{{-1}}
     = (z/d) / (√d · g · d)
     嗯...不太有清晰的物理含义
""")

# ============================================================
# Part 4: 与高维极限的对比
# ============================================================
print(f"{'='*75}")
print("【4. 公式 A vs 高维极限公式 — 渐近行为】")
print(f"{'='*75}")

print(f"""
公式 A:  R = 1 + 4.026 · z^{{1/2}} / (d^{{3/2}} · g)
高维公式: R = 1 - 1/z + 7.19 / (z · d^{{0.42}})

对于超立方晶格 (z=2d, g=4):
公式 A:  R = 1 + 4.026 · √(2d) / (d^{{3/2}} · 4)
             = 1 + 4.026 · √2 / (4d)
             = 1 + 1.424/d

高维公式: R = 1 - 1/(2d) + 7.19/(2d · d^{{0.42}})
              = 1 - 1/(2d) + 3.595/d^{{1.42}}

对于 d→∞:
  公式 A: R-1 ~ 1.424/d
  高维公式: R-1 ~ 3.595/d^{{1.42}} (更快衰减)

但两个公式在 d→∞ 时都给出 R→1 (Bethe 极限)。
""")

# 数值对比
print(f"超立方晶格 (z=2d, g=4) 的 R 值对比:")
print(f"{'d':>3} {'z':>3} {'R_实际':>8} {'R_A':>8} {'R_高维':>8} {'A误差%':>8}")
print("-" * 50)

for d_val in range(3, 12):
    z_val = 2 * d_val
    g_val = 4

    # 实际值（对d≤9使用文献值）
    if d_val <= 9:
        mask = (d == d_val) & (z == z_val)
        if np.sum(mask) > 0:
            R_actual = pc[mask][0] * (z_val - 1)
        else:
            continue
    else:
        break

    R_A = formula_A(z_val, d_val, g_val) * (z_val - 1)
    R_hd = formula_C(z_val, d_val, g_val)
    err_A = abs(R_actual - R_A) / R_actual * 100

    print(f"{d_val:>3} {z_val:>3} {R_actual:>8.4f} {R_A:>8.4f} {R_hd:>8.4f} {err_A:>8.2f}")

# ============================================================
# Part 5: 改进尝试 — 混合公式
# ============================================================
print(f"\n{'='*75}")
print("【5. 改进公式尝试】")
print(f"{'='*75}")

# 思路：在 d 的指数上做微调
# p_c = 1/(z-1) · (1 + a · z^α / (d^β · g^γ))
# 固定 α=0.5, γ=1, 让 β 自由优化

print(f"\n扫描 d 的指数 β (固定 α=0.5, γ=1):")
print(f"{'β':>6} {'a':>8} {'R²':>8} {'LOO_MAPE%':>10}")
print("-" * 40)

from sklearn.model_selection import LeaveOneOut

best_beta = None
best_loo = np.inf

for beta_try in np.arange(0.5, 2.5, 0.1):
    def pred_scan(params, z, d, g):
        a = params[0]
        return 1/(z-1) * (1 + a * np.sqrt(z) / (np.power(d, beta_try) * g))

    def obj_scan(params):
        return np.sum((pc - pred_scan(params, z, d, g))**2)

    res_s = minimize(obj_scan, [4.0], method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
    a_s = res_s.x[0]
    pred_s = pred_scan([a_s], z, d, g)
    R2_s = 1 - np.sum((pc-pred_s)**2)/np.sum((pc-np.mean(pc))**2)

    # LOOCV
    loo = LeaveOneOut()
    loo_errs = []
    for tr_idx, te_idx in loo.split(z):
        def obj_loo(p):
            return np.sum((pc[tr_idx] - pred_scan(p, z[tr_idx], d[tr_idx], g[tr_idx]))**2)
        res_loo = minimize(obj_loo, [a_s], method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        pred_i = pred_scan(res_loo.x, z[te_idx], d[te_idx], g[te_idx])
        loo_errs.append(abs(pc[te_idx[0]] - pred_i[0]) / pc[te_idx[0]])
    loo_mape = np.mean(loo_errs) * 100

    if loo_mape < best_loo:
        best_loo = loo_mape
        best_beta = beta_try

    if abs(beta_try - round(beta_try, 1)) < 0.01:
        print(f"{beta_try:>6.1f} {a_s:>8.3f} {R2_s:>8.4f} {loo_mape:>10.2f}")

print(f"\n最优 β = {best_beta:.1f}, LOOCV MAPE = {best_loo:.2f}%")

# 同样扫描 α
print(f"\n扫描 z 的指数 α (固定 γ=1, β=最优):")
print(f"{'α':>6} {'a':>8} {'R²':>8} {'LOO_MAPE%':>10}")
print("-" * 40)

best_alpha = None
best_loo2 = np.inf

for alpha_try in np.arange(0.0, 1.1, 0.1):
    def pred_scan2(params, z, d, g):
        a = params[0]
        return 1/(z-1) * (1 + a * np.power(z, alpha_try) / (np.power(d, best_beta) * g))

    def obj_scan2(params):
        return np.sum((pc - pred_scan2(params, z, d, g))**2)

    res_s2 = minimize(obj_scan2, [4.0], method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
    a_s2 = res_s2.x[0]
    pred_s2 = pred_scan2([a_s2], z, d, g)
    R2_s2 = 1 - np.sum((pc-pred_s2)**2)/np.sum((pc-np.mean(pc))**2)

    loo = LeaveOneOut()
    loo_errs2 = []
    for tr_idx, te_idx in loo.split(z):
        def obj_loo2(p):
            return np.sum((pc[tr_idx] - pred_scan2(p, z[tr_idx], d[tr_idx], g[tr_idx]))**2)
        res_loo2 = minimize(obj_loo2, [a_s2], method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        pred_i2 = pred_scan2(res_loo2.x, z[te_idx], d[te_idx], g[te_idx])
        loo_errs2.append(abs(pc[te_idx[0]] - pred_i2[0]) / pc[te_idx[0]])
    loo_mape2 = np.mean(loo_errs2) * 100

    if loo_mape2 < best_loo2:
        best_loo2 = loo_mape2
        best_alpha = alpha_try

    print(f"{alpha_try:>6.1f} {a_s2:>8.3f} {R2_s2:>8.4f} {loo_mape2:>10.2f}")

print(f"\n最优 α = {best_alpha:.1f}, LOOCV MAPE = {best_loo2:.2f}%")

# 扫描 γ
print(f"\n扫描 g 的指数 γ (固定 α=最优, β=最优):")
print(f"{'γ':>6} {'a':>8} {'R²':>8} {'LOO_MAPE%':>10}")
print("-" * 40)

best_gamma = None
best_loo3 = np.inf

for gamma_try in np.arange(0.3, 1.6, 0.1):
    def pred_scan3(params, z, d, g):
        a = params[0]
        return 1/(z-1) * (1 + a * np.power(z, best_alpha) / (np.power(d, best_beta) * np.power(g, gamma_try)))

    def obj_scan3(params):
        return np.sum((pc - pred_scan3(params, z, d, g))**2)

    res_s3 = minimize(obj_scan3, [4.0], method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
    a_s3 = res_s3.x[0]
    pred_s3 = pred_scan3([a_s3], z, d, g)
    R2_s3 = 1 - np.sum((pc-pred_s3)**2)/np.sum((pc-np.mean(pc))**2)

    loo = LeaveOneOut()
    loo_errs3 = []
    for tr_idx, te_idx in loo.split(z):
        def obj_loo3(p):
            return np.sum((pc[tr_idx] - pred_scan3(p, z[tr_idx], d[tr_idx], g[tr_idx]))**2)
        res_loo3 = minimize(obj_loo3, [a_s3], method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        pred_i3 = pred_scan3(res_loo3.x, z[te_idx], d[te_idx], g[te_idx])
        loo_errs3.append(abs(pc[te_idx[0]] - pred_i3[0]) / pc[te_idx[0]])
    loo_mape3 = np.mean(loo_errs3) * 100

    if loo_mape3 < best_loo3:
        best_loo3 = loo_mape3
        best_gamma = gamma_try

    print(f"{gamma_try:>6.1f} {a_s3:>8.3f} {R2_s3:>8.4f} {loo_mape3:>10.2f}")

print(f"\n最优 γ = {best_gamma:.1f}, LOOCV MAPE = {best_loo3:.2f}%")

print(f"\n{'='*75}")
print(f"【最终精调结果】")
print(f"{'='*75}")
print(f"  α = {best_alpha:.1f}")
print(f"  β = {best_beta:.1f}")
print(f"  γ = {best_gamma:.1f}")
print(f"  LOOCV MAPE = {best_loo3:.2f}%")
print(f"\n  公式: p_c = 1/(z-1) · (1 + a · z^{best_alpha:.1f} / (d^{best_beta:.1f} · g^{best_gamma:.1f}))")
