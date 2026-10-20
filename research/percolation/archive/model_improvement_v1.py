from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
模型改进实验：修复g=3系统性偏差
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

print("=" * 80)
print("  模型改进实验：修复g=3系统性偏差")
print("=" * 80)

# 加载数据
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values
n = len(pc)

# 当前基线
def model_baseline(params, z, d, g):
    a, alpha, gamma, z0 = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

baseline_params = np.array([12.3346, 0.3368, 0.8477, 9.6171])
pred_baseline = model_baseline(baseline_params, z, d, g)
errors_baseline = np.abs((pc - pred_baseline) / pc) * 100
mape_baseline = np.mean(errors_baseline)

print(f"\n基线模型: MAPE = {mape_baseline:.2f}%")

# ============================================================
# 模型H1: g=3指示函数修正
# ============================================================
print("\n" + "=" * 80)
print("H1: g=3指示函数修正")
print("=" * 80)

def model_H1(params, z, d, g):
    """在基线上添加g=3的衰减因子"""
    a, alpha, gamma, z0, c_g3 = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    # g=3时乘以 (1 + c_g3)，c_g3 < 0 会减少修正
    correction *= (1 + c_g3 * (g == 3))
    return (1 + correction) / (z - 1)

best_H1 = None
best_ss_H1 = np.inf
for _ in range(100):
    p0 = [12.3, 0.34, 0.85, 9.6, np.random.uniform(-0.5, 0.0)]
    res = minimize(lambda p: np.sum((pc - model_H1(p, z, d, g))**2),
                  p0, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_ss_H1:
        best_ss_H1 = res.fun
        best_H1 = res.x

pred_H1 = model_H1(best_H1, z, d, g)
errors_H1 = np.abs((pc - pred_H1) / pc) * 100
mape_H1 = np.mean(errors_H1)
aic_H1 = n * np.log(best_ss_H1 / n) + 2 * 5

print(f"H1参数: a={best_H1[0]:.2f}, α={best_H1[1]:.3f}, γ={best_H1[2]:.3f}, z₀={best_H1[2]:.2f}, c_g3={best_H1[4]:.3f}")
print(f"H1 MAPE: {mape_H1:.2f}% (基线: {mape_baseline:.2f}%, Δ={mape_baseline-mape_H1:+.2f}%)")
print(f"H1 AIC: {aic_H1:.1f}")

# 分类误差
mask_d3 = d == 3
mask_d3_g3 = (d == 3) & (g == 3)
mask_hc8 = (z == 2*d) & (d >= 8)
mask_sq_g3 = (d == 2) & (g == 3) & (pd.Series(names).str.contains('sq-', na=False))

print(f"\nH1分类误差:")
print(f"  d=3, g=3: {np.mean(errors_H1[mask_d3_g3]):.2f}% (基线: {np.mean(errors_baseline[mask_d3_g3]):.2f}%)")
print(f"  d=3 (全部): {np.mean(errors_H1[mask_d3]):.2f}% (基线: {np.mean(errors_baseline[mask_d3]):.2f}%)")
print(f"  sq g=3: {np.mean(errors_H1[mask_sq_g3]):.2f}% (基线: {np.mean(errors_baseline[mask_sq_g3]):.2f}%)")
print(f"  超立方d≥8: {np.mean(errors_H1[mask_hc8]):.2f}% (基线: {np.mean(errors_baseline[mask_hc8]):.2f}%)")

# ============================================================
# 模型H2: d-g交叉项
# ============================================================
print("\n" + "=" * 80)
print("H2: d-g交叉项")
print("=" * 80)

def model_H2(params, z, d, g):
    """添加d/g交叉项"""
    a, alpha, gamma, z0, c_dg = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    # 交叉项：d小g小时额外衰减
    correction *= (1 + c_dg / (d * g))
    return (1 + correction) / (z - 1)

best_H2 = None
best_ss_H2 = np.inf
for _ in range(100):
    p0 = [12.3, 0.34, 0.85, 9.6, np.random.uniform(-2.0, 0.0)]
    res = minimize(lambda p: np.sum((pc - model_H2(p, z, d, g))**2),
                  p0, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_ss_H2:
        best_ss_H2 = res.fun
        best_H2 = res.x

pred_H2 = model_H2(best_H2, z, d, g)
errors_H2 = np.abs((pc - pred_H2) / pc) * 100
mape_H2 = np.mean(errors_H2)
aic_H2 = n * np.log(best_ss_H2 / n) + 2 * 5

print(f"H2参数: a={best_H2[0]:.2f}, α={best_H2[1]:.3f}, γ={best_H2[2]:.3f}, z₀={best_H2[3]:.2f}, c_dg={best_H2[4]:.3f}")
print(f"H2 MAPE: {mape_H2:.2f}% (基线: {mape_baseline:.2f}%, Δ={mape_baseline-mape_H2:+.2f}%)")
print(f"H2 AIC: {aic_H2:.1f}")

print(f"\nH2分类误差:")
print(f"  d=3, g=3: {np.mean(errors_H2[mask_d3_g3]):.2f}%")
print(f"  d=3 (全部): {np.mean(errors_H2[mask_d3]):.2f}%")
print(f"  sq g=3: {np.mean(errors_H2[mask_sq_g3]):.2f}%")
print(f"  超立方d≥8: {np.mean(errors_H2[mask_hc8]):.2f}%")

# ============================================================
# 模型H3: 非线性g依赖 (g=3特殊处理)
# ============================================================
print("\n" + "=" * 80)
print("H3: 非线性g依赖")
print("=" * 80)

def model_H3(params, z, d, g):
    """g依赖改为 g^γ + c·I(g=3)"""
    a, alpha, gamma, z0, c_g3 = params
    g_factor = g**gamma + c_g3 * (g == 3)
    correction = (a * np.exp(-alpha * d) / g_factor) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_H3 = None
best_ss_H3 = np.inf
for _ in range(100):
    p0 = [12.3, 0.34, 0.85, 9.6, np.random.uniform(0.0, 2.0)]
    res = minimize(lambda p: np.sum((pc - model_H3(p, z, d, g))**2),
                  p0, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_ss_H3:
        best_ss_H3 = res.fun
        best_H3 = res.x

pred_H3 = model_H3(best_H3, z, d, g)
errors_H3 = np.abs((pc - pred_H3) / pc) * 100
mape_H3 = np.mean(errors_H3)
aic_H3 = n * np.log(best_ss_H3 / n) + 2 * 5

print(f"H3参数: a={best_H3[0]:.2f}, α={best_H3[1]:.3f}, γ={best_H3[2]:.3f}, z₀={best_H3[3]:.2f}, c_g3={best_H3[4]:.3f}")
print(f"H3 MAPE: {mape_H3:.2f}% (基线: {mape_baseline:.2f}%, Δ={mape_baseline-mape_H3:+.2f}%)")
print(f"H3 AIC: {aic_H3:.1f}")

print(f"\nH3分类误差:")
print(f"  d=3, g=3: {np.mean(errors_H3[mask_d3_g3]):.2f}%")
print(f"  d=3 (全部): {np.mean(errors_H3[mask_d3]):.2f}%")
print(f"  sq g=3: {np.mean(errors_H3[mask_sq_g3]):.2f}%")
print(f"  超立方d≥8: {np.mean(errors_H3[mask_hc8]):.2f}%")

# ============================================================
# 模型H4: 分段α (d≤3 vs d≥4)
# ============================================================
print("\n" + "=" * 80)
print("H4: 分段α (d≤3 vs d≥4)")
print("=" * 80)

def model_H4(params, z, d, g):
    """d≤3用α₁, d≥4用α₂"""
    a, alpha1, alpha2, gamma, z0 = params
    alpha_eff = np.where(d <= 3, alpha1, alpha2)
    correction = (a * np.exp(-alpha_eff * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_H4 = None
best_ss_H4 = np.inf
for _ in range(100):
    p0 = [12.3, np.random.uniform(0.1, 0.3), np.random.uniform(0.3, 0.6), 0.85, 9.6]
    res = minimize(lambda p: np.sum((pc - model_H4(p, z, d, g))**2),
                  p0, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_ss_H4:
        best_ss_H4 = res.fun
        best_H4 = res.x

pred_H4 = model_H4(best_H4, z, d, g)
errors_H4 = np.abs((pc - pred_H4) / pc) * 100
mape_H4 = np.mean(errors_H4)
aic_H4 = n * np.log(best_ss_H4 / n) + 2 * 5

print(f"H4参数: a={best_H4[0]:.2f}, α₁={best_H4[1]:.3f} (d≤3), α₂={best_H4[2]:.3f} (d≥4), γ={best_H4[3]:.3f}, z₀={best_H4[4]:.2f}")
print(f"H4 MAPE: {mape_H4:.2f}% (基线: {mape_baseline:.2f}%, Δ={mape_baseline-mape_H4:+.2f}%)")
print(f"H4 AIC: {aic_H4:.1f}")

print(f"\nH4分类误差:")
print(f"  d=3, g=3: {np.mean(errors_H4[mask_d3_g3]):.2f}%")
print(f"  d=3 (全部): {np.mean(errors_H4[mask_d3]):.2f}%")
print(f"  sq g=3: {np.mean(errors_H4[mask_sq_g3]):.2f}%")
print(f"  超立方d≥8: {np.mean(errors_H4[mask_hc8]):.2f}%")

# ============================================================
# 模型H5: exp(-αd - β/g²) + g=3修正
# ============================================================
print("\n" + "=" * 80)
print("H5: exp(-αd - β/g²)")
print("=" * 80)

def model_H5(params, z, d, g):
    """g也指数衰减"""
    a, alpha, beta, gamma, z0 = params
    correction = a * np.exp(-alpha * d - beta / g**2) / g**gamma * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_H5 = None
best_ss_H5 = np.inf
for _ in range(100):
    p0 = [12.3, np.random.uniform(0.1, 0.5), np.random.uniform(0.0, 3.0), 0.85, 9.6]
    res = minimize(lambda p: np.sum((pc - model_H5(p, z, d, g))**2),
                  p0, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_ss_H5:
        best_ss_H5 = res.fun
        best_H5 = res.x

pred_H5 = model_H5(best_H5, z, d, g)
errors_H5 = np.abs((pc - pred_H5) / pc) * 100
mape_H5 = np.mean(errors_H5)
aic_H5 = n * np.log(best_ss_H5 / n) + 2 * 5

print(f"H5参数: a={best_H5[0]:.2f}, α={best_H5[1]:.3f}, β={best_H5[2]:.3f}, γ={best_H5[3]:.3f}, z₀={best_H5[4]:.2f}")
print(f"H5 MAPE: {mape_H5:.2f}% (基线: {mape_baseline:.2f}%, Δ={mape_baseline-mape_H5:+.2f}%)")
print(f"H5 AIC: {aic_H5:.1f}")

print(f"\nH5分类误差:")
print(f"  d=3, g=3: {np.mean(errors_H5[mask_d3_g3]):.2f}%")
print(f"  d=3 (全部): {np.mean(errors_H5[mask_d3]):.2f}%")
print(f"  sq g=3: {np.mean(errors_H5[mask_sq_g3]):.2f}%")
print(f"  超立方d≥8: {np.mean(errors_H5[mask_hc8]):.2f}%")

# ============================================================
# 综合对比
# ============================================================
print("\n" + "=" * 80)
print("综合对比")
print("=" * 80)

models = [
    ('基线 (4参数)', mape_baseline, np.sum((pc - pred_baseline)**2), 4),
    ('H1: g=3指示 (5参数)', mape_H1, best_ss_H1, 5),
    ('H2: d/g交叉 (5参数)', mape_H2, best_ss_H2, 5),
    ('H3: g非线性 (5参数)', mape_H3, best_ss_H3, 5),
    ('H4: 分段α (5参数)', mape_H4, best_ss_H4, 5),
    ('H5: exp(-β/g²) (5参数)', mape_H5, best_ss_H5, 5),
]

aic_baseline = n * np.log(np.sum((pc - pred_baseline)**2) / n) + 2 * 4

print(f"\n{'模型':<25} {'MAPE':>8} {'AIC':>10} {'ΔAIC':>8} {'d=3,g=3':>10} {'超立方d≥8':>12}")
print("-" * 80)
for name, mape, ssr, k in models:
    aic = n * np.log(ssr / n) + 2 * k
    delta_aic = aic - aic_baseline
    err_d3g3 = np.mean(np.abs((pc[mask_d3_g3] - {'基线': pred_baseline, 'H1': pred_H1, 'H2': pred_H2, 'H3': pred_H3, 'H4': pred_H4, 'H5': pred_H5}[name.split(':')[0].strip()][mask_d3_g3]) / pc[mask_d3_g3]) * 100)
    err_hc8 = np.mean(np.abs((pc[mask_hc8] - {'基线': pred_baseline, 'H1': pred_H1, 'H2': pred_H2, 'H3': pred_H3, 'H4': pred_H4, 'H5': pred_H5}[name.split(':')[0].strip()][mask_hc8]) / pc[mask_hc8]) * 100)
    print(f"{name:<25} {mape:>7.2f}% {aic:>10.1f} {delta_aic:>+8.1f} {err_d3g3:>9.2f}% {err_hc8:>11.2f}%")

print("\n" + "=" * 80)
print("实验完成")
print("=" * 80)
