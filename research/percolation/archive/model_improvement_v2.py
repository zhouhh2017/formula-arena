from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
模型改进实验 v2: 基于v1结果深入优化
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

# 加载数据
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values
n = len(pc)

# 分类mask
mask_d3 = d == 3
mask_d3_g3 = (d == 3) & (g == 3)
mask_hc8 = (z == 2*d) & (d >= 8)
mask_sq = (d == 2) & (pd.Series(names).str.contains('sq-', na=False))
mask_sq_g3 = mask_sq & (g == 3)
mask_sq_g4 = mask_sq & (g == 4)

def model_baseline(params, z, d, g):
    a, alpha, gamma, z0 = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

bp = np.array([12.3346, 0.3368, 0.8477, 9.6171])
pred_base = model_baseline(bp, z, d, g)
mape_base = np.mean(np.abs((pc - pred_base) / pc) * 100)

# ============================================================
# 通用拟合函数
# ============================================================
def fit_model(model_func, n_params, n_trials=150, p0_gen=None):
    best_p = None
    best_ss = np.inf
    for trial in range(n_trials):
        if p0_gen is not None:
            p0 = p0_gen(trial)
        else:
            p0 = [np.random.uniform(5, 30), np.random.uniform(0.1, 0.5),
                  np.random.uniform(0.3, 1.5), np.random.uniform(3, 20)]
            for _ in range(n_params - 4):
                p0.append(np.random.uniform(-1, 1))
        res = minimize(lambda p: np.sum((pc - model_func(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 80000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < best_ss:
            best_ss = res.fun
            best_p = res.x
    return best_p, best_ss

def eval_model(name, params, model_func, n_params):
    pred = model_func(params, z, d, g)
    errors = np.abs((pc - pred) / pc) * 100
    mape = np.mean(errors)
    ssr = np.sum((pc - pred)**2)
    aic = n * np.log(ssr / n) + 2 * n_params

    print(f"\n{name}:")
    print(f"  MAPE: {mape:.2f}% (基线: {mape_base:.2f}%)")
    print(f"  AIC: {aic:.1f} (基线: -668.3)")
    print(f"  d=3: {np.mean(errors[mask_d3]):.2f}% (基线: {np.mean(errors_base_d3):.2f}%)")
    print(f"  d=3,g=3: {np.mean(errors[mask_d3_g3]):.2f}% (基线: {np.mean(errors_base_d3g3):.2f}%)")
    print(f"  sq g=3: {np.mean(errors[mask_sq_g3]):.2f}% (基线: {np.mean(errors_base_sq_g3):.2f}%)")
    print(f"  超立方d≥8: {np.mean(errors[mask_hc8]):.2f}% (基线: {np.mean(errors_base_hc8):.2f}%)")
    print(f"  参数: {[f'{p:.4f}' for p in params]}")
    return mape, aic, errors

errors_base_d3 = np.abs((pc - pred_base) / pc * 100)[mask_d3]
errors_base_d3g3 = np.abs((pc - pred_base) / pc * 100)[mask_d3_g3]
errors_base_sq_g3 = np.abs((pc - pred_base) / pc * 100)[mask_sq_g3]
errors_base_hc8 = np.abs((pc - pred_base) / pc * 100)[mask_hc8]

print("=" * 80)
print("  模型改进 v2: 基于v1结果深入优化")
print("=" * 80)
print(f"基线 MAPE: {mape_base:.2f}%")

# ============================================================
# H4b: 分段α (确认 + 细化)
# ============================================================
print(f"\n{'='*80}")
print("H4b: 分段α (确认)")

def model_H4(params, z, d, g):
    a, alpha1, alpha2, gamma, z0 = params
    alpha_eff = np.where(d <= 3, alpha1, alpha2)
    correction = (a * np.exp(-alpha_eff * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

p_H4, ss_H4 = fit_model(model_H4, 5, 200,
    lambda t: [13.7, 0.39, 0.34, 0.85, 9.7] if t==0 else
    [np.random.uniform(10, 20), np.random.uniform(0.25, 0.50),
     np.random.uniform(0.25, 0.45), np.random.uniform(0.7, 1.0),
     np.random.uniform(7, 13)])
eval_model("H4: 分段α", p_H4, model_H4, 5)

# ============================================================
# H6: 分段α + d=3常数修正
# ============================================================
print(f"\n{'='*80}")
print("H6: 分段α + d=3乘法修正")

def model_H6(params, z, d, g):
    a, alpha1, alpha2, gamma, z0, f3 = params
    alpha_eff = np.where(d <= 3, alpha1, alpha2)
    correction = (a * np.exp(-alpha_eff * d) / g**gamma) * (1 - np.exp(-z / z0))
    correction *= np.where(d <= 3, f3, 1.0)  # d≤3时修正乘以f3
    return (1 + correction) / (z - 1)

p_H6, ss_H6 = fit_model(model_H6, 6, 200,
    lambda t: [12.3, 0.34, 0.34, 0.85, 9.6, 0.85] if t==0 else
    [np.random.uniform(8, 20), np.random.uniform(0.2, 0.5),
     np.random.uniform(0.2, 0.5), np.random.uniform(0.6, 1.2),
     np.random.uniform(5, 15), np.random.uniform(0.5, 1.2)])
eval_model("H6: 分段α+d=3×f3", p_H6, model_H6, 6)

# ============================================================
# H7: d=3,g=3 特异性修正
# ============================================================
print(f"\n{'='*80}")
print("H7: d=3,g=3特异性修正")

def model_H7(params, z, d, g):
    a, alpha, gamma, z0, c_d3g3 = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    correction *= (1 + c_d3g3 * ((d == 3) & (g == 3)))
    return (1 + correction) / (z - 1)

p_H7, ss_H7 = fit_model(model_H7, 5, 200,
    lambda t: [12.3, 0.34, 0.85, 9.6, -0.15] if t==0 else
    [np.random.uniform(8, 20), np.random.uniform(0.2, 0.5),
     np.random.uniform(0.5, 1.2), np.random.uniform(5, 15),
     np.random.uniform(-0.5, 0.0)])
eval_model("H7: d=3,g=3修正", p_H7, model_H7, 5)

# ============================================================
# H8: 连续α(d) = α₀ + α₁/(1+exp(-k(d-3.5)))
# ============================================================
print(f"\n{'='*80}")
print("H8: 连续α(d) — logistic过渡")

def model_H8(params, z, d, g):
    a, alpha_lo, alpha_hi, k, gamma, z0 = params
    # logistic: d小时α=alpha_lo, d大时α=alpha_hi
    alpha_eff = alpha_lo + (alpha_hi - alpha_lo) / (1 + np.exp(-k * (d - 3.5)))
    correction = (a * np.exp(-alpha_eff * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

p_H8, ss_H8 = fit_model(model_H8, 6, 200,
    lambda t: [12.3, 0.39, 0.30, 2.0, 0.85, 9.6] if t==0 else
    [np.random.uniform(8, 20), np.random.uniform(0.25, 0.55),
     np.random.uniform(0.15, 0.40), np.random.uniform(0.5, 5.0),
     np.random.uniform(0.6, 1.2), np.random.uniform(5, 15)])
eval_model("H8: logistic α(d)", p_H8, model_H8, 6)

# ============================================================
# H9: 2D g=3 特异性修正 (sq系列问题)
# ============================================================
print(f"\n{'='*80}")
print("H9: 2D g=3 特异性修正")

def model_H9(params, z, d, g):
    a, alpha, gamma, z0, c_2d_g3 = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    correction *= (1 + c_2d_g3 * ((d == 2) & (g == 3)))
    return (1 + correction) / (z - 1)

p_H9, ss_H9 = fit_model(model_H9, 5, 200,
    lambda t: [12.3, 0.34, 0.85, 9.6, -0.10] if t==0 else
    [np.random.uniform(8, 20), np.random.uniform(0.2, 0.5),
     np.random.uniform(0.5, 1.2), np.random.uniform(5, 15),
     np.random.uniform(-0.5, 0.0)])
eval_model("H9: 2D,g=3修正", p_H9, model_H9, 5)

# ============================================================
# H10: d=3修正 + 2D g=3修正 (组合)
# ============================================================
print(f"\n{'='*80}")
print("H10: d=3修正 + 2D g=3修正 (组合)")

def model_H10(params, z, d, g):
    a, alpha, gamma, z0, c_d3, c_2d_g3 = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    correction *= (1 + c_d3 * (d == 3)) * (1 + c_2d_g3 * ((d == 2) & (g == 3)))
    return (1 + correction) / (z - 1)

p_H10, ss_H10 = fit_model(model_H10, 6, 200,
    lambda t: [12.3, 0.34, 0.85, 9.6, -0.15, -0.10] if t==0 else
    [np.random.uniform(8, 20), np.random.uniform(0.2, 0.5),
     np.random.uniform(0.5, 1.2), np.random.uniform(5, 15),
     np.random.uniform(-0.5, 0.0), np.random.uniform(-0.5, 0.0)])
eval_model("H10: d=3+2D,g=3", p_H10, model_H10, 6)

# ============================================================
# H11: 分段α + 2D g=3修正
# ============================================================
print(f"\n{'='*80}")
print("H11: 分段α + 2D g=3修正")

def model_H11(params, z, d, g):
    a, alpha1, alpha2, gamma, z0, c_2d_g3 = params
    alpha_eff = np.where(d <= 3, alpha1, alpha2)
    correction = (a * np.exp(-alpha_eff * d) / g**gamma) * (1 - np.exp(-z / z0))
    correction *= (1 + c_2d_g3 * ((d == 2) & (g == 3)))
    return (1 + correction) / (z - 1)

p_H11, ss_H11 = fit_model(model_H11, 6, 200,
    lambda t: [13.7, 0.39, 0.34, 0.85, 9.7, -0.10] if t==0 else
    [np.random.uniform(10, 20), np.random.uniform(0.25, 0.50),
     np.random.uniform(0.25, 0.45), np.random.uniform(0.7, 1.0),
     np.random.uniform(7, 13), np.random.uniform(-0.4, 0.0)])
eval_model("H11: 分段α+2D,g=3", p_H11, model_H11, 6)

# ============================================================
# H12: exp(-α·d^δ) — 非线性衰减
# ============================================================
print(f"\n{'='*80}")
print("H12: exp(-α·d^δ)")

def model_H12(params, z, d, g):
    a, alpha, delta, gamma, z0 = params
    correction = (a * np.exp(-alpha * d**delta) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

p_H12, ss_H12 = fit_model(model_H12, 5, 200,
    lambda t: [12.3, 0.1, 0.9, 0.85, 9.6] if t==0 else
    [np.random.uniform(5, 25), np.random.uniform(0.01, 0.3),
     np.random.uniform(0.6, 1.3), np.random.uniform(0.5, 1.2),
     np.random.uniform(5, 15)])
eval_model("H12: exp(-α·d^δ)", p_H12, model_H12, 5)

# ============================================================
# 综合对比表
# ============================================================
print(f"\n{'='*80}")
print("综合对比表")
print(f"{'='*80}")

all_models = [
    ("基线(4参数)", bp, model_baseline, 4),
    ("H4: 分段α(5)", p_H4, model_H4, 5),
    ("H6: 分段α+f3(6)", p_H6, model_H6, 6),
    ("H7: d3g3修正(5)", p_H7, model_H7, 5),
    ("H8: logistic(6)", p_H8, model_H8, 6),
    ("H9: 2D_g3修正(5)", p_H9, model_H9, 5),
    ("H10: d3+2Dg3(6)", p_H10, model_H10, 6),
    ("H11: 分段α+2Dg3(6)", p_H11, model_H11, 6),
    ("H12: d^δ(5)", p_H12, model_H12, 5),
]

print(f"\n{'模型':<22} {'MAPE':>7} {'AIC':>8} {'d=3':>7} {'d3g3':>7} {'sq_g3':>7} {'hc≥8':>7}")
print("-" * 72)

for name, params, func, k in all_models:
    pred = func(params, z, d, g)
    errors = np.abs((pc - pred) / pc) * 100
    mape = np.mean(errors)
    ssr = np.sum((pc - pred)**2)
    aic = n * np.log(ssr / n) + 2 * k
    e_d3 = np.mean(errors[mask_d3])
    e_d3g3 = np.mean(errors[mask_d3_g3])
    e_sq_g3 = np.mean(errors[mask_sq_g3])
    e_hc8 = np.mean(errors[mask_hc8])
    print(f"{name:<22} {mape:>6.2f}% {aic:>8.1f} {e_d3:>6.2f}% {e_d3g3:>6.2f}% {e_sq_g3:>6.2f}% {e_hc8:>6.2f}%")

# ============================================================
# 10-fold CV 对最佳模型
# ============================================================
print(f"\n{'='*80}")
print("最佳模型10-fold CV")
print(f"{'='*80}")

# 选最佳（基于MAPE和AIC的综合评估）
# 先评估所有模型的CV
np.random.seed(42)
for name, params, func, k in all_models[:4]:  # 只评估前4个
    cv_errors = []
    for fold in range(10):
        test_idx = np.arange(fold * (n//10), (fold+1) * (n//10) if fold < 9 else n)
        train_idx = np.setdiff1d(np.arange(n), test_idx)

        best_fp = None
        best_fss = np.inf
        for _ in range(30):
            if _ == 0:
                p0 = params.copy()
            else:
                p0 = params * np.random.uniform(0.8, 1.2, k)
            res = minimize(lambda p: np.sum((pc[train_idx] - func(p, z[train_idx], d[train_idx], g[train_idx]))**2),
                          p0, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_fss:
                best_fss = res.fun
                best_fp = res.x

        pred_test = func(best_fp, z[test_idx], d[test_idx], g[test_idx])
        cv_errors.extend(np.abs((pc[test_idx] - pred_test) / pc[test_idx]) * 100)

    cv_mape = np.mean(cv_errors)
    print(f"  {name}: CV MAPE = {cv_mape:.2f}%")

print(f"\n{'='*80}")
print("实验完成")
print(f"{'='*80}")
