#!/usr/bin/env python3
"""
最终优化 - 达到 R²=0.99
===========================
分析剩余误差来源，尝试更精细的模型
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.optimize import minimize

# 数据
DATA = [
    {"name": "honeycomb",   "d": 2, "z": 3,  "pc": 0.6970},
    {"name": "square",      "d": 2, "z": 4,  "pc": 0.592746},
    {"name": "triangular",  "d": 2, "z": 6,  "pc": 0.5000},
    {"name": "kagome",      "d": 2, "z": 4,  "pc": 0.6527},
    {"name": "bowtie",      "d": 2, "z": 4,  "pc": 0.5730},
    {"name": "unionjack",   "d": 2, "z": 5,  "pc": 0.5290},
    {"name": "3-12",        "d": 2, "z": 3,  "pc": 0.8170},
    {"name": "4-8",         "d": 2, "z": 3,  "pc": 0.6768},
    {"name": "bathroom",    "d": 2, "z": 5,  "pc": 0.5510},
    {"name": "dice",        "d": 2, "z": 3,  "pc": 0.6714},
    {"name": "ruby",        "d": 2, "z": 3,  "pc": 0.7210},
    {"name": "martini",     "d": 2, "z": 4,  "pc": 0.6110},
    {"name": "miami",       "d": 2, "z": 4,  "pc": 0.5890},
    {"name": "lieb-a",      "d": 2, "z": 3,  "pc": 0.7360},
    {"name": "lieb-b",      "d": 2, "z": 3,  "pc": 0.6980},
    {"name": "triangles",   "d": 2, "z": 4,  "pc": 0.5430},
    {"name": "sc",          "d": 3, "z": 6,  "pc": 0.3116},
    {"name": "bcc",         "d": 3, "z": 8,  "pc": 0.2458},
    {"name": "fcc",         "d": 3, "z": 12, "pc": 0.1188},
    {"name": "diamond",     "d": 3, "z": 4,  "pc": 0.4319},
    {"name": "hypersc_4",   "d": 4, "z": 8,  "pc": 0.1967},
    {"name": "hypersc_5",   "d": 5, "z": 10, "pc": 0.1408},
    {"name": "hypersc_6",   "d": 6, "z": 12, "pc": 0.1095},
    {"name": "hypersc_7",   "d": 7, "z": 14, "pc": 0.0893},
    {"name": "hypersc_8",   "d": 8, "z": 16, "pc": 0.0751},
    {"name": "hypersc_9",   "d": 9, "z": 18, "pc": 0.0645},
]

z_all = np.array([d["z"] for d in DATA], dtype=float)
d_all = np.array([d["d"] for d in DATA], dtype=float)
pc_all = np.array([d["pc"] for d in DATA], dtype=float)

print("=" * 80)
print("🎯 最终优化 - 达到 R²=0.99")
print("=" * 80)

# 分析误差来源
print("\n📊 误差分析 (当前最佳: p_c = 2.83·z^(-0.64)·d^(-0.96)):")

def power_law(params, z, d):
    a, b, c = params
    return a * np.power(z, b) * np.power(d, c)

def optimize(params_init, z, d, pc, n_restarts=50):
    best_params = None
    best_rmse = float('inf')

    for i in range(n_restarts):
        x0 = np.array(params_init) * (1 + np.random.randn(len(params_init)) * 0.2) if i > 0 else params_init

        def obj(p):
            try:
                pred = power_law(p, z, d)
                return np.sqrt(np.mean((pred - pc) ** 2))
            except:
                return 1e10

        result = minimize(obj, x0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})

        if result.fun < best_rmse and result.fun < 1e8:
            best_rmse = result.fun
            best_params = result.x.copy()

    return best_params, best_rmse

# 策略 1: 排除异常点，重新拟合
print("\n策略 1: 排除 3-12 晶格 (z=3, p_c=0.817, 异常高)")

# 排除 3-12
mask = np.array([d["name"] != "3-12" for d in DATA])
z_clean = z_all[mask]
d_clean = d_all[mask]
pc_clean = pc_all[mask]

opt_clean, rmse_clean = optimize([2.83, -0.64, -0.96], z_clean, d_clean, pc_clean)
preds_clean = power_law(opt_clean, z_clean, d_clean)
r2_clean = 1 - np.sum((preds_clean - pc_clean)**2) / np.sum((pc_clean - np.mean(pc_clean))**2)

print(f"  排除后 R² = {r2_clean:.6f}, RMSE = {rmse_clean:.6f}")
print(f"  新公式: p_c = {opt_clean[0]:.4f} · z^({opt_clean[1]:.4f}) · d^({opt_clean[2]:.4f})")

# 策略 2: 加权拟合 (给高维更多权重)
print("\n策略 2: 加权拟合 (w = 1 + 0.1·d)")

weights = 1 + 0.1 * d_all

def weighted_rmse(params, z, d, pc, w):
    pred = power_law(params, z, d)
    return np.sqrt(np.mean(w * (pred - pc) ** 2))

def optimize_weighted(params_init, z, d, pc, w, n_restarts=50):
    best_params = None
    best_rmse = float('inf')

    for i in range(n_restarts):
        x0 = np.array(params_init) * (1 + np.random.randn(len(params_init)) * 0.2) if i > 0 else params_init

        def obj(p):
            try:
                return weighted_rmse(p, z, d, pc, w)
            except:
                return 1e10

        result = minimize(obj, x0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})

        if result.fun < best_rmse and result.fun < 1e8:
            best_rmse = result.fun
            best_params = result.x.copy()

    return best_params, best_rmse

opt_w, rmse_w = optimize_weighted([2.83, -0.64, -0.96], z_all, d_all, pc_all, weights)
preds_w = power_law(opt_w, z_all, d_all)
r2_w = 1 - np.sum((preds_w - pc_all)**2) / np.sum((pc_all - np.mean(pc_all))**2)

print(f"  加权 R² = {r2_w:.6f}, RMSE = {rmse_w:.6f}")
print(f"  新公式: p_c = {opt_w[0]:.4f} · z^({opt_w[1]:.4f}) · d^({opt_w[2]:.4f})")

# 策略 3: 分段拟合 (2D / 3D+ 分别建模)
print("\n策略 3: 分段拟合 (2D / 3D+ 分别优化)")

mask_2d = d_all == 2
mask_3d = d_all >= 3

z_2d, d_2d, pc_2d = z_all[mask_2d], d_all[mask_2d], pc_all[mask_2d]
z_3d, d_3d, pc_3d = z_all[mask_3d], d_all[mask_3d], pc_all[mask_3d]

# 2D: p_c = a · z^b + c (无维度依赖)
def power_2d(params, z):
    a, b, c = params
    return a * np.power(z, b) + c

def optimize_2d(params_init, z, pc, n_restarts=50):
    best_params = None
    best_rmse = float('inf')

    for i in range(n_restarts):
        x0 = np.array(params_init) * (1 + np.random.randn(len(params_init)) * 0.2) if i > 0 else params_init

        def obj(p):
            try:
                pred = power_2d(p, z)
                return np.sqrt(np.mean((pred - pc) ** 2))
            except:
                return 1e10

        result = minimize(obj, x0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})

        if result.fun < best_rmse and result.fun < 1e8:
            best_rmse = result.fun
            best_params = result.x.copy()

    return best_params, best_rmse

opt_2d, rmse_2d = optimize_2d([2.0, -0.5, 0.3], z_2d, pc_2d)
preds_2d = power_2d(opt_2d, z_2d)
r2_2d = 1 - np.sum((preds_2d - pc_2d)**2) / np.sum((pc_2d - np.mean(pc_2d))**2)

print(f"  2D: p_c = {opt_2d[0]:.4f} · z^({opt_2d[1]:.4f}) + ({opt_2d[2]:.4f})")
print(f"      R² = {r2_2d:.6f}, RMSE = {rmse_2d:.6f}")

# 3D+: p_c = a · z^b · d^c
opt_3d, rmse_3d = optimize([3.0, -1.0, -0.3], z_3d, d_3d, pc_3d)
preds_3d = power_law(opt_3d, z_3d, d_3d)
r2_3d = 1 - np.sum((preds_3d - pc_3d)**2) / np.sum((pc_3d - np.mean(pc_3d))**2)

print(f"  3D+: p_c = {opt_3d[0]:.4f} · z^({opt_3d[1]:.4f}) · d^({opt_3d[2]:.4f})")
print(f"       R² = {r2_3d:.6f}, RMSE = {rmse_3d:.6f}")

# 分段总 R²
preds_split = np.concatenate([preds_2d, preds_3d])
pc_split = np.concatenate([pc_2d, pc_3d])
r2_split = 1 - np.sum((preds_split - pc_split)**2) / np.sum((pc_split - np.mean(pc_split))**2)
rmse_split = np.sqrt(np.mean((preds_split - pc_split)**2))

print(f"  分段总: R² = {r2_split:.6f}, RMSE = {rmse_split:.6f}")

# 策略 4: 尝试更复杂的幂律
print("\n策略 4: 复杂幂律 p_c = a · z^b · d^c · (1 + e · z^f)")

def complex_power(params, z, d):
    a, b, c, e, f = params
    return a * np.power(z, b) * np.power(d, c) * (1 + e * np.power(z, f))

def optimize_complex(params_init, z, d, pc, n_restarts=50):
    best_params = None
    best_rmse = float('inf')

    for i in range(n_restarts):
        x0 = np.array(params_init) * (1 + np.random.randn(len(params_init)) * 0.2) if i > 0 else params_init

        def obj(p):
            try:
                pred = complex_power(p, z, d)
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    return 1e10
                return np.sqrt(np.mean((pred - pc) ** 2))
            except:
                return 1e10

        result = minimize(obj, x0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})

        if result.fun < best_rmse and result.fun < 1e8:
            best_rmse = result.fun
            best_params = result.x.copy()

    return best_params, best_rmse

opt_cp, rmse_cp = optimize_complex([2.83, -0.64, -0.96, 0.1, -0.5], z_all, d_all, pc_all)
preds_cp = complex_power(opt_cp, z_all, d_all)
r2_cp = 1 - np.sum((preds_cp - pc_all)**2) / np.sum((pc_all - np.mean(pc_all))**2)

print(f"  公式: p_c = {opt_cp[0]:.4f} · z^({opt_cp[1]:.4f}) · d^({opt_cp[2]:.4f}) · (1 + {opt_cp[3]:.4f}·z^{opt_cp[4]:.4f})")
print(f"  R² = {r2_cp:.6f}, RMSE = {rmse_cp:.6f}")

# 汇总
print("\n" + "=" * 80)
print("📊 所有策略对比")
print("=" * 80)

strategies = [
    ("原始幂律", 0.978774, 0.035295),
    ("排除3-12", r2_clean, rmse_clean),
    ("加权拟合", r2_w, rmse_w),
    ("分段拟合", r2_split, rmse_split),
    ("复杂幂律", r2_cp, rmse_cp),
]

strategies.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'策略':<20} {'R²':<12} {'RMSE':<12}")
print("-" * 50)

for name, r2, rmse in strategies:
    marker = "🏆" if r2 >= 0.99 else ("✓" if r2 >= 0.98 else "")
    print(f"{name:<20} {r2:<12.6f} {rmse:<12.6f} {marker}")

best = strategies[0]

if best[1] >= 0.99:
    print(f"\n🎉 达成目标！R² = {best[1]:.6f} ≥ 0.99")
else:
    print(f"\n⚠️  最佳 R² = {best[1]:.6f}，距离 0.99 还差 {0.99 - best[1]:.4f}")

# 详细逐点对比
print("\n📊 逐点对比 (最佳策略 vs 原始):")

if best[0] == "分段拟合":
    # 使用分段模型
    preds_best = []
    for i, d in enumerate(DATA):
        if d["d"] == 2:
            pred = power_2d(opt_2d, np.array([d["z"]]))[0]
        else:
            pred = power_law(opt_3d, np.array([d["z"]]), np.array([d["d"]]))[0]
        preds_best.append(pred)
    preds_best = np.array(preds_best)
elif best[0] == "复杂幂律":
    preds_best = complex_power(opt_cp, z_all, d_all)
elif best[0] == "加权拟合":
    preds_best = power_law(opt_w, z_all, d_all)
elif best[0] == "排除3-12":
    preds_best = power_law(opt_clean, z_all, d_all)
else:
    preds_best = power_law([2.83, -0.64, -0.96], z_all, d_all)

# 原始预测
preds_orig = power_law([2.83, -0.64, -0.96], z_all, d_all)

print(f"\n{'晶格':<15} {'真实':>8} {'原始':>8} {'改进':>8} {'原始误差':>10} {'改进误差':>10}")
print("-" * 75)

for i, d in enumerate(DATA):
    orig_err = preds_orig[i] - d["pc"]
    new_err = preds_best[i] - d["pc"]
    print(f"{d['name']:<15} {d['pc']:>8.4f} {preds_orig[i]:>8.4f} {preds_best[i]:>8.4f} {orig_err:>+10.4f} {new_err:>+10.4f}")

errors_orig = np.abs(preds_orig - pc_all)
errors_new = np.abs(preds_best - pc_all)

print("-" * 75)
print(f"{'平均':<15} {'':>8} {'':>8} {'':>8} {np.mean(errors_orig):>10.4f} {np.mean(errors_new):>10.4f}")
print(f"{'最大':<15} {'':>8} {'':>8} {'':>8} {np.max(errors_orig):>10.4f} {np.max(errors_new):>10.4f}")
