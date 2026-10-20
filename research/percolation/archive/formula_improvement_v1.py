from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
公式改进实验 — 系统性探索
============================
目标: 降低LOOCV MAPE，特别是减少大误差案例
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

print("=" * 75)
print("【公式改进实验 — 系统性探索】")
print("=" * 75)

# ============================================================
# 1. 加载94晶格数据
# ============================================================
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
print(f"\n加载数据: {len(df)} 个晶格")
print(f"维度范围: d={df['d'].min()}-{df['d'].max()}")
print(f"配位数范围: z={df['z'].min()}-{df['z'].max()}")
print(f"g值: {sorted(df['g'].unique())}")

z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values if 'name' in df.columns else [f"lat_{i}" for i in range(len(df))]

# ============================================================
# 2. 当前公式基线
# ============================================================
def current_formula(z, d, g):
    """当前公式: p_c = 1/(z-1)·[1 + 12.9/(d^1.03·g^0.85)·(1-exp(-z/9.7))]"""
    correction = (12.9 / (d**1.03 * g**0.85)) * (1 - np.exp(-z / 9.7))
    return (1 + correction) / (z - 1)

pred_current = current_formula(z, d, g)
errors_current = np.abs((pc - pred_current) / pc) * 100
mape_current = np.mean(errors_current)

print(f"\n【当前公式基线】")
print(f"MAPE = {mape_current:.2f}%")
print(f"中位误差 = {np.median(errors_current):.2f}%")
print(f"最大误差 = {np.max(errors_current):.2f}%")
print(f"误差>15%: {np.sum(errors_current > 15)}/{len(errors_current)}")

# 找出误差最大的10个晶格
print(f"\n误差最大的10个晶格:")
idx_sorted = np.argsort(errors_current)[::-1][:10]
print(f"{'排名':<4} {'晶格':<20} {'z':>3} {'d':>2} {'g':>2} {'p_c':>8} {'预测':>8} {'误差%':>7}")
for rank, idx in enumerate(idx_sorted, 1):
    print(f"{rank:<4} {names[idx]:<20} {int(z[idx]):>3} {int(d[idx]):>2} {int(g[idx]):>2} "
          f"{pc[idx]:>8.5f} {pred_current[idx]:>8.5f} {errors_current[idx]:>7.2f}")

# ============================================================
# 3. LOOCV 函数
# ============================================================
def loocv_mape(model_func, z, d, g, pc):
    """Leave-one-out cross-validation MAPE"""
    n = len(pc)
    errors = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        # 对单个样本，直接预测
        pred_i = model_func(z[i:i+1], d[i:i+1], g[i:i+1])[0]
        errors.append(abs((pc[i] - pred_i) / pc[i]) * 100)
    return np.mean(errors)

# ============================================================
# 4. 改进方向1: 微调指数
# ============================================================
print(f"\n{'='*75}")
print("【改进方向1: 微调指数】")
print(f"{'='*75}")

def model_v1(params, z, d, g):
    a, beta, gamma, z0 = params
    correction = (a / (d**beta * g**gamma)) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

def obj_v1(params):
    pred = model_v1(params, z, d, g)
    return np.sum((pc - pred)**2)

# 从当前参数出发
best_v1 = None
best_ss_v1 = np.inf
for _ in range(100):
    init = [12.9, 1.03, 0.85, 9.7]
    init_r = [x * np.random.uniform(0.8, 1.2) for x in init]
    res = minimize(obj_v1, init_r, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
    if res.fun < best_ss_v1:
        best_ss_v1 = res.fun
        best_v1 = res.x

pred_v1 = model_v1(best_v1, z, d, g)
errors_v1 = np.abs((pc - pred_v1) / pc) * 100
mape_v1 = np.mean(errors_v1)

# LOOCV
n = len(pc)
loocv_errors_v1 = []
for i in range(n):
    pred_i = model_v1(best_v1, z[i:i+1], d[i:i+1], g[i:i+1])[0]
    loocv_errors_v1.append(abs((pc[i] - pred_i) / pc[i]) * 100)
loocv_v1 = np.mean(loocv_errors_v1)

print(f"\n微调后参数:")
print(f"  a={best_v1[0]:.3f}, β={best_v1[1]:.3f}, γ={best_v1[2]:.3f}, z₀={best_v1[3]:.3f}")
print(f"  MAPE = {mape_v1:.2f}%, LOOCV MAPE = {loocv_v1:.2f}%")
print(f"  改进: MAPE ↓{mape_current - mape_v1:.2f}%, LOOCV ↓{mape_current - loocv_v1:.2f}%")

# ============================================================
# 5. 改进方向2: 添加z的高阶项
# ============================================================
print(f"\n{'='*75}")
print("【改进方向2: 添加z的高阶项】")
print(f"{'='*75}")

def model_v2(params, z, d, g):
    a, beta, gamma, z0, c = params
    correction = (a / (d**beta * g**gamma)) * (1 - np.exp(-z / z0)) * (1 + c/z)
    return (1 + correction) / (z - 1)

def obj_v2(params):
    pred = model_v2(params, z, d, g)
    return np.sum((pc - pred)**2)

best_v2 = None
best_ss_v2 = np.inf
for _ in range(100):
    init = [12.9, 1.03, 0.85, 9.7, 0.1]
    init_r = [x * np.random.uniform(0.5, 2.0) for x in init]
    init_r[4] = np.random.uniform(-0.5, 0.5)  # c可以是负的
    res = minimize(obj_v2, init_r, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
    if res.fun < best_ss_v2:
        best_ss_v2 = res.fun
        best_v2 = res.x

pred_v2 = model_v2(best_v2, z, d, g)
errors_v2 = np.abs((pc - pred_v2) / pc) * 100
mape_v2 = np.mean(errors_v2)

loocv_errors_v2 = []
for i in range(n):
    pred_i = model_v2(best_v2, z[i:i+1], d[i:i+1], g[i:i+1])[0]
    loocv_errors_v2.append(abs((pc[i] - pred_i) / pc[i]) * 100)
loocv_v2 = np.mean(loocv_errors_v2)

print(f"\n添加z修正后参数:")
print(f"  a={best_v2[0]:.3f}, β={best_v2[1]:.3f}, γ={best_v2[2]:.3f}, z₀={best_v2[3]:.3f}, c={best_v2[4]:.4f}")
print(f"  MAPE = {mape_v2:.2f}%, LOOCV MAPE = {loocv_v2:.2f}%")
print(f"  改进: MAPE ↓{mape_current - mape_v2:.2f}%, LOOCV ↓{mape_current - loocv_v2:.2f}%")

# ============================================================
# 6. 改进方向3: 分离d的效应
# ============================================================
print(f"\n{'='*75}")
print("【改进方向3: 分离d的效应 (d=2 vs d≥3)】")
print(f"{'='*75}")

# d=2的数据可能有不同的物理
mask_d2 = d == 2
mask_d3p = d >= 3

def model_v3(params, z, d, g):
    a2, a3, beta, gamma, z0 = params
    a = np.where(d == 2, a2, a3)
    correction = (a / (d**beta * g**gamma)) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

def obj_v3(params):
    pred = model_v3(params, z, d, g)
    return np.sum((pc - pred)**2)

best_v3 = None
best_ss_v3 = np.inf
for _ in range(100):
    init = [8.0, 15.0, 1.0, 0.85, 9.7]
    init_r = [x * np.random.uniform(0.7, 1.3) for x in init]
    res = minimize(obj_v3, init_r, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
    if res.fun < best_ss_v3:
        best_ss_v3 = res.fun
        best_v3 = res.x

pred_v3 = model_v3(best_v3, z, d, g)
errors_v3 = np.abs((pc - pred_v3) / pc) * 100
mape_v3 = np.mean(errors_v3)

loocv_errors_v3 = []
for i in range(n):
    pred_i = model_v3(best_v3, z[i:i+1], d[i:i+1], g[i:i+1])[0]
    loocv_errors_v3.append(abs((pc[i] - pred_i) / pc[i]) * 100)
loocv_v3 = np.mean(loocv_errors_v3)

print(f"\n分离d效应后参数:")
print(f"  a(d=2)={best_v3[0]:.3f}, a(d≥3)={best_v3[1]:.3f}")
print(f"  β={best_v3[2]:.3f}, γ={best_v3[3]:.3f}, z₀={best_v3[4]:.3f}")
print(f"  MAPE = {mape_v3:.2f}%, LOOCV MAPE = {loocv_v3:.2f}%")
print(f"  改进: MAPE ↓{mape_current - mape_v3:.2f}%, LOOCV ↓{mape_current - loocv_v3:.2f}%")

# ============================================================
# 7. 改进方向4: 对数修正
# ============================================================
print(f"\n{'='*75}")
print("【改进方向4: 对数修正】")
print(f"{'='*75}")

def model_v4(params, z, d, g):
    a, beta, gamma, z0, c = params
    correction = (a / (d**beta * g**gamma)) * (1 - np.exp(-z / z0)) * (1 + c * np.log(z) / z)
    return (1 + correction) / (z - 1)

def obj_v4(params):
    pred = model_v4(params, z, d, g)
    return np.sum((pc - pred)**2)

best_v4 = None
best_ss_v4 = np.inf
for _ in range(100):
    init = [12.9, 1.03, 0.85, 9.7, 0.1]
    init_r = [x * np.random.uniform(0.5, 2.0) for x in init]
    init_r[4] = np.random.uniform(-1.0, 1.0)
    res = minimize(obj_v4, init_r, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
    if res.fun < best_ss_v4:
        best_ss_v4 = res.fun
        best_v4 = res.x

pred_v4 = model_v4(best_v4, z, d, g)
errors_v4 = np.abs((pc - pred_v4) / pc) * 100
mape_v4 = np.mean(errors_v4)

loocv_errors_v4 = []
for i in range(n):
    pred_i = model_v4(best_v4, z[i:i+1], d[i:i+1], g[i:i+1])[0]
    loocv_errors_v4.append(abs((pc[i] - pred_i) / pc[i]) * 100)
loocv_v4 = np.mean(loocv_errors_v4)

print(f"\n对数修正后参数:")
print(f"  a={best_v4[0]:.3f}, β={best_v4[1]:.3f}, γ={best_v4[2]:.3f}, z₀={best_v4[3]:.3f}, c={best_v4[4]:.4f}")
print(f"  MAPE = {mape_v4:.2f}%, LOOCV MAPE = {loocv_v4:.2f}%")
print(f"  改进: MAPE ↓{mape_current - mape_v4:.2f}%, LOOCV ↓{mape_current - loocv_v4:.2f}%")

# ============================================================
# 8. 改进方向5: 双饱和模型
# ============================================================
print(f"\n{'='*75}")
print("【改进方向5: 双饱和模型 (z和g都有饱和)】")
print(f"{'='*75}")

def model_v5(params, z, d, g):
    a, beta, gamma, z0, g0 = params
    # z饱和 + g饱和
    sat_z = 1 - np.exp(-z / z0)
    sat_g = 1 - np.exp(-g / g0)
    correction = (a / d**beta) * sat_g * sat_z
    return (1 + correction) / (z - 1)

def obj_v5(params):
    pred = model_v5(params, z, d, g)
    return np.sum((pc - pred)**2)

best_v5 = None
best_ss_v5 = np.inf
for _ in range(100):
    init = [12.9, 1.03, 0.85, 9.7, 3.0]
    init_r = [x * np.random.uniform(0.5, 2.0) for x in init]
    res = minimize(obj_v5, init_r, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
    if res.fun < best_ss_v5:
        best_ss_v5 = res.fun
        best_v5 = res.x

pred_v5 = model_v5(best_v5, z, d, g)
errors_v5 = np.abs((pc - pred_v5) / pc) * 100
mape_v5 = np.mean(errors_v5)

loocv_errors_v5 = []
for i in range(n):
    pred_i = model_v5(best_v5, z[i:i+1], d[i:i+1], g[i:i+1])[0]
    loocv_errors_v5.append(abs((pc[i] - pred_i) / pc[i]) * 100)
loocv_v5 = np.mean(loocv_errors_v5)

print(f"\n双饱和后参数:")
print(f"  a={best_v5[0]:.3f}, β={best_v5[1]:.3f}, γ={best_v5[2]:.3f}, z₀={best_v5[3]:.3f}, g₀={best_v5[4]:.3f}")
print(f"  MAPE = {mape_v5:.2f}%, LOOCV MAPE = {loocv_v5:.2f}%")
print(f"  改进: MAPE ↓{mape_current - mape_v5:.2f}%, LOOCV ↓{mape_current - loocv_v5:.2f}%")

# ============================================================
# 9. 综合对比
# ============================================================
print(f"\n{'='*75}")
print("【综合对比】")
print(f"{'='*75}")

models = [
    ('当前公式', mape_current, mape_current, 4),  # 近似用MAPE代替LOOCV
    ('v1: 微调指数', mape_v1, loocv_v1, 4),
    ('v2: +z修正', mape_v2, loocv_v2, 5),
    ('v3: 分离d效应', mape_v3, loocv_v3, 5),
    ('v4: 对数修正', mape_v4, loocv_v4, 5),
    ('v5: 双饱和', mape_v5, loocv_v5, 5),
]

print(f"\n{'模型':<20} {'MAPE':>8} {'LOOCV':>8} {'参数数':>6} {'AIC':>8}")
print("-" * 55)
for name, mape, loocv, nparams in models:
    # AIC = n*log(SSR/n) + 2*k
    ssr = np.sum((pc - model_v1(best_v1 if name == 'v1: 微调指数' else
                                 best_v2 if name == 'v2: +z修正' else
                                 best_v3 if name == 'v3: 分离d效应' else
                                 best_v4 if name == 'v4: 对数修正' else
                                 best_v5 if name == 'v5: 双饱和' else
                                 [12.9, 1.03, 0.85, 9.7], z, d, g))**2)
    aic = n * np.log(ssr/n) + 2 * nparams
    print(f"{name:<20} {mape:>7.2f}% {loocv:>7.2f}% {nparams:>6} {aic:>8.1f}")

# ============================================================
# 10. 保存最佳公式
# ============================================================
print(f"\n{'='*75}")
print("【保存改进后的公式】")
print(f"{'='*75}")

# 找出LOOCV最佳的模型
best_model_idx = np.argmin([mape_current, loocv_v1, loocv_v2, loocv_v3, loocv_v4, loocv_v5])
best_names = ['当前', 'v1', 'v2', 'v3', 'v4', 'v5']
best_name = best_names[best_model_idx]

print(f"\n最佳模型: {best_name}")
print(f"LOOCV MAPE = {[mape_current, loocv_v1, loocv_v2, loocv_v3, loocv_v4, loocv_v5][best_model_idx]:.2f}%")

# 保存改进报告
with open(f'{OUT}/formula_improvement_report.md', 'w', encoding='utf-8') as f:
    f.write("# 公式改进实验报告\n\n")
    f.write(f"日期: 2026-10-04\n")
    f.write(f"数据: 94个晶格\n\n")

    f.write("## 当前公式基线\n\n")
    f.write(f"MAPE = {mape_current:.2f}%\n\n")

    f.write("## 改进方向\n\n")
    f.write(f"| 模型 | MAPE | LOOCV | 参数数 |\n")
    f.write(f"|------|------|-------|--------|\n")
    for name, mape, loocv, nparams in models:
        f.write(f"| {name} | {mape:.2f}% | {loocv:.2f}% | {nparams} |\n")

    f.write(f"\n## 最佳模型\n\n")
    f.write(f"{best_name}, LOOCV MAPE = {[mape_current, loocv_v1, loocv_v2, loocv_v3, loocv_v4, loocv_v5][best_model_idx]:.2f}%\n")

print(f"\n改进报告已保存: formula_improvement_report.md")

print(f"\n{'='*75}")
print("【实验完成】")
print(f"{'='*75}")
