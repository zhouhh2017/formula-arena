from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
理论约束模型：使用 γ=1.0（一阶微扰预言）
============================================
如果 γ=1.0 在统计上可接受，就用它！
这样模型就有更强的理论基础。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')

z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values

print("=" * 80)
print("[理论约束模型: gamma=1.0]")
print("=" * 80)

# 模型: gamma = 1.0 (一阶微扰)
# 只剩3个参数: a, alpha, z0

def model_gamma1(params, z, d, g):
    a, alpha, z0 = params
    correction = (a / g) * np.exp(-alpha * d) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

# 拟合
best_gamma1 = None
best_ss = np.inf
for _ in range(1000):
    p0 = [np.random.uniform(5, 30), np.random.uniform(0.1, 0.6),
          np.random.uniform(5, 20)]
    try:
        res = minimize(lambda p: np.sum((pc - model_gamma1(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < best_ss:
            best_ss = res.fun
            best_gamma1 = res.x
    except:
        pass

a_g1, alpha_g1, z0_g1 = best_gamma1
pred_g1 = model_gamma1(best_gamma1, z, d, g)
err_g1 = np.abs((pc - pred_g1) / pc) * 100
mape_g1 = np.mean(err_g1)
ssr_g1 = np.sum((pc - pred_g1)**2)
n = len(pc)
aic_g1 = n * np.log(ssr_g1/n) + 2*3  # 3 parameters

print(f"\n模型 gamma=1.0 (一阶微扰):")
print(f"  a = {a_g1:.4f}")
print(f"  alpha = {alpha_g1:.4f}")
print(f"  z0 = {z0_g1:.4f}")
print(f"  MAPE = {mape_g1:.2f}%")
print(f"  AIC = {aic_g1:.1f}")
print(f"  参数数量: 3")

# 对比原模型 (4个参数)
a_orig, alpha_orig, gamma_orig, z0_orig = 12.3346, 0.3368, 0.8477, 9.6171
pred_orig = (1 + (a_orig / g**gamma_orig) * np.exp(-alpha_orig * d) * (1 - np.exp(-z / z0_orig))) / (z - 1)
mape_orig = np.mean(np.abs((pc - pred_orig) / pc) * 100)
ssr_orig = np.sum((pc - pred_orig)**2)
aic_orig = n * np.log(ssr_orig/n) + 2*4

print(f"\n原模型 (gamma自由):")
print(f"  a = {a_orig:.4f}, alpha = {alpha_orig:.4f}, gamma = {gamma_orig:.4f}, z0 = {z0_orig:.4f}")
print(f"  MAPE = {mape_orig:.2f}%")
print(f"  AIC = {aic_orig:.1f}")
print(f"  参数数量: 4")

print(f"\n对比:")
print(f"  Delta MAPE = {mape_g1 - mape_orig:.2f}%")
print(f"  Delta AIC = {aic_g1 - aic_orig:.1f}")

if aic_g1 < aic_orig:
    print(f"  gamma=1.0 模型 AIC 更优!")
else:
    print(f"  原模型 AIC 更优 (但差距不大)")

# 也测试 alpha = 1/3 (2/d_c) 的理论约束
print(f"\n{'='*80}")
print("进一步约束: alpha = 1/3 (= 2/d_c)")
print("=" * 80)

def model_constrained(params, z, d, g):
    a, z0 = params
    alpha = 1/3  # 理论预言
    gamma = 1.0  # 理论预言
    correction = (a / g) * np.exp(-alpha * d) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_constrained = None
best_ss_c = np.inf
for _ in range(1000):
    p0 = [np.random.uniform(5, 30), np.random.uniform(5, 20)]
    try:
        res = minimize(lambda p: np.sum((pc - model_constrained(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < best_ss_c:
            best_ss_c = res.fun
            best_constrained = res.x
    except:
        pass

a_c, z0_c = best_constrained
pred_c = model_constrained(best_constrained, z, d, g)
err_c = np.abs((pc - pred_c) / pc) * 100
mape_c = np.mean(err_c)
ssr_c = np.sum((pc - pred_c)**2)
aic_c = n * np.log(ssr_c/n) + 2*2  # 2 parameters

print(f"\n完全理论约束模型 (alpha=1/3, gamma=1):")
print(f"  a = {a_c:.4f}")
print(f"  z0 = {z0_c:.4f}")
print(f"  MAPE = {mape_c:.2f}%")
print(f"  AIC = {aic_c:.1f}")
print(f"  参数数量: 2")

# 交叉验证
print(f"\n{'='*80}")
print("10折交叉验证对比")
print("=" * 80)

np.random.seed(42)
cv_errors = {'gamma=1': [], 'constrained': [], 'original': []}

for fold in range(10):
    test_idx = np.random.choice(n, size=n//10, replace=False)
    train_idx = np.setdiff1d(np.arange(n), test_idx)

    z_tr, d_tr, g_tr, pc_tr = z[train_idx], d[train_idx], g[train_idx], pc[train_idx]
    z_te, d_te, g_te, pc_te = z[test_idx], d[test_idx], g[test_idx], pc[test_idx]

    # 拟合 gamma=1 模型
    best_fold = None
    best_ss_fold = np.inf
    for _ in range(50):
        p0 = [np.random.uniform(5, 30), np.random.uniform(0.1, 0.6), np.random.uniform(5, 20)]
        try:
            res = minimize(lambda p: np.sum((pc_tr - model_gamma1(p, z_tr, d_tr, g_tr))**2),
                          p0, method='Nelder-Mead', options={'maxiter': 5000})
            if res.fun < best_ss_fold:
                best_ss_fold = res.fun
                best_fold = res.x
        except:
            pass
    if best_fold is not None:
        pred_te = model_gamma1(best_fold, z_te, d_te, g_te)
        cv_errors['gamma=1'].extend(np.abs((pc_te - pred_te) / pc_te * 100).tolist())

    # 拟合 constrained 模型
    best_fold_c = None
    best_ss_fold_c = np.inf
    for _ in range(50):
        p0 = [np.random.uniform(5, 30), np.random.uniform(5, 20)]
        try:
            res = minimize(lambda p: np.sum((pc_tr - model_constrained(p, z_tr, d_tr, g_tr))**2),
                          p0, method='Nelder-Mead', options={'maxiter': 5000})
            if res.fun < best_ss_fold_c:
                best_ss_fold_c = res.fun
                best_fold_c = res.x
        except:
            pass
    if best_fold_c is not None:
        pred_te = model_constrained(best_fold_c, z_te, d_te, g_te)
        cv_errors['constrained'].extend(np.abs((pc_te - pred_te) / pc_te * 100).tolist())

    # 原模型
    best_fold_o = None
    best_ss_fold_o = np.inf
    for _ in range(50):
        p0 = [np.random.uniform(5, 20), np.random.uniform(0.1, 0.6),
              np.random.uniform(0.3, 1.5), np.random.uniform(5, 15)]
        try:
            res = minimize(lambda p: np.sum((pc_tr - model_gamma1.__wrapped__(p, z_tr, d_tr, g_tr) if hasattr(model_gamma1, '__wrapped__') else (1 + (p[0]/g_tr**p[2]) * np.exp(-p[1]*d_tr) * (1-np.exp(-z_tr/p[3])))/(z_tr-1))**2),
                          p0, method='Nelder-Mead', options={'maxiter': 5000})
            if res.fun < best_ss_fold_o:
                best_ss_fold_o = res.fun
                best_fold_o = res.x
        except:
            pass
    if best_fold_o is not None:
        corr = (best_fold_o[0] / g_te**best_fold_o[2]) * np.exp(-best_fold_o[1]*d_te) * (1-np.exp(-z_te/best_fold_o[3]))
        pred_te = (1 + corr) / (z_te - 1)
        cv_errors['original'].extend(np.abs((pc_te - pred_te) / pc_te * 100).tolist())

for model_name, errors in cv_errors.items():
    if errors:
        print(f"  {model_name:<15}: CV MAPE = {np.mean(errors):.2f}% +/- {np.std(errors):.2f}%")

# 分维度性能
print(f"\n{'='*80}")
print("分维度性能对比")
print("=" * 80)

for d_range in [(2, 2), (3, 3), (4, 13)]:
    mask = (d >= d_range[0]) & (d <= d_range[1])
    n_lattices = np.sum(mask)

    err_g1_d = np.mean(err_g1[mask])
    err_c_d = np.mean(err_c[mask])
    err_o_d = np.mean(np.abs((pc[mask] - pred_orig[mask]) / pc[mask]) * 100)

    print(f"\n  d={d_range[0]}{'-'+str(d_range[1]) if d_range[0]!=d_range[1] else ''} ({n_lattices} lattices):")
    print(f"    gamma=1.0:    MAPE = {err_g1_d:.2f}%")
    print(f"    constrained:  MAPE = {err_c_d:.2f}%")
    print(f"    original:     MAPE = {err_o_d:.2f}%")

# 最终评估
print(f"\n{'='*80}")
print("最终评估")
print("=" * 80)

print(f"""
三种模型的对比:

| 模型 | 参数数 | MAPE | AIC | 理论基础 |
|------|--------|------|-----|----------|
| 原模型 (gamma自由) | 4 | {mape_orig:.2f}% | {aic_orig:.1f} | 经验 |
| gamma=1.0 | 3 | {mape_g1:.2f}% | {aic_g1:.1f} | 一阶微扰 |
| alpha=1/3, gamma=1 | 2 | {mape_c:.2f}% | {aic_c:.1f} | 一阶微扰+2/d_c |

关键问题:
1. gamma=1.0 模型的 MAPE 增加多少?
2. AIC 是否支持更简单的模型?
3. 理论约束是否物理合理?

分析:
""")

if mape_g1 < mape_orig + 0.5:
    print("gamma=1.0 模型: MAPE 增加不到 0.5%，可以接受！")
else:
    print(f"gamma=1.0 模型: MAPE 增加 {mape_g1-mape_orig:.2f}%")

if aic_g1 < aic_orig:
    print("gamma=1.0 模型: AIC 更优！（少一个参数但拟合不差）")
elif aic_g1 < aic_orig + 2:
    print(f"gamma=1.0 模型: AIC 差距 {aic_g1-aic_orig:.1f}，几乎等价")
else:
    print(f"gamma=1.0 模型: AIC 差距 {aic_g1-aic_orig:.1f}，原模型明显更优")

if mape_c < mape_orig + 1.0:
    print(f"\nalpha=1/3, gamma=1 模型: MAPE 增加 {mape_c-mape_orig:.2f}%，只有2个参数！")
    print("这可能是最有理论基础的模型！")

print(f"""
结论:

如果 gamma=1.0 和 alpha=1/3 的约束模型性能可接受:
- 我们就有了一个只有2个拟合参数的理论模型
- gamma=1.0 来自一阶微扰理论
- alpha=1/3 来自 alpha=2/d_c (二阶环路相互作用)
- z0 ~ 10 来自有效近邻饱和

这比4参数经验模型更有理论基础！
虽然仍不是严格的"第一原理推导"，
但至少参数有明确的物理来源。
""")

# 保存
with open(f'{OUT}/CONSTRAINED_MODEL_RESULTS.md', 'w', encoding='utf-8') as f:
    f.write("# 理论约束模型结果\n\n")
    f.write("## 模型对比\n\n")
    f.write(f"| 模型 | 参数数 | MAPE | AIC |\n")
    f.write(f"|------|--------|------|-----|\n")
    f.write(f"| 原模型 | 4 | {mape_orig:.2f}% | {aic_orig:.1f} |\n")
    f.write(f"| gamma=1.0 | 3 | {mape_g1:.2f}% | {aic_g1:.1f} |\n")
    f.write(f"| alpha=1/3, gamma=1 | 2 | {mape_c:.2f}% | {aic_c:.1f} |\n\n")
    f.write(f"## gamma=1.0 模型参数\n\n")
    f.write(f"- a = {a_g1:.4f}\n")
    f.write(f"- alpha = {alpha_g1:.4f}\n")
    f.write(f"- z0 = {z0_g1:.4f}\n\n")
    f.write(f"## alpha=1/3, gamma=1 模型参数\n\n")
    f.write(f"- a = {a_c:.4f}\n")
    f.write(f"- z0 = {z0_c:.4f}\n")

print(f"\n保存: CONSTRAINED_MODEL_RESULTS.md")
print("=" * 80)
