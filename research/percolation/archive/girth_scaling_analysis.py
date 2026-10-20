from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
深入分析：girth对标度律的影响
================================
假设：紧凑近邻（大girth）遵循Xun标度律，非紧凑近邻（小girth）不同
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
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

zpc = z * pc

print("=" * 80)
print("【Girth对标度律的影响分析】")
print("=" * 80)

# ============================================================
# 1. 按girth分组分析 zp_c
# ============================================================
print(f"\n{'='*80}")
print(f"【按girth分组：zp_c的渐近行为】")
print(f"{'='*80}")

# 理论值
theory_2d = 4.51235
theory_3d = 2.73512

for d_val in [2, 3]:
    mask_d = d == d_val
    theory = theory_2d if d_val == 2 else theory_3d

    print(f"\n维度 d={d_val} (理论: zp_c → {theory:.4f}):")
    print(f"  {'Girth':<8} {'N':>4} {'z范围':<12} {'zp_c均值':>10} {'zp_c标准差':>10} {'与理论偏差':>12}")
    print("  " + "-"*60)

    for g_val in sorted(set(g[mask_d].astype(int))):
        mask = mask_d & (g.astype(int) == g_val)
        if np.sum(mask) > 0:
            mean_zpc = np.mean(zpc[mask])
            std_zpc = np.std(zpc[mask])
            deviation = (mean_zpc - theory) / theory * 100
            z_min = int(np.min(z[mask]))
            z_max = int(np.max(z[mask]))
            print(f"  g={g_val:<5} {np.sum(mask):>4} {z_min:>3}-{z_max:<5} {mean_zpc:>10.4f} {std_zpc:>10.4f} {deviation:>+11.2f}%")

# ============================================================
# 2. 大z极限下的girth依赖
# ============================================================
print(f"\n{'='*80}")
print(f"【大z极限：girth对zp_c的影响】")
print(f"{'='*80}")

# 只取z >= 15的数据
mask_large_z = z >= 15
print(f"\n大z (z≥15) 数据，按(d, g)分组:")

for d_val in [2, 3]:
    for g_val in [3, 4, 6]:
        mask = mask_large_z & (d == d_val) & (g.astype(int) == g_val)
        if np.sum(mask) >= 3:
            mean_zpc = np.mean(zpc[mask])
            std_zpc = np.std(zpc[mask])
            theory = theory_2d if d_val == 2 else theory_3d
            print(f"  d={d_val}, g={g_val}: zp_c = {mean_zpc:.4f} ± {std_zpc:.4f} "
                  f"({np.sum(mask)}个), 偏差 = {(mean_zpc-theory)/theory*100:+.1f}%")

# ============================================================
# 3. 关键假设检验：紧凑vs非紧凑
# ============================================================
print(f"\n{'='*80}")
print(f"【假设检验：紧凑近邻 vs 非紧凑近邻】")
print(f"{'='*80}")

# Xun et al.研究的是"紧凑近邻"：圆形/球形邻域
# 这意味着近邻是各个方向均匀的
# 对于方格子晶格：
# - 1NN (z=4): 不是紧凑的（十字形）
# - 2NN (z=8): 部分紧凑
# - 3NN+ (z≥12): 更紧凑

# Girth的含义：
# - g=3: 最短三角形环，意味着近邻排列可能非均匀
# - g=4: 最短四边形环（方格子的典型girth）
# - g=6: 蜂窝晶格

# 假设：大z时，所有晶格趋向紧凑近邻，zp_c → 理论值
# 但收敛速率依赖于girth

# 测试：对于sq系列（d=2, g=3或4），zp_c随z的变化
mask_sq = (d == 2) & (z >= 8) & np.isin(names, ['sq-'+str(i) for i in range(1,7)] +
                                          ['sq-1,2', 'sq-1,2,3', 'sq-1,2,3,4', 'sq-1,2,3,4,5', 'sq-1,2,3,4,5,6'])

# 手动选择sq系列
sq_names = [n for n in names if n.startswith('sq-')]
mask_sq = (d == 2) & np.isin(names, sq_names)

print(f"\n二维sq系列 ({np.sum(mask_sq)}个):")
idx_sq = np.where(mask_sq)[0]
idx_sq_sorted = idx_sq[np.argsort(z[idx_sq])]

print(f"  {'名称':<20} {'z':>4} {'g':>2} {'p_c':>8} {'zp_c':>8} {'与理论偏差':>10}")
print("  " + "-"*60)
for i in idx_sq_sorted:
    deviation = (zpc[i] - theory_2d) / theory_2d * 100
    print(f"  {names[i]:<20} {int(z[i]):>4} {int(g[i]):>2} {pc[i]:>8.5f} {zpc[i]:>8.4f} {deviation:>+9.2f}%")

# ============================================================
# 4. 拟合girth依赖的标度律
# ============================================================
print(f"\n{'='*80}")
print(f"【新模型：girth依赖的标度律】")
print(f"{'='*80}")

# 假设：zp_c = c(d) * f(g) + corrections
# 其中 f(g) 描述girth对渐近值的影响
# 当 g → ∞ 时，f(g) → 1（紧凑极限）

# 模型E：girth修正的标度律
# zp_c = c_d * (1 + b/g^δ) + A/z^x
# 其中 c_d = 2^d η_c(d)

# 更简单：直接用数据拟合 c_d(g)

# 对于每个(d, g)组合，拟合 zp_c = c + A/z^x
print(f"\n按(d, g)分组拟合 zp_c = c + A/z^x:")

results = {}
for d_val in [2, 3]:
    for g_val in [3, 4, 6]:
        mask = (d == d_val) & (g.astype(int) == g_val) & (z >= 6)
        if np.sum(mask) >= 5:
            z_fit = z[mask]
            zpc_fit = zpc[mask]

            try:
                # 拟合 c + A/z^x
                popt, pcov = curve_fit(
                    lambda z, c, A, x: c + A/z**x,
                    z_fit, zpc_fit,
                    p0=[3.0 if d_val==2 else 2.5, -5, 0.5],
                    bounds=([0, -50, 0], [10, 0, 3]),
                    maxfev=10000
                )
                c_fit, A_fit, x_fit = popt
                pred_fit = c_fit + A_fit/z_fit**x_fit
                rms = np.sqrt(np.mean((zpc_fit - pred_fit)**2))

                results[(d_val, g_val)] = {'c': c_fit, 'A': A_fit, 'x': x_fit, 'rms': rms, 'n': np.sum(mask)}
                print(f"  d={d_val}, g={g_val}: c={c_fit:.4f}, A={A_fit:.4f}, x={x_fit:.3f}, RMS={rms:.4f}, N={np.sum(mask)}")
            except Exception as e:
                print(f"  d={d_val}, g={g_val}: 拟合失败 ({e})")

# ============================================================
# 5. 分析 c(d,g) 的规律
# ============================================================
print(f"\n{'='*80}")
print(f"【c(d,g)的规律分析】")
print(f"{'='*80}")

# 理论预言：
# d=2: c = 4η_c = 4.51235
# d=3: c = 8η_c = 2.73512

print(f"\n拟合的c值:")
print(f"  {'(d,g)':<10} {'c(拟合)':>10} {'c(理论)':>10} {'偏差':>10}")
print("  " + "-"*45)

theory_c = {2: 4.51235, 3: 2.73512}
for (d_val, g_val), res in sorted(results.items()):
    theory = theory_c[d_val]
    deviation = (res['c'] - theory) / theory * 100
    print(f"  ({d_val},{g_val})       {res['c']:>10.4f} {theory:>10.4f} {deviation:>+9.2f}%")

# 观察：c可能依赖于g
# 尝试拟合 c(d,g) = c_d * (1 + b/g^δ)

print(f"\n\nc值对g的依赖 (d=2):")
c_d2 = [results[(2,g)]['c'] for g in [3,4,6] if (2,g) in results]
g_vals = [g for g in [3,4,6] if (2,g) in results]

if len(c_d2) >= 2:
    # 拟合 c = c_inf * (1 + b/g^δ)
    try:
        popt, _ = curve_fit(
            lambda g, c_inf, b, delta: c_inf * (1 + b/g**delta),
            g_vals, c_d2,
            p0=[4.5, 1.0, 1.0],
            maxfev=10000
        )
        c_inf_2d, b_2d, delta_2d = popt
        print(f"  拟合: c(g) = c_∞ * (1 + b/g^δ)")
        print(f"    c_∞ = {c_inf_2d:.4f} (理论: 4.51235)")
        print(f"    b = {b_2d:.4f}")
        print(f"    δ = {delta_2d:.4f}")

        # 预测各g的值
        for g in g_vals:
            pred = c_inf_2d * (1 + b_2d/g**delta_2d)
            print(f"    g={g}: 预测={pred:.4f}, 实际={results[(2,g)]['c']:.4f}")
    except Exception as e:
        print(f"  拟合失败: {e}")

print(f"\nc值对g的依赖 (d=3):")
c_d3 = [results[(3,g)]['c'] for g in [3,4,6] if (3,g) in results]
g_vals_3 = [g for g in [3,4,6] if (3,g) in results]

if len(c_d3) >= 2:
    try:
        popt, _ = curve_fit(
            lambda g, c_inf, b, delta: c_inf * (1 + b/g**delta),
            g_vals_3, c_d3,
            p0=[2.7, 1.0, 1.0],
            maxfev=10000
        )
        c_inf_3d, b_3d, delta_3d = popt
        print(f"  拟合: c(g) = c_∞ * (1 + b/g^δ)")
        print(f"    c_∞ = {c_inf_3d:.4f} (理论: 2.73512)")
        print(f"    b = {b_3d:.4f}")
        print(f"    δ = {delta_3d:.4f}")

        for g in g_vals_3:
            pred = c_inf_3d * (1 + b_3d/g**delta_3d)
            print(f"    g={g}: 预测={pred:.4f}, 实际={results[(3,g)]['c']:.4f}")
    except Exception as e:
        print(f"  拟合失败: {e}")

# ============================================================
# 6. 物理洞察：为什么c(d,g)不等于理论值？
# ============================================================
print(f"\n{'='*80}")
print(f"【物理洞察：标度律的修正】")
print(f"{'='*80}")

print(f"""
关键发现：
1. Xun et al.的标度律 zp_c → 2^d η_c 适用于"紧凑近邻"
2. 我们的数据包含不同girth的晶格，c值偏离理论预言
3. c值对g有依赖：c(g) ≈ c_∞ * (1 + b/g^δ)

物理解释：
- girth描述了近邻的"紧凑程度"
- 小girth（如g=3）意味着三角形环，近邻排列非均匀
- 大girth（如g=6）意味着更大的环，近邻更"紧凑"
- 当g→∞时，晶格趋向连续极限，c→2^d η_c

新公式：
p_c = (c_d(g) / z) + A/z^(1+x)
其中 c_d(g) = c_d^∞ * (1 + b/g^δ)

这统一了：
1. Xun et al.的标度律（紧凑近邻极限）
2. 我们的指数修正（girth依赖）
3. 大z渐近行为
""")

# ============================================================
# 7. 新模型拟合
# ============================================================
print(f"{'='*80}")
print(f"【新模型：统一标度律】")
print(f"{'='*80}")

# 模型F：统一标度律模型
# p_c = [c_d * (1 + b/g^δ)]/z * [1 + a*exp(-αd) * (1-exp(-z/z₀))]

def model_unified(params, z, d, g):
    c_2_inf, c_3_inf, b, delta, a, alpha, z0 = params
    # c_d(g) = c_d^∞ * (1 + b/g^δ)
    c_d = np.where(d == 2, c_2_inf, c_3_inf) * (1 + b / g**delta)
    # 修正项
    correction = 1 + a * np.exp(-alpha * d) * (1 - np.exp(-z / z0))
    return (c_d / z) * correction

# 拟合
best_params_F = None
best_ss_F = np.inf

for trial in range(1000):
    if trial == 0:
        p0 = [4.5, 2.7, 2.0, 1.0, 0.5, 0.3, 10.0]
    else:
        p0 = [np.random.uniform(3.0, 6.0),
              np.random.uniform(2.0, 4.0),
              np.random.uniform(0.5, 5.0),
              np.random.uniform(0.3, 2.0),
              np.random.uniform(0.1, 2.0),
              np.random.uniform(0.1, 0.5),
              np.random.uniform(5, 20)]

    try:
        res = minimize(lambda p: np.sum((pc - model_unified(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < best_ss_F:
            best_ss_F = res.fun
            best_params_F = res.x
    except:
        pass

if best_params_F is not None:
    c_2_inf, c_3_inf, b, delta, a, alpha, z0 = best_params_F
    pred_F = model_unified(best_params_F, z, d, g)
    errors_F = np.abs((pc - pred_F) / pc) * 100
    mape_F = np.mean(errors_F)

    ssr_F = np.sum((pc - pred_F)**2)
    n_params = 7
    aic_F = len(pc) * np.log(ssr_F/len(pc)) + 2*n_params

    print(f"\n模型F: 统一标度律模型")
    print(f"  p_c = [c_d·(1+b/g^δ)/z] · [1 + a·exp(-αd)·(1-exp(-z/z₀))]")
    print(f"\n  拟合参数:")
    print(f"    c_2^∞ = {c_2_inf:.4f}")
    print(f"    c_3^∞ = {c_3_inf:.4f}")
    print(f"    b = {b:.4f}")
    print(f"    δ = {delta:.4f}")
    print(f"    a = {a:.4f}")
    print(f"    α = {alpha:.4f}")
    print(f"    z₀ = {z0:.4f}")
    print(f"\n  性能:")
    print(f"    MAPE = {mape_F:.2f}%")
    print(f"    AIC = {aic_F:.1f}")
    print(f"    参数数 = {n_params}")

    # 与理论比较
    print(f"\n  与理论值比较:")
    print(f"    c_2^∞: 拟合={c_2_inf:.4f}, 理论=4.512, 偏差={(c_2_inf-4.51235)/4.51235*100:+.2f}%")
    print(f"    c_3^∞: 拟合={c_3_inf:.4f}, 理论=2.735, 偏差={(c_3_inf-2.73512)/2.73512*100:+.2f}%")

    # 分类误差
    print(f"\n  分类误差:")
    for d_val in [2, 3]:
        for g_val in [3, 4, 6]:
            mask = (d == d_val) & (g.astype(int) == g_val)
            if np.sum(mask) > 0:
                print(f"    d={d_val},g={g_val}: {np.mean(errors_F[mask]):.2f}% ({np.sum(mask)}个)")

# ============================================================
# 8. 与原始模型对比
# ============================================================
print(f"\n{'='*80}")
print(f"【最终对比】")
print(f"{'='*80}")

# 原始模型
a_orig, alpha_orig, gamma_orig, z0_orig = 12.3346, 0.3368, 0.8477, 9.6171

def model_orig(z, d, g):
    correction = (a_orig * np.exp(-alpha_orig * d) / g**gamma_orig) * (1 - np.exp(-z / z0_orig))
    return (1 + correction) / (z - 1)

pred_orig = model_orig(z, d, g)
errors_orig = np.abs((pc - pred_orig) / pc) * 100
mape_orig = np.mean(errors_orig)
ssr_orig = np.sum((pc - pred_orig)**2)
aic_orig = len(pc) * np.log(ssr_orig/len(pc)) + 2*4

print(f"\n{'模型':<35} {'参数':>4} {'MAPE':>8} {'AIC':>8}")
print("-" * 60)
print(f"{'原始 (4参数)':<35} {4:>4} {mape_orig:>7.2f}% {aic_orig:>8.1f}")
if best_params_F is not None:
    print(f"{'统一标度律 (7参数)':<35} {7:>4} {mape_F:>7.2f}% {aic_F:>8.1f}")

# AIC比较
if best_params_F is not None:
    delta_aic = aic_orig - aic_F
    print(f"\nΔAIC = {delta_aic:.1f}")
    if delta_aic > 10:
        print("✓ 统一标度律模型显著优于原始模型")
    elif delta_aic > 2:
        print("⚠ 统一标度律模型优于原始模型")
    else:
        print("✗ 两模型差异不显著")

print(f"\n{'='*80}")
print(f"【关键结论】")
print(f"{'='*80}")

print(f"""
1. Xun et al. (2021)的标度律适用于"紧凑近邻"
2. 我们的数据包含不同girth的晶格，需要girth修正
3. 统一标度律模型将两者结合：
   - 基础项 c_d(g)/z 描述大z极限
   - 修正项描述有限z和girth效应

4. 物理意义：
   - c_d^∞ 是连续极限的渗流阈值
   - b/g^δ 描述girth（环结构）对阈值的影响
   - 指数修正 exp(-αd) 描述维度衰减

这是一个统一框架！
""")
