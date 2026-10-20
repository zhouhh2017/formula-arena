from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
验证Xun et al. (2021)标度律并与当前模型对比
=============================================
关键发现：
1. 站点渗流大z极限：zp_c ~ 2^d η_c
   - 二维：zp_c → 4η_c = 4.51235
   - 三维：zp_c → 8η_c = 2.7351
2. 键渗流有限z修正：zp_c - 1 ~ z^{-(d-1)/d}
   - 三维：x = 2/3
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

print("=" * 80)
print("【验证Xun et al. (2021)标度律】")
print("=" * 80)

# 加载数据
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values

# 计算 zp_c
zpc = z * pc

# ============================================================
# 1. 测试大z极限标度律
# ============================================================
print(f"\n{'='*80}")
print(f"【测试1：大z极限标度律 zp_c ~ 2^d η_c】")
print(f"{'='*80}")

# 理论预言值
eta_c_2d = 1.128087  # 圆盘连续渗流阈值
eta_c_3d = 0.34189   # 球连续渗流阈值
theory_2d = 4 * eta_c_2d  # = 4.51235
theory_3d = 8 * eta_c_3d  # = 2.7351

print(f"\n理论预言值:")
print(f"  二维: zp_c → 4η_c = {theory_2d:.5f}")
print(f"  三维: zp_c → 8η_c = {theory_3d:.5f}")

# 按维度和z值分类
for d_val in [2, 3]:
    mask_d = d == d_val
    # 只取大z的数据（z >= 20）
    mask_large_z = mask_d & (z >= 20)

    if np.sum(mask_large_z) > 0:
        mean_zpc = np.mean(zpc[mask_large_z])
        std_zpc = np.std(zpc[mask_large_z])
        theory = theory_2d if d_val == 2 else theory_3d
        deviation = (mean_zpc - theory) / theory * 100

        print(f"\n  d={d_val}, z≥20 ({np.sum(mask_large_z)}个):")
        print(f"    均值: zp_c = {mean_zpc:.4f} ± {std_zpc:.4f}")
        print(f"    理论: {theory:.4f}")
        print(f"    偏差: {deviation:+.2f}%")

        # 显示具体数据
        for i in np.where(mask_large_z)[0]:
            if z[i] >= 30:  # 只显示z≥30的
                print(f"      {names[i]:<20} z={int(z[i]):>3}, zp_c={zpc[i]:.4f}")

# ============================================================
# 2. 测试有限z修正的幂律形式
# ============================================================
print(f"\n{'='*80}")
print(f"【测试2：有限z修正形式】")
print(f"{'='*80}")

# Xun et al. 提出：zp_c = c + b/z^x
# 或者等价地：p_c = c/z + b/z^(1+x)

# 对于d=2，测试 c = 4.51235
# 对于d=3，测试 c = 2.7351

# 拟合形式：zp_c = c + A/z^x
def model_zpc(z, c, A, x):
    return c + A / z**x

# d=2 数据
mask_2d = (d == 2) & (z >= 8)
z_2d = z[mask_2d]
zpc_2d = zpc[mask_2d]

# 固定 c = 4.51235，拟合 A 和 x
popt_2d, pcov_2d = curve_fit(lambda z, A, x: 4.51235 + A/z**x, z_2d, zpc_2d,
                              p0=[-1, 0.5], maxfev=10000)
A_2d, x_2d = popt_2d
pred_2d = 4.51235 + A_2d / z_2d**x_2d
residuals_2d = zpc_2d - pred_2d
rms_2d = np.sqrt(np.mean(residuals_2d**2))

print(f"\n二维拟合 (c固定为4.51235):")
print(f"  zp_c = 4.51235 + A/z^x")
print(f"  A = {A_2d:.4f}, x = {x_2d:.4f}")
print(f"  RMS误差 = {rms_2d:.4f}")

# d=3 数据
mask_3d = (d == 3) & (z >= 8)
z_3d = z[mask_3d]
zpc_3d = zpc[mask_3d]

# 固定 c = 2.7351，拟合 A 和 x
popt_3d, pcov_3d = curve_fit(lambda z, A, x: 2.7351 + A/z**x, z_3d, zpc_3d,
                              p0=[-1, 0.5], maxfev=10000)
A_3d, x_3d = popt_3d
pred_3d = 2.7351 + A_3d / z_3d**x_3d
residuals_3d = zpc_3d - pred_3d
rms_3d = np.sqrt(np.mean(residuals_3d**2))

print(f"\n三维拟合 (c固定为2.7351):")
print(f"  zp_c = 2.7351 + A/z^x")
print(f"  A = {A_3d:.4f}, x = {x_3d:.4f}")
print(f"  RMS误差 = {rms_3d:.4f}")

# 与理论 x = (d-1)/d 比较
print(f"\n与理论 x = (d-1)/d 比较:")
print(f"  二维: 拟合 x = {x_2d:.3f}, 理论 x = 0.500, 偏差 = {x_2d-0.5:+.3f}")
print(f"  三维: 拟合 x = {x_3d:.3f}, 理论 x = 0.667, 偏差 = {x_3d-2/3:+.3f}")

# ============================================================
# 3. 基于标度律的新模型
# ============================================================
print(f"\n{'='*80}")
print(f"【基于标度律的新模型设计】")
print(f"{'='*80}")

# 模型A: 直接基于Xun et al.的标度律
# p_c = (c_d + A/z^x) / z
# 其中 c_d = 2^d η_c

# 我们需要 c_d 依赖于维度
# 假设 η_c(d) 是d的函数
# 对于d=2: η_c = 1.128087
# 对于d=3: η_c = 0.34189

# 尝试参数化 η_c(d) = η_0 * exp(-λd)
# 这样可以统一描述维度依赖

def model_scaling(params, z, d):
    """基于标度律的模型"""
    eta_0, lambda_d, A, x = params
    # η_c(d) = η_0 * exp(-λd)
    eta_c_d = eta_0 * np.exp(-lambda_d * d)
    # c_d = 2^d * η_c(d)
    c_d = (2**d) * eta_c_d
    # zp_c = c_d + A/z^x
    return (c_d + A / z**x) / z

# 拟合
best_params = None
best_ss = np.inf

for trial in range(500):
    if trial == 0:
        p0 = [1.0, 0.5, -1.0, 0.5]
    else:
        p0 = [np.random.uniform(0.5, 2.0),
              np.random.uniform(0.3, 1.0),
              np.random.uniform(-3, 0),
              np.random.uniform(0.3, 1.0)]

    try:
        res = minimize(lambda p: np.sum((pc - model_scaling(p, z, d))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.fun < best_ss:
            best_ss = res.fun
            best_params = res.x
    except:
        pass

if best_params is not None:
    eta_0, lambda_d, A, x = best_params
    pred_scaling = model_scaling(best_params, z, d)
    errors_scaling = np.abs((pc - pred_scaling) / pc) * 100
    mape_scaling = np.mean(errors_scaling)

    print(f"\n模型A: 基于标度律")
    print(f"  p_c = (2^d η_c(d) + A/z^x) / z")
    print(f"  其中 η_c(d) = η_0 * exp(-λd)")
    print(f"\n  拟合参数:")
    print(f"    η_0 = {eta_0:.4f}")
    print(f"    λ = {lambda_d:.4f}")
    print(f"    A = {A:.4f}")
    print(f"    x = {x:.4f}")
    print(f"\n  性能:")
    print(f"    MAPE = {mape_scaling:.2f}%")
    print(f"    d=2误差 = {np.mean(errors_scaling[d==2]):.2f}%")
    print(f"    d=3误差 = {np.mean(errors_scaling[d==3]):.2f}%")

# ============================================================
# 4. 混合模型：标度律 + 指数维度修正
# ============================================================
print(f"\n{'='*80}")
print(f"【混合模型：标度律 + 指数维度修正】")
print(f"{'='*80}")

# 结合我们的指数发现和Xun的标度律
# p_c = (1/(z-1)) * [1 + f(d) * g(girth) * S(z)]
# 其中 f(d) 包含指数衰减

# 但修正项应该与 z^{-x} 兼容

# 模型B: 标度律启发的指数模型
def model_hybrid(params, z, d, g):
    """混合模型"""
    eta_0, lambda_d, a, alpha, gamma, z0 = params
    # 基础项：基于标度律
    eta_c_d = eta_0 * np.exp(-lambda_d * d)
    c_d = (2**d) * eta_c_d
    # 修正项：指数维度衰减
    correction = a * np.exp(-alpha * d) / g**gamma * (1 - np.exp(-z / z0))
    # 组合
    return (c_d / z) * (1 + correction / c_d)

# 简化：假设 c_d/z 是主导项
# p_c ≈ c_d/z + correction/z
# 这与我们的原始模型类似，但基础项不同

def model_hybrid_v2(params, z, d, g):
    """混合模型v2"""
    a, alpha, gamma, z0 = params
    # 基础项：Bethe近似 1/(z-1)
    # 修正项：指数维度衰减
    correction = a * np.exp(-alpha * d) / g**gamma * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

# 这实际上是我们的原始模型！
# 所以问题不在于模型形式，而在于参数拟合

# 模型C: 强制满足标度律的模型
# 在大z极限下，p_c → c_d/z
# 这意味着 correction → c_d - 1 当 z → ∞
# 所以 a * exp(-αd) / g^γ = c_d - 1 = 2^d η_c(d) - 1

def model_constrained(params, z, d, g):
    """强制满足标度律的模型"""
    eta_0, lambda_d, gamma, z0 = params
    # η_c(d) = η_0 * exp(-λd)
    eta_c_d = eta_0 * np.exp(-lambda_d * d)
    c_d = (2**d) * eta_c_d
    # 约束：a = c_d - 1 = 2^d η_c(d) - 1
    # 但这样就没有自由度了...

    # 改为：a(d) = 2^d η_c(d) - 1 + 额外修正
    # 但这样太复杂

    # 简单版本：直接用标度律形式
    return c_d / z

# 这个模型太简单，没有girth依赖
# 需要更复杂的模型

# ============================================================
# 5. 关键洞察：为什么原模型在d=2有偏差
# ============================================================
print(f"\n{'='*80}")
print(f"【关键洞察：分析d=2系统性偏差】")
print(f"{'='*80}")

# 我们的模型
a, alpha, gamma, z0 = 12.3346, 0.3368, 0.8477, 9.6171

def model_original(z, d, g):
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

pred_orig = model_original(z, d, g)

# 分析d=2, z大的情况
mask_2d_large = (d == 2) & (z >= 20)
if np.sum(mask_2d_large) > 0:
    print(f"\nd=2, z≥20的数据:")
    for i in np.where(mask_2d_large)[0]:
        z_i = z[i]
        pc_i = pc[i]
        pred_i = pred_orig[i]
        error_i = (pred_i - pc_i) / pc_i * 100

        # 计算 zp_c
        zpc_i = z_i * pc_i
        zpc_pred = z_i * pred_i

        print(f"  {names[i]:<20} z={int(z_i):>3}, p_c={pc_i:.5f}, zp_c={zpc_i:.4f}, "
              f"预测zp_c={zpc_pred:.4f}, 误差={error_i:+.2f}%")

# 理论预言
print(f"\n理论预言: d=2时 zp_c → 4.51235")

# ============================================================
# 6. 新模型：结合标度律和指数衰减
# ============================================================
print(f"\n{'='*80}")
print(f"【新模型：标度律 + girth修正】")
print(f"{'='*80}")

# 模型D: 标度律基础 + girth修正
# zp_c = c_d * [1 + f(girth) * exp(-αd) * (1 - exp(-z/z0))]
# p_c = c_d/z * [1 + f(girth) * exp(-αd) * (1 - exp(-z/z0))]

def model_new_scaling(params, z, d, g):
    """新模型：基于标度律"""
    c_2, c_3, a, alpha, gamma, z0 = params
    # c_d 分段：d=2和d≥3
    c_d = np.where(d == 2, c_2, c_3)
    # 修正项
    correction = a * np.exp(-alpha * d) / g**gamma * (1 - np.exp(-z / z0))
    return (c_d / z) * (1 + correction)

# 拟合
best_params_D = None
best_ss_D = np.inf

for trial in range(500):
    if trial == 0:
        p0 = [4.5, 2.7, 1.0, 0.3, 0.8, 10.0]
    else:
        p0 = [np.random.uniform(4.0, 5.0),
              np.random.uniform(2.5, 3.0),
              np.random.uniform(0.5, 2.0),
              np.random.uniform(0.1, 0.5),
              np.random.uniform(0.5, 1.2),
              np.random.uniform(5, 15)]

    try:
        res = minimize(lambda p: np.sum((pc - model_new_scaling(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.fun < best_ss_D:
            best_ss_D = res.fun
            best_params_D = res.x
    except:
        pass

if best_params_D is not None:
    c_2, c_3, a, alpha, gamma, z0 = best_params_D
    pred_D = model_new_scaling(best_params_D, z, d, g)
    errors_D = np.abs((pc - pred_D) / pc) * 100
    mape_D = np.mean(errors_D)

    ssr_D = np.sum((pc - pred_D)**2)
    aic_D = len(pc) * np.log(ssr_D/len(pc)) + 2*6

    print(f"\n模型D: 标度律基础 + girth修正")
    print(f"  p_c = (c_d/z) * [1 + a·exp(-αd)/g^γ · (1-exp(-z/z₀))]")
    print(f"\n  拟合参数:")
    print(f"    c_2 (d=2) = {c_2:.4f}")
    print(f"    c_3 (d≥3) = {c_3:.4f}")
    print(f"    a = {a:.4f}")
    print(f"    α = {alpha:.4f}")
    print(f"    γ = {gamma:.4f}")
    print(f"    z₀ = {z0:.4f}")
    print(f"\n  性能:")
    print(f"    MAPE = {mape_D:.2f}%")
    print(f"    AIC = {aic_D:.1f}")
    print(f"    d=2误差 = {np.mean(errors_D[d==2]):.2f}%")
    print(f"    d=3误差 = {np.mean(errors_D[d==3]):.2f}%")

    # 与理论值比较
    print(f"\n  与理论值比较:")
    print(f"    c_2: 拟合={c_2:.4f}, 理论=4.51235, 偏差={(c_2-4.51235)/4.51235*100:+.2f}%")
    print(f"    c_3: 拟合={c_3:.4f}, 理论=2.73510, 偏差={(c_3-2.7351)/2.7351*100:+.2f}%")

# ============================================================
# 7. 与原始模型对比
# ============================================================
print(f"\n{'='*80}")
print(f"【对比：原始模型 vs 标度律模型】")
print(f"{'='*80}")

pred_orig = model_original(z, d, g)
errors_orig = np.abs((pc - pred_orig) / pc) * 100

ssr_orig = np.sum((pc - pred_orig)**2)
aic_orig = len(pc) * np.log(ssr_orig/len(pc)) + 2*4

print(f"\n{'模型':<30} {'MAPE':>8} {'AIC':>8} {'d=2误差':>10} {'d=3误差':>10}")
print("-" * 70)
print(f"{'原始模型 (4参数)':<30} {np.mean(errors_orig):>7.2f}% {aic_orig:>8.1f} "
      f"{np.mean(errors_orig[d==2]):>9.2f}% {np.mean(errors_orig[d==3]):>9.2f}%")

if best_params_D is not None:
    print(f"{'标度律模型 (6参数)':<30} {mape_D:>7.2f}% {aic_D:>8.1f} "
          f"{np.mean(errors_D[d==2]):>9.2f}% {np.mean(errors_D[d==3]):>9.2f}%")

print(f"\n结论:")
if mape_D < np.mean(errors_orig):
    print(f"  ✓ 标度律模型MAPE改善 {np.mean(errors_orig) - mape_D:.2f}%")
else:
    print(f"  ✗ 标度律模型MAPE恶化 {mape_D - np.mean(errors_orig):.2f}%")

# ============================================================
# 8. 保存发现
# ============================================================
print(f"\n{'='*80}")
print(f"【保存关键发现】")
print(f"{'='*80}")

with open(f'{OUT}/SCALING_LAW_DISCOVERY.md', 'w', encoding='utf-8') as f:
    f.write("# 标度律发现报告\n\n")
    f.write("> 日期: 2026-10-04\n")
    f.write("> 关键文献: Xun, Hao & Ziff (2021), arXiv:2111.10975\n\n")

    f.write("## 关键发现\n\n")
    f.write("### 1. Xun et al. (2021)的标度律\n\n")
    f.write("站点渗流大z极限：\n")
    f.write("- 二维：zp_c → 4η_c = 4.51235\n")
    f.write("- 三维：zp_c → 8η_c = 2.7351\n")
    f.write("- 一般：zp_c → 2^d η_c\n\n")

    f.write("键渗流有限z修正：\n")
    f.write("- zp_c - 1 ~ z^{-(d-1)/d}\n")
    f.write("- 三维：x = 2/3\n\n")

    f.write("### 2. 我们的模型与标度律的关系\n\n")
    f.write("原始模型：p_c = (1/(z-1)) × [1 + correction]\n")
    f.write("在大z极限下：p_c → (1 + a·exp(-αd)/g^γ) / z\n\n")
    f.write("这意味着 zp_c → 1 + a·exp(-αd)/g^γ\n")
    f.write("但标度律预言 zp_c → 2^d η_c(d)\n\n")

    f.write("**关键矛盾**：我们的模型在大z极限下给出的是常数，")
    f.write("但这个常数应该等于 2^d η_c(d)，而我们的参数没有这个约束。\n\n")

    f.write("### 3. 新模型方向\n\n")
    f.write("模型D：p_c = (c_d/z) × [1 + correction]\n")
    f.write("其中 c_d 由标度律确定（或拟合）\n")
    f.write("这样可以同时满足：\n")
    f.write("- 大z极限的标度律\n")
    f.write("- 有限z的指数修正\n")
    f.write("- girth依赖\n\n")

    if best_params_D is not None:
        f.write(f"### 4. 初步拟合结果\n\n")
        f.write(f"- c_2 = {c_2:.4f} (理论4.51235)\n")
        f.write(f"- c_3 = {c_3:.4f} (理论2.7351)\n")
        f.write(f"- MAPE = {mape_D:.2f}%\n")
        f.write(f"- AIC = {aic_D:.1f}\n")

print(f"\n保存: SCALING_LAW_DISCOVERY.md")
print(f"\n{'='*80}")
print(f"【分析完成】")
print(f"{'='*80}")
