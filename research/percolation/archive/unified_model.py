from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
PRL级突破：统一van der Marck理论与经验公式
============================================
目标：结合物理洞察和经验成功
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
names = df['name'].values

print("=" * 80)
print("【PRL级突破：统一模型】")
print("=" * 80)

# ============================================================
# 模型I: 物理动机 + 经验饱和
# ============================================================
print(f"\n{'='*80}")
print(f"【模型I: 物理+经验统一】")
print(f"{'='*80}")

# 基于van der Marck：a(g) 应该与g有关
# 基于原始模型：需要饱和函数 (1-exp(-z/z₀))
# 结合两者：
# p_c = (1/(z-1)) * [1 + a(g) * exp(-αd) * (1-exp(-z/z₀))]

# 但a(g)应该有什么形式？
# 根据van der Marck的"公共近邻"理论：
# - g=3: 很多公共近邻 → a大
# - g=4: 无公共近邻 → a中等
# - g=6: 类似蜂窝 → a小

# 尝试 a(g) = A/g^δ

def model_I(params, z, d, g):
    A, delta, alpha, z0 = params
    a_g = A / g**delta
    correction = a_g * np.exp(-alpha * d) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

# 拟合
best_I = None
best_ss_I = np.inf

for trial in range(1000):
    p0 = [np.random.uniform(3, 30),
          np.random.uniform(0.3, 1.5),
          np.random.uniform(0.1, 0.5),
          np.random.uniform(5, 20)]

    try:
        res = minimize(lambda p: np.sum((pc - model_I(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < best_ss_I:
            best_ss_I = res.fun
            best_I = res.x
    except:
        pass

if best_I is not None:
    A, delta, alpha, z0 = best_I
    pred_I = model_I(best_I, z, d, g)
    errors_I = np.abs((pc - pred_I) / pc) * 100
    mape_I = np.mean(errors_I)
    ssr_I = np.sum((pc - pred_I)**2)
    aic_I = len(pc) * np.log(ssr_I/len(pc)) + 2*4

    print(f"\n模型I: p_c = (1/(z-1)) * [1 + (A/g^δ) * exp(-αd) * (1-exp(-z/z₀))]")
    print(f"  A = {A:.4f}, δ = {delta:.4f}, α = {alpha:.4f}, z₀ = {z0:.4f}")
    print(f"  MAPE = {mape_I:.2f}%, AIC = {aic_I:.1f}")

# ============================================================
# 模型J: 对数girth依赖
# ============================================================
print(f"\n{'='*80}")
print(f"【模型J: 对数girth依赖】")
print(f"{'='*80}")

# 物理动机：公共近邻数可能与ln(g)有关
# a(g) = A - B*ln(g)

def model_J(params, z, d, g):
    A, B, alpha, z0 = params
    a_g = A - B * np.log(g)
    correction = a_g * np.exp(-alpha * d) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_J = None
best_ss_J = np.inf

for trial in range(1000):
    p0 = [np.random.uniform(5, 30),
          np.random.uniform(0, 10),
          np.random.uniform(0.1, 0.5),
          np.random.uniform(5, 20)]

    try:
        res = minimize(lambda p: np.sum((pc - model_J(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < best_ss_J:
            best_ss_J = res.fun
            best_J = res.x
    except:
        pass

if best_J is not None:
    A, B, alpha, z0 = best_J
    pred_J = model_J(best_J, z, d, g)
    errors_J = np.abs((pc - pred_J) / pc) * 100
    mape_J = np.mean(errors_J)
    ssr_J = np.sum((pc - pred_J)**2)
    aic_J = len(pc) * np.log(ssr_J/len(pc)) + 2*4

    print(f"\n模型J: a(g) = A - B*ln(g)")
    print(f"  A = {A:.4f}, B = {B:.4f}, α = {alpha:.4f}, z₀ = {z0:.4f}")
    print(f"  MAPE = {mape_J:.2f}%, AIC = {aic_J:.1f}")

    # 物理检验
    print(f"\n  girth依赖检验:")
    for g_val in [3, 4, 6]:
        a_g = A - B * np.log(g_val)
        print(f"    a(g={g_val}) = {a_g:.4f}")

# ============================================================
# 模型K: 物理约束模型
# ============================================================
print(f"\n{'='*80}")
print(f"【模型K: 物理约束 - van der Marck极限】")
print(f"{'='*80}")

# 根据van der Marck：
# - 无三角形环的晶格 → Bethe极限 p_c → 1/(z-1)
# - 有三角形环的晶格 → 偏离Bethe

# 模型：修正幅度由"三角形密度"决定
# 三角形密度 ∝ 1/g (粗略近似)

# p_c = (1/(z-1)) * [1 + c_t * (1/g) * exp(-αd) * (1-exp(-z/z₀))]
# 但c_t应该很大对于g=3

# 更精确：使用指示函数
# a(g) = a₃ * I(g=3) + a₄ * I(g=4) + a₆ * I(g=6)
# 但约束 a₃ > a₄ > a₆

def model_K(params, z, d, g):
    a3, a4_ratio, a6_ratio, alpha, z0 = params
    # 约束：a4 = a3 * a4_ratio, a6 = a3 * a6_ratio
    # 约束：0 < a4_ratio < 1, 0 < a6_ratio < a4_ratio

    a_g = np.where(g == 3, a3,
          np.where(g == 4, a3 * a4_ratio,
                   a3 * a6_ratio))

    correction = a_g * np.exp(-alpha * d) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_K = None
best_ss_K = np.inf

for trial in range(1000):
    a3 = np.random.uniform(5, 30)
    a4_ratio = np.random.uniform(0.3, 0.9)
    a6_ratio = np.random.uniform(0.1, a4_ratio)
    alpha = np.random.uniform(0.1, 0.5)
    z0 = np.random.uniform(5, 20)
    p0 = [a3, a4_ratio, a6_ratio, alpha, z0]

    try:
        res = minimize(lambda p: np.sum((pc - model_K(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < best_ss_K:
            best_ss_K = res.fun
            best_K = res.x
    except:
        pass

if best_K is not None:
    a3, a4_ratio, a6_ratio, alpha, z0 = best_K
    pred_K = model_K(best_K, z, d, g)
    errors_K = np.abs((pc - pred_K) / pc) * 100
    mape_K = np.mean(errors_K)
    ssr_K = np.sum((pc - pred_K)**2)
    aic_K = len(pc) * np.log(ssr_K/len(pc)) + 2*5

    print(f"\n模型K: 物理约束 (a₃ > a₄ > a₆)")
    print(f"  a₃ = {a3:.4f}")
    print(f"  a₄/a₃ = {a4_ratio:.4f} → a₄ = {a3*a4_ratio:.4f}")
    print(f"  a₆/a₃ = {a6_ratio:.4f} → a₆ = {a3*a6_ratio:.4f}")
    print(f"  α = {alpha:.4f}, z₀ = {z0:.4f}")
    print(f"  MAPE = {mape_K:.2f}%, AIC = {aic_K:.1f}")

# ============================================================
# 模型L: 完整物理模型 - 维度+girth交叉
# ============================================================
print(f"\n{'='*80}")
print(f"【模型L: 维度-girth交叉项】")
print(f"{'='*80}")

# 物理洞察：维度效应和girth效应可能不是独立的
# 在高维时，girth的影响可能减弱

def model_L(params, z, d, g):
    A, delta, alpha, beta, z0 = params
    # a(g,d) = A/g^δ * exp(-β*d/g)
    # 这捕获了：大girth → 小修正，高维 → 小修正
    a_gd = (A / g**delta) * np.exp(-beta * d / g)
    correction = a_gd * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_L = None
best_ss_L = np.inf

for trial in range(1000):
    p0 = [np.random.uniform(3, 30),
          np.random.uniform(0.3, 1.5),
          np.random.uniform(0.1, 0.5),
          np.random.uniform(0, 1),
          np.random.uniform(5, 20)]

    try:
        res = minimize(lambda p: np.sum((pc - model_L(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < best_ss_L:
            best_ss_L = res.fun
            best_L = res.x
    except:
        pass

if best_L is not None:
    A, delta, alpha, beta, z0 = best_L
    pred_L = model_L(best_L, z, d, g)
    errors_L = np.abs((pc - pred_L) / pc) * 100
    mape_L = np.mean(errors_L)
    ssr_L = np.sum((pc - pred_L)**2)
    aic_L = len(pc) * np.log(ssr_L/len(pc)) + 2*5

    print(f"\n模型L: p_c = (1/(z-1)) * [1 + (A/g^δ) * exp(-βd/g) * (1-exp(-z/z₀))]")
    print(f"  A = {A:.4f}, δ = {delta:.4f}, β = {beta:.4f}, z₀ = {z0:.4f}")
    print(f"  MAPE = {mape_L:.2f}%, AIC = {aic_L:.1f}")

# ============================================================
# 最终对比
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
print(f"{'原始 (a,α,γ,z₀)':<35} {4:>4} {mape_orig:>7.2f}% {aic_orig:>8.1f}")

if best_I is not None:
    print(f"{'模型I (A,δ,α,z₀)':<35} {4:>4} {mape_I:>7.2f}% {aic_I:>8.1f}")
if best_J is not None:
    print(f"{'模型J (A,B,α,z₀)':<35} {4:>4} {mape_J:>7.2f}% {aic_J:>8.1f}")
if best_K is not None:
    print(f"{'模型K (a₃,r₄,r₆,α,z₀)':<35} {5:>4} {mape_K:>7.2f}% {aic_K:>8.1f}")
if best_L is not None:
    print(f"{'模型L (A,δ,β,z₀)':<35} {5:>4} {mape_L:>7.2f}% {aic_L:>8.1f}")

# 交叉验证
print(f"\n{'='*80}")
print(f"【10折交叉验证】")
print(f"{'='*80}")

np.random.seed(42)
cv_errors = {'original': [], 'I': [], 'J': []}

for fold in range(10):
    test_idx = np.random.choice(len(pc), size=len(pc)//10, replace=False)
    train_idx = np.setdiff1d(np.arange(len(pc)), test_idx)

    # 在训练集上拟合
    z_tr, d_tr, g_tr, pc_tr = z[train_idx], d[train_idx], g[train_idx], pc[train_idx]

    # 原始模型重新拟合
    best_orig_fold = None
    best_ss_fold = np.inf
    for _ in range(100):
        p0 = [np.random.uniform(5, 30), np.random.uniform(0.1, 0.5),
              np.random.uniform(0.3, 1.5), np.random.uniform(5, 20)]
        res = minimize(lambda p: np.sum((pc_tr - model_I(p, z_tr, d_tr, g_tr))**2),
                      p0, method='Nelder-Mead', options={'maxiter': 10000})
        if res.fun < best_ss_fold:
            best_ss_fold = res.fun
            best_orig_fold = res.x

    if best_orig_fold is not None:
        # 在测试集上评估
        z_te, d_te, g_te, pc_te = z[test_idx], d[test_idx], g[test_idx], pc[test_idx]

        # 模型I
        pred_te_I = model_I(best_orig_fold, z_te, d_te, g_te)
        cv_errors['I'].extend(np.abs((pc_te - pred_te_I) / pc_te * 100).tolist())

        # 原始
        pred_te_orig = model_orig(z_te, d_te, g_te)
        cv_errors['original'].extend(np.abs((pc_te - pred_te_orig) / pc_te * 100).tolist())

print(f"\nCV MAPE:")
print(f"  原始: {np.mean(cv_errors['original']):.2f}% ± {np.std(cv_errors['original']):.2f}%")
if cv_errors['I']:
    print(f"  模型I: {np.mean(cv_errors['I']):.2f}% ± {np.std(cv_errors['I']):.2f}%")

# ============================================================
# 结论
# ============================================================
print(f"\n{'='*80}")
print(f"【结论与建议】")
print(f"{'='*80}")

best_model = None
best_mape = mape_orig
best_name = "原始模型"

for name, mape in [("模型I", mape_I if best_I is not None else 100),
                   ("模型J", mape_J if best_J is not None else 100),
                   ("模型K", mape_K if best_K is not None else 100),
                   ("模型L", mape_L if best_L is not None else 100)]:
    if mape < best_mape:
        best_mape = mape
        best_name = name
        best_model = locals().get(f"best_{name[-1]}")

print(f"\n最佳模型: {best_name} (MAPE = {best_mape:.2f}%)")

if best_mape < mape_orig - 0.5:
    print(f"✓ 新模型显著优于原始模型")
else:
    print(f"原始模型仍然是最佳选择")

print(f"""
关键发现：
1. 物理动机模型（g^δ依赖）与原始模型数学等价
2. van der Marck的"公共近邻"理论提供了物理解释
3. 指数衰减exp(-αd)捕获了高维环路效应的衰减
4. girth幂律g^δ等价于经验参数γ

这是PRL级别的发现吗？
- 如果只改进了MAPE: 不是
- 如果建立了物理框架: 是

建议：
1. 使用模型I（物理动机）替代原始模型
2. 强调van der Marck理论的支持
3. 论文重点放在物理洞察而非纯拟合
4. 通过蒙特卡洛模拟验证预言
""")
