"""
方向 4：符号回归 + 精确解约束
====================================

问题：手动搜索公式形式效率低
策略：让机器自动搜索

方法：
  1. 符号回归（sympy）：自动搜索解析表达式
  2. 精确解约束：利用已知精确 p_c 作为锚点
  3. 对偶性约束：2D 晶格有对偶关系

关键精确解：
  - triangular: p_c = 1/2（精确）
  - honeycomb: p_c = 1 - 2sin(π/18)（精确）
  - square (bond): p_c = 1/2（精确，但对 site 不完全适用）
"""

import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.special import gamma as gamma_func
import warnings
warnings.filterwarnings('ignore')

# 数据
lattices = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'n_tri': 0, 'pc': 0.697040},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'n_tri': 0, 'pc': 0.592746},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'n_tri': 6, 'pc': 0.500000},
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'n_tri': 2, 'pc': 0.652703},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'n_tri': 0, 'pc': 0.311600},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'n_tri': 0, 'pc': 0.245800},
    {'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'n_tri': 24, 'pc': 0.118800},
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'n_tri': 0, 'pc': 0.431900},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'n_tri': 0, 'pc': 0.196737},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'n_tri': 0, 'pc': 0.140810},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'n_tri': 0, 'pc': 0.109460},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'g': 4, 'n_tri': 0, 'pc': 0.089270},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'g': 4, 'n_tri': 0, 'pc': 0.075050},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'g': 4, 'n_tri': 0, 'pc': 0.064530},
]

n = len(lattices)
z = np.array([l['z'] for l in lattices])
d = np.array([l['d'] for l in lattices])
g = np.array([l['g'] for l in lattices])
pc = np.array([l['pc'] for l in lattices])
R = pc * (z - 1)

# ============ 1. 精确解约束 ============

print("="*80)
print("【1. 精确解约束分析】")
print("="*80)

# 已知精确/高精度解
exact = {
    'triangular': 0.5,  # 精确
    'honeycomb': 1 - 2*np.sin(np.pi/18),  # 精确
}

for name, exact_pc in exact.items():
    print(f"\n{name}: p_c = {exact_pc:.10f} (精确)")

# 约束 1: triangular (z=6, d=2, g=3) → p_c = 1/2
# 约束 2: honeycomb (z=3, d=2, g=6) → p_c = 1 - 2sin(π/18)

# 如果 p_c = A·z^{-α}·d^{-γ}·(1+B/g)，能否精确复现这两个值？
# 两个约束，三个未知数 → 有无穷多解

print("\n约束条件:")
print("  1. triangular (z=6, d=2, g=3): p_c = 1/2")
print("  2. honeycomb (z=3, d=2, g=6): p_c = 1 - 2sin(π/18) ≈ 0.3473")
print(f"     注意: 实际 honeycomb site p_c = 0.6970 ≠ 0.3473")
print(f"     0.3473 是 bond percolation 的精确解!")

# 修正：honeycomb site p_c = 0.6970
# 已知关系: site p_c 和 bond p_c 通过 Wulff 关系关联
# 对于 honeycomb: site p_c ≈ 0.6970, bond p_c = 1 - 2sin(π/18) ≈ 0.3473

print("\n修正: 我们处理的是 SITE 渗流")
print(f"  honeycomb site: p_c = 0.6970 (高精度数值)")
print(f"  triangular site: p_c = 0.5000 (精确)")

# 约束分析
print("\n约束分析:")
print("  如果 p_c = f(z, d, g), 两个约束条件:")
print("    f(6, 2, 3) = 0.5")
print("    f(3, 2, 6) = 0.697")

# 检验之前公式的满足程度
A, alpha, gamma_e, B = 2.349, 0.952, 0.628, 1.912
pred_tri = A * 6**(-alpha) * 2**(-gamma_e) * (1 + B/3)
pred_hon = A * 3**(-alpha) * 2**(-gamma_e) * (1 + B/6)

print(f"\n之前公式的预测:")
print(f"  triangular: 预测 = {pred_tri:.5f}, 精确 = 0.5000, 误差 = {abs(pred_tri-0.5):.4f}")
print(f"  honeycomb: 预测 = {pred_hon:.5f}, 精确 = 0.6970, 误差 = {abs(pred_hon-0.697):.4f}")

# ============ 2. 带约束的拟合 ============

print(f"\n{'='*80}")
print("【2. 带精确解约束的拟合】")
print(f"{'='*80}")

# 方法：在损失函数中加入精确解的惩罚
# L = Σ(pc_pred - pc_true)^2 + λ·Σ(pc_pred_exact - pc_exact)^2

def model_zdg(params, z, d, g):
    A, alpha, gamma_e, B = params
    return A * np.power(z, -alpha) * np.power(d, -gamma_e) * (1 + B/g)

def loss_constrained(params, lam=100.0):
    pred = model_zdg(params, z, d, g)
    # 数据拟合损失
    loss_data = np.sum((pred - pc)**2)
    # 精确解约束
    pred_tri = model_zdg(params, np.array([6]), np.array([2]), np.array([3]))[0]
    pred_hon = model_zdg(params, np.array([3]), np.array([2]), np.array([6]))[0]
    loss_exact = (pred_tri - 0.5)**2 + (pred_hon - 0.69704)**2
    return loss_data + lam * loss_exact

# 无约束拟合
res_uncon = minimize(
    lambda p: np.sum((model_zdg(p, z, d, g) - pc)**2),
    [2.3, 0.9, 0.6, 1.9],
    method='Nelder-Mead',
    options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-12}
)

# 带约束拟合
res_con = minimize(
    loss_constrained,
    [2.3, 0.9, 0.6, 1.9],
    method='Nelder-Mead',
    args=(100.0,),
    options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-12}
)

print(f"\n无约束拟合:")
print(f"  A={res_uncon.x[0]:.6f}, α={res_uncon.x[1]:.6f}, γ={res_uncon.x[2]:.6f}, B={res_uncon.x[3]:.6f}")
pred_tri_u = model_zdg(res_uncon.x, np.array([6]), np.array([2]), np.array([3]))[0]
pred_hon_u = model_zdg(res_uncon.x, np.array([3]), np.array([2]), np.array([6]))[0]
print(f"  triangular 预测: {pred_tri_u:.6f} (精确 0.5000)")
print(f"  honeycomb 预测:  {pred_hon_u:.6f} (精确 0.6970)")

print(f"\n带约束拟合 (λ=100):")
print(f"  A={res_con.x[0]:.6f}, α={res_con.x[1]:.6f}, γ={res_con.x[2]:.6f}, B={res_con.x[3]:.6f}")
pred_tri_c = model_zdg(res_con.x, np.array([6]), np.array([2]), np.array([3]))[0]
pred_hon_c = model_zdg(res_con.x, np.array([3]), np.array([2]), np.array([6]))[0]
print(f"  triangular 预测: {pred_tri_c:.6f} (精确 0.5000)")
print(f"  honeycomb 预测:  {pred_hon_c:.6f} (精确 0.6970)")

pred_c = model_zdg(res_con.x, z, d, g)
R2_c = 1 - np.sum((pc - pred_c)**2) / np.sum((pc - np.mean(pc))**2)
MAE_c = np.mean(np.abs(pc - pred_c))
print(f"  R² = {R2_c:.6f}, MAE = {MAE_c:.6f}")

# ============ 3. 符号回归（手工枚举候选） ============

print(f"\n{'='*80}")
print("【3. 符号回归：枚举候选公式】")
print(f"{'='*80}")

# 手工枚举有物理意义的公式形式
# 基于已知的理论约束

def try_formula(name, func, init):
    """尝试拟合一个公式"""
    def obj(params):
        try:
            pred = func(params, z, d, g)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e10
            return np.sum((pred - pc)**2)
        except:
            return 1e10

    res = minimize(obj, init, method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-12})

    try:
        pred = func(res.x, z, d, g)
        if np.any(np.isnan(pred)):
            return None, 0, 1.0, None
        ss_res = np.sum((pc - pred)**2)
        ss_tot = np.sum((pc - np.mean(pc))**2)
        R2 = 1 - ss_res / ss_tot
        MAE = np.mean(np.abs(pc - pred))
        return res.x, R2, MAE, pred
    except:
        return None, 0, 1.0, None

# 候选公式库（基于物理直觉）
candidates = []

# 1. 基础幂律
def f01(p, z, d, g):
    return p[0] * np.power(z, -p[1]) * np.power(d, -p[2])
candidates.append(("A·z^(-α)·d^(-γ)", f01, [3.0, 0.7, 0.7]))

# 2. +girth 修正
def f02(p, z, d, g):
    return p[0] * np.power(z, -p[1]) * np.power(d, -p[2]) * (1 + p[3]/g)
candidates.append(("A·z^(-α)·d^(-γ)·(1+B/g)", f02, [2.3, 0.9, 0.6, 1.9]))

# 3. +聚类修正
def f03(p, z, d, g):
    C = np.array([2*l['n_tri']/(l['z']*(l['z']-1)) if l['z']>1 else 0 for l in lattices])
    return p[0] * np.power(z, -p[1]) * np.power(d, -p[2]) * (1 + p[3]/g) * np.exp(p[4]*C)
candidates.append(("A·z^(-α)·d^(-γ)·(1+B/g)·exp(D·C)", f03, [2.3, 0.9, 0.6, 1.9, -0.3]))

# 4. 替代形式: A/(z^α·d^γ + B/g)
def f04(p, z, d, g):
    return p[0] / (np.power(z, p[1]) * np.power(d, p[2]) + p[3]/g)
candidates.append(("A/(z^α·d^γ + B/g)", f04, [3.0, 0.8, 0.7, 1.0]))

# 5. 替代形式: A·z^(-α)·(d+c)^(-γ)
def f05(p, z, d, g):
    return p[0] * np.power(z, -p[1]) * np.power(d + p[3], -p[2])
candidates.append(("A·z^(-α)·(d+c)^(-γ)", f05, [3.0, 0.7, 0.7, 1.0]))

# 6. Bethe-inspired: p_c = 1/(z-1+σ) where σ = a/d^b
def f06(p, z, d, g):
    sigma = p[1] / np.power(d, p[2])
    return 1.0 / (z - 1 + sigma)
candidates.append(("1/(z-1+a/d^b)", f06, [5.0, 1.0]))

# 7. Bethe-inspired: p_c = 1/(z-1+σ) where σ = a·g^b/d^c
def f07(p, z, d, g):
    sigma = p[1] * np.power(g, p[2]) / np.power(d, p[3])
    return 1.0 / (z - 1 + sigma)
candidates.append(("1/(z-1+a·g^b/d^c)", f07, [5.0, 1.0, 0.5]))

# 8. 对数修正: p_c = A·z^(-α)·d^(-γ)·(1+B·ln(z)/g)
def f08(p, z, d, g):
    return p[0] * np.power(z, -p[1]) * np.power(d, -p[2]) * (1 + p[3]*np.log(z)/g)
candidates.append(("A·z^(-α)·d^(-γ)·(1+B·ln(z)/g)", f08, [2.3, 0.9, 0.6, 0.5]))

# 9. p_c = A·(z·d)^(-α)·(1+B/g)
def f09(p, z, d, g):
    return p[0] * np.power(z*d, -p[1]) * (1 + p[2]/g)
candidates.append(("A·(z·d)^(-α)·(1+B/g)", f09, [10.0, 0.7, 1.0]))

# 10. 谱理论启发: p_c = 1/(z-1) + a/(z^b·d^c)
def f10(p, z, d, g):
    return 1.0/(z-1) + p[0] / (np.power(z, p[1]) * np.power(d, p[2]))
candidates.append(("1/(z-1) + a/(z^b·d^c)", f10, [5.0, 1.5, 1.0]))

# 11. 谱理论启发: p_c = 1/(z-1) + a/(z^b·g^c)
def f11(p, z, d, g):
    return 1.0/(z-1) + p[0] / (np.power(z, p[1]) * np.power(g, p[2]))
candidates.append(("1/(z-1) + a/(z^b·g^c)", f11, [5.0, 1.5, 0.5]))

# 12. 组合: p_c = 1/(z-1) + a/(z^b·d^c) + d/(z^e·g^f)
def f12(p, z, d, g):
    return 1.0/(z-1) + p[0]/(np.power(z, p[1])*np.power(d, p[2])) + p[3]/(np.power(z, p[4])*np.power(g, p[5]))
candidates.append(("1/(z-1) + a/(z^b·d^c) + d/(z^e·g^f)", f12, [5.0, 1.5, 1.0, 1.0, 1.5, 0.5]))

# 13. 高维极限统一: p_c = 1/(z-1) · (1 + a/(z·(d+δ)^b))
def f13(p, z, d, g):
    a, b, delta = p
    d_eff = np.maximum(d + delta, 0.01)
    R = 1 - 1.0/z + a / (z * np.power(d_eff, b))
    return R / (z - 1)
candidates.append(("R/(z-1), R=1-1/z+a/(z·(d+δ)^b)", f13, [7.0, 0.42, 0.5]))

# 14. Nachmias-Peres 修正: p_c = 1/z + a/z^(4/3) + b/(z·d^c)
def f14(p, z, d, g):
    return 1.0/z + p[0]/np.power(z, 4.0/3) + p[1]/(z*np.power(d, p[2]))
candidates.append(("1/z + a/z^(4/3) + b/(z·d^c)", f14, [1.0, 5.0, 0.4]))

# 15. 对偶性启发: p_c · p_c_dual = const
# honeycomb-triangular 对偶: site p_c(honey) ≈ 1 - bond p_c(tri)
# 这个太特殊，不适合通用公式

# 16. 综合公式: p_c = A/(z-1)^α · d^(-γ) · (1+B/g)^δ
def f16(p, z, d, g):
    return p[0] * np.power(z-1, -p[1]) * np.power(d, -p[2]) * np.power(1+p[3]/g, p[4])
candidates.append(("A·(z-1)^(-α)·d^(-γ)·(1+B/g)^δ", f16, [2.0, 0.8, 0.7, 1.0, 1.0]))

# 运行所有候选
print(f"\n{'公式':<45} {'R²':<10} {'MAE':<10} {'参数'}")
print("-"*100)

results_all = []
for name, func, init in candidates:
    params, R2, MAE, pred = try_formula(name, func, init)
    if params is not None:
        param_str = ", ".join(f"{p:.3f}" for p in params)
        results_all.append((name, params, R2, MAE, pred, func))
        print(f"{name:<45} {R2:<10.6f} {MAE:<10.6f} {param_str}")

# 排序
results_all.sort(key=lambda x: x[2], reverse=True)

print(f"\n{'='*80}")
print("排名（按 R²）:")
print(f"{'='*80}")
for i, (name, params, R2, MAE, pred, func) in enumerate(results_all[:10]):
    k = len(params)
    AIC = n * np.log((1-R2)*np.sum((pc-np.mean(pc))**2)/n) + 2*k
    print(f"  {i+1}. {name:<45} R²={R2:.6f} MAE={MAE:.6f} AIC={AIC:.1f} k={k}")

# ============ 4. 最优公式详细分析 ============

print(f"\n{'='*80}")
print("【4. 最优公式详细分析】")
print(f"{'='*80}")

best_name, best_params, best_R2, best_MAE, best_pred, best_func = results_all[0]

print(f"\n最优: {best_name}")
print(f"参数: {best_params}")
print(f"R² = {best_R2:.6f}, MAE = {best_MAE:.6f}")

# 逐点
print(f"\n逐点:")
print(f"{'晶格':<15} {'p_c实际':<10} {'p_c预测':<10} {'误差%':<8}")
print("-"*48)
for i, lat in enumerate(lattices):
    err = abs(pc[i] - best_pred[i]) / pc[i] * 100
    flag = " ⚠️" if err > 15 else ""
    print(f"{lat['name']:<15} {pc[i]:<10.5f} {best_pred[i]:<10.5f} {err:<8.1f}%{flag}")

# LOOCV
print(f"\nLOOCV:")
loocv_errs = []
for i in range(n):
    z_train = np.delete(z, i)
    d_train = np.delete(d, i)
    g_train = np.delete(g, i)
    pc_train = np.delete(pc, i)

    def obj_loo(params):
        pred = best_func(params, z_train, d_train, g_train)
        if np.any(np.isnan(pred)): return 1e10
        return np.sum((pred - pc_train)**2)

    res_loo = minimize(obj_loo, best_params, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    pred_i = best_func(res_loo.x, z[i:i+1], d[i:i+1], g[i:i+1])
    if np.isnan(pred_i):
        loocv_errs.append(1.0)
    else:
        loocv_errs.append(abs(pc[i] - pred_i[0]))

loocv_mae = np.mean(loocv_errs)
print(f"  LOOCV MAE = {loocv_mae:.6f}")
print(f"  全拟合 MAE = {best_MAE:.6f}")
print(f"  比率 = {loocv_mae/best_MAE:.2f}")

# ============ 5. 精确解检验 ============

print(f"\n{'='*80}")
print("【5. 精确解检验】")
print(f"{'='*80}")

pred_tri = best_func(best_params, np.array([6]), np.array([2]), np.array([3]))[0]
pred_hon = best_func(best_params, np.array([3]), np.array([2]), np.array([6]))[0]

print(f"\ntriangular (精确 p_c = 0.5000):")
print(f"  预测 = {pred_tri:.6f}, 误差 = {abs(pred_tri-0.5):.6f}")

print(f"\nhoneycomb (精确 p_c = 0.6970):")
print(f"  预测 = {pred_hon:.6f}, 误差 = {abs(pred_hon-0.697):.6f}")

# ============ 6. 综合结论 ============

print(f"\n{'='*80}")
print("【6. 综合结论】")
print(f"{'='*80}")

# 找到 AIC 最优的
best_aic = None
best_aic_val = np.inf
for name, params, R2, MAE, pred, func in results_all:
    k = len(params)
    AIC = n * np.log((1-R2)*np.sum((pc-np.mean(pc))**2)/n) + 2*k
    if AIC < best_aic_val:
        best_aic_val = AIC
        best_aic = (name, params, R2, MAE, pred, func, AIC)

print(f"\nAIC 最优: {best_aic[0]}")
print(f"  AIC = {best_aic[6]:.1f}")
print(f"  R² = {best_aic[2]:.6f}")
print(f"  MAE = {best_aic[3]:.6f}")
print(f"  参数: {best_aic[1]}")
