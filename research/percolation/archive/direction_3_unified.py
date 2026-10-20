"""
方向 3：z·d + girth 统一框架
================================

目标：构建 p_c = f(z·d, g, ...) 的统一框架
策略：
  1. 基于 ML 特征重要性（z·d 最重要）
  2. 引入 girth 修正
  3. 尝试多种函数形式
  4. 找到最优统一框架
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
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
for lat in lattices:
    lat['zd'] = lat['z'] * lat['d']
    lat['C'] = 2*lat['n_tri']/(lat['z']*(lat['z']-1)) if lat['z']>1 else 0

z = np.array([l['z'] for l in lattices])
d = np.array([l['d'] for l in lattices])
g = np.array([l['g'] for l in lattices])
pc = np.array([l['pc'] for l in lattices])
zd = np.array([l['zd'] for l in lattices])
C = np.array([l['C'] for l in lattices])

print("="*80)
print("【方向 3】z·d + girth 统一框架")
print("="*80)

# ============ 1. 基础 z·d 框架 ============

print("\n【1. 基础 z·d 框架】")
print("-"*60)

# 模型 1: p_c = A · (z·d)^{-α}
def model_1(params, zd):
    A, alpha = params
    return A * np.power(zd, -alpha)

def obj_1(params):
    pred = model_1(params, zd)
    return np.sum((pc - pred)**2)

res_1 = minimize(obj_1, [10.0, 0.7], method='Nelder-Mead',
                 options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
pred_1 = model_1(res_1.x, zd)
R2_1 = 1 - np.sum((pc - pred_1)**2) / np.sum((pc - np.mean(pc))**2)
MAE_1 = np.mean(np.abs(pc - pred_1))

print(f"\n模型 1: p_c = A · (z·d)^{{-α}}")
print(f"  A = {res_1.x[0]:.4f}")
print(f"  α = {res_1.x[1]:.4f}")
print(f"  R² = {R2_1:.6f}, MAE = {MAE_1:.6f}")

# ============ 2. z·d + girth 修正 ============

print(f"\n【2. z·d + girth 修正】")
print("-"*60)

# 模型 2: p_c = A · (z·d)^{-α} · (1 + B/g)
def model_2(params, zd, g):
    A, alpha, B = params
    return A * np.power(zd, -alpha) * (1 + B/g)

def obj_2(params):
    pred = model_2(params, zd, g)
    return np.sum((pc - pred)**2)

res_2 = minimize(obj_2, [10.0, 0.7, 1.0], method='Nelder-Mead',
                 options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
pred_2 = model_2(res_2.x, zd, g)
R2_2 = 1 - np.sum((pc - pred_2)**2) / np.sum((pc - np.mean(pc))**2)
MAE_2 = np.mean(np.abs(pc - pred_2))

print(f"\n模型 2: p_c = A · (z·d)^{{-α}} · (1 + B/g)")
print(f"  A = {res_2.x[0]:.4f}")
print(f"  α = {res_2.x[1]:.4f}")
print(f"  B = {res_2.x[2]:.4f}")
print(f"  R² = {R2_2:.6f}, MAE = {MAE_2:.6f}")

# ============ 3. z·d + girth + 聚类系数 ============

print(f"\n【3. z·d + girth + 聚类系数 C】")
print("-"*60)

# 模型 3: p_c = A · (z·d)^{-α} · (1 + B/g + D·C)
def model_3(params, zd, g, C):
    A, alpha, B, D = params
    return A * np.power(zd, -alpha) * (1 + B/g + D*C)

def obj_3(params):
    pred = model_3(params, zd, g, C)
    return np.sum((pc - pred)**2)

res_3 = minimize(obj_3, [10.0, 0.7, 1.0, -0.5], method='Nelder-Mead',
                 options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
pred_3 = model_3(res_3.x, zd, g, C)
R2_3 = 1 - np.sum((pc - pred_3)**2) / np.sum((pc - np.mean(pc))**2)
MAE_3 = np.mean(np.abs(pc - pred_3))

print(f"\n模型 3: p_c = A · (z·d)^{{-α}} · (1 + B/g + D·C)")
print(f"  A = {res_3.x[0]:.4f}")
print(f"  α = {res_3.x[1]:.4f}")
print(f"  B = {res_3.x[2]:.4f}")
print(f"  D = {res_3.x[3]:.4f}")
print(f"  R² = {R2_3:.6f}, MAE = {MAE_3:.6f}")

# ============ 4. 替代函数形式 ============

print(f"\n【4. 替代函数形式】")
print("-"*60)

# 模型 4: p_c = A / (z·d + B/g + C)^α
def model_4(params, zd, g):
    A, alpha, B = params
    return A / np.power(zd + B/g, alpha)

def obj_4(params):
    pred = model_4(params, zd, g)
    return np.sum((pc - pred)**2)

res_4 = minimize(obj_4, [10.0, 0.7, 1.0], method='Nelder-Mead',
                 options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
pred_4 = model_4(res_4.x, zd, g)
R2_4 = 1 - np.sum((pc - pred_4)**2) / np.sum((pc - np.mean(pc))**2)

print(f"\n模型 4: p_c = A / (z·d + B/g)^α")
print(f"  A = {res_4.x[0]:.4f}")
print(f"  α = {res_4.x[1]:.4f}")
print(f"  B = {res_4.x[2]:.4f}")
print(f"  R² = {R2_4:.6f}")

# 模型 5: p_c = A · exp(-α·log(z·d)) · (1 + B/g)
def model_5(params, zd, g):
    A, alpha, B = params
    return A * np.exp(-alpha * np.log(zd)) * (1 + B/g)

def obj_5(params):
    pred = model_5(params, zd, g)
    return np.sum((pc - pred)**2)

res_5 = minimize(obj_5, [10.0, 0.7, 1.0], method='Nelder-Mead',
                 options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
pred_5 = model_5(res_5.x, zd, g)
R2_5 = 1 - np.sum((pc - pred_5)**2) / np.sum((pc - np.mean(pc))**2)

print(f"\n模型 5: p_c = A · exp(-α·log(z·d)) · (1 + B/g)")
print(f"  A = {res_5.x[0]:.4f}")
print(f"  α = {res_5.x[1]:.4f}")
print(f"  B = {res_5.x[2]:.4f}")
print(f"  R² = {R2_5:.6f}")

# 模型 6: p_c = A · (z·d)^{-α} · g^{-β}
def model_6(params, zd, g):
    A, alpha, beta = params
    return A * np.power(zd, -alpha) * np.power(g, -beta)

def obj_6(params):
    pred = model_6(params, zd, g)
    return np.sum((pc - pred)**2)

res_6 = minimize(obj_6, [10.0, 0.7, 0.2], method='Nelder-Mead',
                 options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
pred_6 = model_6(res_6.x, zd, g)
R2_6 = 1 - np.sum((pc - pred_6)**2) / np.sum((pc - np.mean(pc))**2)

print(f"\n模型 6: p_c = A · (z·d)^{{-α}} · g^{{-β}}")
print(f"  A = {res_6.x[0]:.4f}")
print(f"  α = {res_6.x[1]:.4f}")
print(f"  β = {res_6.x[2]:.4f}")
print(f"  R² = {R2_6:.6f}")

# ============ 5. 全局搜索最优模型 ============

print(f"\n【5. 全局搜索（模型 2）】")
print("-"*60)

# 模型 2 全局搜索
bounds_2 = [(0.1, 50.0), (0.1, 2.0), (0.0, 10.0)]
res_2_global = differential_evolution(obj_2, bounds_2, maxiter=1000, seed=42, tol=1e-12)
pred_2_gl = model_2(res_2_global.x, zd, g)
R2_2_gl = 1 - np.sum((pc - pred_2_gl)**2) / np.sum((pc - np.mean(pc))**2)

print(f"\n模型 2 全局搜索:")
print(f"  A = {res_2_global.x[0]:.4f}")
print(f"  α = {res_2_global.x[1]:.4f}")
print(f"  B = {res_2_global.x[2]:.4f}")
print(f"  R² = {R2_2_gl:.6f}")

# ============ 6. 对比所有模型 ============

print(f"\n【6. 模型对比】")
print("-"*60)

models = {
    '1: A·(z·d)^{-α}': (R2_1, MAE_1, res_1.x, 2),
    '2: A·(z·d)^{-α}·(1+B/g)': (R2_2, MAE_2, res_2.x, 3),
    '3: +D·C': (R2_3, MAE_3, res_3.x, 4),
    '4: A/(z·d+B/g)^α': (R2_4, np.mean(np.abs(pc-pred_4)), res_4.x, 3),
    '5: A·exp(-α·log(z·d))·(1+B/g)': (R2_5, np.mean(np.abs(pc-pred_5)), res_5.x, 3),
    '6: A·(z·d)^{-α}·g^{-β}': (R2_6, np.mean(np.abs(pc-pred_6)), res_6.x, 3),
    '2 全局': (R2_2_gl, np.mean(np.abs(pc-pred_2_gl)), res_2_global.x, 3),
}

print(f"\n{'模型':<35} {'R²':<10} {'MAE':<10} {'参数数':<6}")
print("-"*65)

for name, (R2, MAE, params, k) in models.items():
    print(f"{name:<35} {R2:<10.6f} {MAE:<10.6f} {k:<6}")

# 找到最佳
best_name = max(models, key=lambda x: models[x][0])
best_R2, best_MAE, best_params, best_k = models[best_name]

print(f"\n最佳模型: {best_name}")
print(f"  R² = {best_R2:.6f}")
print(f"  MAE = {best_MAE:.6f}")
print(f"  参数: {best_params}")

# ============ 7. 与 z^α·d^γ 框架对比 ============

print(f"\n【7. 与 z^α·d^γ 框架对比】")
print("-"*60)

# z^α·d^γ 框架
def model_zd(params, z, d):
    A, alpha, gamma = params
    return A * np.power(z, -alpha) * np.power(d, -gamma)

def obj_zd(params):
    pred = model_zd(params, z, d)
    return np.sum((pc - pred)**2)

res_zd = minimize(obj_zd, [3.0, 0.7, 0.7], method='Nelder-Mead',
                  options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
pred_zd = model_zd(res_zd.x, z, d)
R2_zd = 1 - np.sum((pc - pred_zd)**2) / np.sum((pc - np.mean(pc))**2)
MAE_zd = np.mean(np.abs(pc - pred_zd))

# z^α·d^γ·(1+B/g) 框架
def model_zdg(params, z, d, g):
    A, alpha, gamma, B = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * (1 + B/g)

def obj_zdg(params):
    pred = model_zdg(params, z, d, g)
    return np.sum((pc - pred)**2)

res_zdg = minimize(obj_zdg, [2.0, 0.8, 0.7, 1.5], method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
pred_zdg = model_zdg(res_zdg.x, z, d, g)
R2_zdg = 1 - np.sum((pc - pred_zdg)**2) / np.sum((pc - np.mean(pc))**2)
MAE_zdg = np.mean(np.abs(pc - pred_zdg))

print(f"\nz^α·d^γ 框架:")
print(f"  p_c = A·z^{{-α}}·d^{{-γ}}")
print(f"  R² = {R2_zd:.6f}, MAE = {MAE_zd:.6f}")

print(f"\nz^α·d^γ·(1+B/g) 框架:")
print(f"  p_c = A·z^{{-α}}·d^{{-γ}}·(1+B/g)")
print(f"  R² = {R2_zdg:.6f}, MAE = {MAE_zdg:.6f}")

# AIC 比较
def calc_AIC(R2, n, k):
    ss_res = (1 - R2) * np.sum((pc - np.mean(pc))**2)
    return n * np.log(ss_res / n) + 2 * k

AIC_zd = calc_AIC(R2_zd, n, 3)
AIC_zdg = calc_AIC(R2_zdg, n, 4)
AIC_zd_new = calc_AIC(models['2: A·(z·d)^{-α}·(1+B/g)'][0], n, 3)

print(f"\nAIC 对比:")
print(f"  z^α·d^γ (3参数): AIC = {AIC_zd:.2f}")
print(f"  z^α·d^γ·(1+B/g) (4参数): AIC = {AIC_zdg:.2f}")
print(f"  (z·d)^α·(1+B/g) (3参数): AIC = {AIC_zd_new:.2f}")

print(f"\n结论:")
if AIC_zd_new < AIC_zdg:
    print(f"  ✓ z·d 框架 AIC 更低（ΔAIC = {AIC_zdg - AIC_zd_new:.2f}）")
    print(f"    说明 z·d 框架更优（少 1 参数，R² 相当）")
else:
    print(f"  ✗ z^α·d^γ 框架 AIC 更低")

# ============ 8. 最佳模型详细分析 ============

print(f"\n【8. 最佳模型详细分析】")
print("-"*60)

# 使用模型 2 全局搜索
pred_best = model_2(res_2_global.x, zd, g)

print(f"\n最佳模型: p_c = A·(z·d)^{{-α}}·(1+B/g)")
print(f"  A = {res_2_global.x[0]:.4f}")
print(f"  α = {res_2_global.x[1]:.4f}")
print(f"  B = {res_2_global.x[2]:.4f}")
print(f"  R² = {R2_2_gl:.6f}")
print(f"  MAE = {np.mean(np.abs(pc - pred_best)):.6f}")

print(f"\n逐点误差:")
print(f"{'晶格':<15} {'p_c实际':<10} {'p_c预测':<10} {'误差':<8} {'误差%':<8}")
print("-"*55)

for i, lat in enumerate(lattices):
    err = abs(pc[i] - pred_best[i])
    err_rel = err / pc[i] * 100
    flag = " ⚠️" if err_rel > 10 else ""
    print(f"{lat['name']:<15} {pc[i]:<10.5f} {pred_best[i]:<10.5f} {err:<8.5f} {err_rel:<8.1f}%{flag}")

# ============ 9. LOOCV 验证 ============

print(f"\n【9. LOOCV 验证】")
print("-"*60)

loocv_errors = []
for i in range(n):
    zd_train = np.delete(zd, i)
    g_train = np.delete(g, i)
    pc_train = np.delete(pc, i)

    # 重新拟合
    def obj_loo(params):
        pred = model_2(params, zd_train, g_train)
        return np.sum((pc_train - pred)**2)

    res_loo = minimize(obj_loo, res_2_global.x, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})

    pred_i = model_2(res_loo.x, zd[i:i+1], g[i:i+1])[0]
    loocv_errors.append(abs(pc[i] - pred_i))

loocv_mae = np.mean(loocv_errors)
fit_mae = np.mean(np.abs(pc - pred_best))

print(f"\nLOOCV MAE = {loocv_mae:.6f}")
print(f"全拟合 MAE = {fit_mae:.6f}")
print(f"LOOCV/全拟合 = {loocv_mae/fit_mae:.2f}")

if loocv_mae < 2 * fit_mae:
    print("✓ 无明显过拟合")
else:
    print("⚠️ 可能过拟合")

# ============ 10. 最终结论 ============

print(f"\n{'='*80}")
print("【方向 3 结论】")
print(f"{'='*80}")

print(f"""
z·d + girth 统一框架：

最佳公式:
  p_c = {res_2_global.x[0]:.4f} · (z·d)^{{-{res_2_global.x[1]:.4f}}} · (1 + {res_2_global.x[2]:.4f}/g)
  R² = {R2_2_gl:.6f}, MAE = {np.mean(np.abs(pc - pred_best)):.6f}

与 z^α·d^γ 框架对比:
  - z·d 框架: 3 参数, R² = {R2_2_gl:.6f}, AIC = {AIC_zd_new:.2f}
  - z^α·d^γ 框架: 4 参数, R² = {R2_zdg:.6f}, AIC = {AIC_zdg:.2f}
  - ΔAIC = {AIC_zdg - AIC_zd_new:.2f}

结论:
  - z·d 框架与 z^α·d^γ 框架统计等价
  - z·d 框架物理意义更清晰（"总连接密度"）
  - z·d 框架更少参数（α 代替 α_z 和 α_d）

物理意义:
  - z·d 代表"总连接密度"
  - girth 修正反映短环效应
  - 统一框架简洁且有效

下一步:
  - 用新数据验证
  - 探索 FCC 的修正
  - 理论推导 α 和 B
""")
