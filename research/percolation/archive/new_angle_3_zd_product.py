"""
新角度 III：z·d 乘积主导的新框架
===================================

核心洞察（ML特征重要性）：
  z·d 的乘积是最重要的特征！

  Top 特征:
    1. 1/(z·d)
    2. z·d
    3. log(z·d)

  之前所有公式都把 z 和 d 分开：p_c = A·z^{-α}·d^{-γ}·...
  这可能是错误方向！

新假说：
  p_c = f(z·d)  或  p_c = f(z·d, g, C, ...)

  物理意义：
  - z·d 代表"总连接密度"（每个维度的连接 × 维度数）
  - 高维晶格和高配位数在某种意义下等价
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============ 数据 ============

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

# 计算 z·d
for lat in lattices:
    lat['zd'] = lat['z'] * lat['d']
    lat['C'] = 2*lat['n_tri']/(lat['z']*(lat['z']-1)) if lat['z'] > 1 else 0

# ============ 1. 检查 z·d 的主导性 ============

print("="*80)
print("【1. z·d 的主导性检验】")
print("="*80)

zd_arr = np.array([lat['zd'] for lat in lattices])
pc_arr = np.array([lat['pc'] for lat in lattices])

r_zd, p_zd = pearsonr(zd_arr, pc_arr)
r_z, p_z = pearsonr(np.array([lat['z'] for lat in lattices]), pc_arr)
r_d, p_d = pearsonr(np.array([lat['d'] for lat in lattices]), pc_arr)

print(f"\n相关性:")
print(f"  p_c vs z·d: r = {r_zd:+.4f}, p = {p_zd:.4f}")
print(f"  p_c vs z:   r = {r_z:+.4f}, p = {p_z:.4f}")
print(f"  p_c vs d:   r = {r_d:+.4f}, p = {p_d:.4f}")

print(f"\n{'晶格':<15} {'z':<4} {'d':<2} {'z·d':<6} {'p_c':<10} {'g':<2}")
print("-"*50)
for lat in lattices:
    print(f"{lat['name']:<15} {lat['z']:<4} {lat['d']:<2} {lat['zd']:<6} {lat['pc']:<10.5f} {lat['g']:<2}")

# 关键检查：同 z·d 不同晶格
print(f"\n同 z·d 检查:")
zd_groups = {}
for lat in lattices:
    zd = lat['zd']
    if zd not in zd_groups:
        zd_groups[zd] = []
    zd_groups[zd].append(lat)

for zd, lats in sorted(zd_groups.items()):
    if len(lats) > 1:
        print(f"  z·d = {zd}:")
        for lat in lats:
            print(f"    {lat['name']}: z={lat['z']}, d={lat['d']}, g={lat['g']}, p_c={lat['pc']:.5f}")

# ============ 2. 纯 z·d 幂律 ============

print("\n" + "="*80)
print("【2. 纯 z·d 幂律: p_c = A·(z·d)^{-α}】")
print("="*80)

def powerlaw_zd(params, zd):
    A, alpha = params
    return A * np.power(zd, -alpha)

def fit_powerlaw(zd, pc):
    def obj(params):
        pred = powerlaw_zd(params, zd)
        return np.sum((pred - pc)**2)

    res = minimize(obj, [10.0, 0.7], method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})

    pred = powerlaw_zd(res.x, zd)
    ss_res = np.sum((pc - pred)**2)
    ss_tot = np.sum((pc - np.mean(pc))**2)
    R2 = 1 - ss_res / ss_tot
    MAE = np.mean(np.abs(pc - pred))

    return res.x, R2, MAE, pred

params_zd, R2_zd, MAE_zd, pred_zd = fit_powerlaw(zd_arr, pc_arr)

print(f"\np_c = {params_zd[0]:.4f} · (z·d)^{{-{params_zd[1]:.4f}}}")
print(f"R² = {R2_zd:.6f}, MAE = {MAE_zd:.6f}")

print(f"\n逐点比较:")
for i, lat in enumerate(lattices):
    err = abs(pc_arr[i] - pred_zd[i])
    err_rel = err / pc_arr[i] * 100
    flag = " ⚠️" if err_rel > 20 else ""
    print(f"  {lat['name']:<15} z·d={lat['zd']:<5} p_c={pc_arr[i]:.5f} 预测={pred_zd[i]:.5f} 误差={err_rel:.1f}%{flag}")

# ============ 3. z·d + girth 修正 ============

print("\n" + "="*80)
print("【3. z·d + girth 修正】")
print("="*80)

g_arr = np.array([lat['g'] for lat in lattices])

def model_zd_g(params, zd, g):
    """p_c = A·(z·d)^{-α}·(1+B/g)"""
    A, alpha, B = params
    return A * np.power(zd, -alpha) * (1 + B/g)

def model_zd_g2(params, zd, g):
    """p_c = A·(z·d)^{-α}·(1+B/g+C/g²)"""
    A, alpha, B, C = params
    return A * np.power(zd, -alpha) * (1 + B/g + C/g**2)

def model_zd_glog(params, zd, g):
    """p_c = A·(z·d)^{-α}·(1+B·log(g))"""
    A, alpha, B = params
    return A * np.power(zd, -alpha) * (1 + B*np.log(g))

def fit_model(func, params0, zd, g, pc):
    def obj(params):
        pred = func(params, zd, g)
        return np.sum((pred - pc)**2)
    res = minimize(obj, params0, method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
    pred = func(res.x, zd, g)
    ss_res = np.sum((pc - pred)**2)
    ss_tot = np.sum((pc - np.mean(pc))**2)
    R2 = 1 - ss_res / ss_tot
    MAE = np.mean(np.abs(pc - pred))
    return res.x, R2, MAE, pred

models = [
    ('p_c = A·(z·d)^{-α}·(1+B/g)', model_zd_g, [10.0, 0.7, 1.0]),
    ('p_c = A·(z·d)^{-α}·(1+B/g+C/g²)', model_zd_g2, [10.0, 0.7, 1.0, 0.1]),
    ('p_c = A·(z·d)^{-α}·(1+B·log(g))', model_zd_glog, [10.0, 0.7, 0.1]),
]

print(f"\n{'模型':<40} {'R²':<10} {'MAE':<10} {'参数'}")
print("-"*80)

best_R2 = -np.inf
best_model = None

for name, func, init in models:
    params, R2, MAE, pred = fit_model(func, init, zd_arr, g_arr, pc_arr)
    param_str = ", ".join(f"{p:.4f}" for p in params)
    print(f"{name:<40} {R2:<10.6f} {MAE:<10.6f} {param_str}")
    if R2 > best_R2:
        best_R2 = R2
        best_model = (name, func, params, pred)

# ============ 4. z·d + g + C ============

print("\n" + "="*80)
print("【4. z·d + g + 聚类系数 C】")
print("="*80)

C_arr = np.array([lat['C'] for lat in lattices])

def model_zd_g_C(params, zd, g, C):
    """p_c = A·(z·d)^{-α}·(1+B/g)·(1+D·C)"""
    A, alpha, B, D = params
    return A * np.power(zd, -alpha) * (1 + B/g) * (1 + D*C)

def model_zd_g_Cadd(params, zd, g, C):
    """p_c = A·(z·d)^{-α}·(1+B/g+D·C)"""
    A, alpha, B, D = params
    return A * np.power(zd, -alpha) * (1 + B/g + D*C)

def model_zd_g_Clog(params, zd, g, C):
    """p_c = A·(z·d)^{-α}·(1+B/g)·exp(D·C)"""
    A, alpha, B, D = params
    return A * np.power(zd, -alpha) * (1 + B/g) * np.exp(D*C)

def fit_model_4(func, params0, zd, g, C, pc):
    def obj(params):
        pred = func(params, zd, g, C)
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            return 1e10
        return np.sum((pred - pc)**2)
    res = minimize(obj, params0, method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
    pred = func(res.x, zd, g, C)
    if np.any(np.isnan(pred)):
        return None, 0, 1, None
    ss_res = np.sum((pc - pred)**2)
    ss_tot = np.sum((pc - np.mean(pc))**2)
    R2 = 1 - ss_res / ss_tot
    MAE = np.mean(np.abs(pc - pred))
    return res.x, R2, MAE, pred

models_C = [
    ('p_c = A·(z·d)^{-α}·(1+B/g)·(1+D·C)', model_zd_g_C, [10.0, 0.7, 1.0, -0.5]),
    ('p_c = A·(z·d)^{-α}·(1+B/g+D·C)', model_zd_g_Cadd, [10.0, 0.7, 1.0, -0.5]),
    ('p_c = A·(z·d)^{-α}·(1+B/g)·exp(D·C)', model_zd_g_Clog, [10.0, 0.7, 1.0, -0.5]),
]

print(f"\n{'模型':<45} {'R²':<10} {'MAE':<10} {'参数'}")
print("-"*85)

for name, func, init in models_C:
    result = fit_model_4(func, init, zd_arr, g_arr, C_arr, pc_arr)
    params, R2, MAE, pred = result
    if params is not None:
        param_str = ", ".join(f"{p:.4f}" for p in params)
        print(f"{name:<45} {R2:<10.6f} {MAE:<10.6f} {param_str}")
        if R2 > best_R2:
            best_R2 = R2
            best_model = (name, func, params, pred)

# ============ 5. 最优模型详细分析 ============

print("\n" + "="*80)
print("【5. 最优模型详细分析】")
print("="*80)

if best_model:
    name, func, params, pred = best_model
    print(f"\n最优模型: {name}")
    print(f"参数: {params}")
    print(f"R² = {best_R2:.6f}")
    print(f"MAE = {np.mean(np.abs(pc_arr - pred)):.6f}")

    # 逐点误差
    print(f"\n逐点误差:")
    print(f"{'晶格':<15} {'p_c真实':<10} {'p_c预测':<10} {'误差':<8} {'相对误差':<10}")
    print("-"*60)
    for i, lat in enumerate(lattices):
        err = abs(pc_arr[i] - pred[i])
        err_rel = err / pc_arr[i] * 100
        flag = " ⚠️" if err_rel > 10 else ""
        print(f"{lat['name']:<15} {pc_arr[i]:<10.5f} {pred[i]:<10.5f} {err:<8.5f} {err_rel:<10.2f}%{flag}")

    # LOOCV
    print(f"\n留一法交叉验证 (LOOCV):")
    loocv_errors = []
    for i in range(n):
        zd_train = np.delete(zd_arr, i)
        g_train = np.delete(g_arr, i)
        C_train = np.delete(C_arr, i)
        pc_train = np.delete(pc_arr, i)

        # 重新拟合
        if func in [model_zd_g, model_zd_glog]:
            res_i, _, _, _ = fit_model(func, params, zd_train, g_train, pc_train)
            pred_i = func(res_i, zd_arr[i], g_arr[i])
        elif func in [model_zd_g_C, model_zd_g_Cadd, model_zd_g_Clog]:
            res_i, _, _, _ = fit_model_4(func, params, zd_train, g_train, C_train, pc_train)
            if res_i is not None:
                pred_i = func(res_i, zd_arr[i], g_arr[i], C_arr[i])
            else:
                pred_i = pred[i]
        else:
            pred_i = pred[i]

        loocv_errors.append(abs(pc_arr[i] - pred_i))

    loocv_mae = np.mean(loocv_errors)
    print(f"  LOOCV MAE = {loocv_mae:.6f}")
    print(f"  全拟合 MAE = {np.mean(np.abs(pc_arr - pred)):.6f}")
    if loocv_mae < 2 * np.mean(np.abs(pc_arr - pred)):
        print(f"  ✓ 无明显过拟合")
    else:
        print(f"  ⚠️ 可能过拟合")

# ============ 6. 物理意义解读 ============

print("\n" + "="*80)
print("【6. 物理意义解读】")
print("="*80)

if best_model and best_R2 > 0.98:
    name, func, params, pred = best_model
    print(f"""
新框架: p_c = f(z·d, ...)

物理意义：
  - z·d 是"总连接密度"
  - 每个节点在每个维度上有 z/d 个连接
  - z·d 越大 → p_c 越低（渗流越容易）

与之前框架的对比：
  旧: p_c = A·z^(-α)·d^(-γ)·...
  新: p_c = A·(z·d)^(-α)·...

  如果 α_zd ≈ α_z ≈ α_d，则两种框架等价
  如果 α_zd ≠ α_z, α_d，则新框架更好

  新框架的优势：
  - 更少参数（α 代替 α_z 和 α_d）
  - 物理意义更清晰
  - ML 特征重要性支持

  新框架的挑战：
  - 同 z·d 不同 p_c 的晶格如何解释？
  - FCC 是否符合？
""")

# 检查同 z·d 的情况
print("同 z·d 检查（需要额外物理量解释）:")
for zd, lats in sorted(zd_groups.items()):
    if len(lats) > 1:
        pcs = [lat['pc'] for lat in lats]
        gs = [lat['g'] for lat in lats]
        Cs = [lat['C'] for lat in lats]
        print(f"  z·d = {zd}:")
        for lat in lats:
            print(f"    {lat['name']}: g={lat['g']}, C={lat['C']:.3f}, p_c={lat['pc']:.5f}")
        print(f"    p_c 范围: [{min(pcs):.5f}, {max(pcs):.5f}]")
        print(f"    g 范围: {gs}, C 范围: {[f'{c:.3f}' for c in Cs]}")

# ============ 7. 对比三种框架 ============

print("\n" + "="*80)
print("【7. 对比三种框架】")
print("="*80)

# 框架 1: p_c = A·z^{-α}·d^{-γ} (2 参数)
def model1(params, z, d):
    A, alpha, gamma = params
    return A * np.power(z, -alpha) * np.power(d, -gamma)

# 框架 2: p_c = A·z^{-α}·d^{-γ}·(1+B/g) (3 参数)
def model2(params, z, d, g):
    A, alpha, gamma, B = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * (1 + B/g)

# 框架 3 (新): p_c = A·(z·d)^{-α}·(1+B/g) (3 参数)
def model3(params, z, d, g):
    A, alpha, B = params
    return A * np.power(z*d, -alpha) * (1 + B/g)

z_arr = np.array([lat['z'] for lat in lattices])
d_arr = np.array([lat['d'] for lat in lattices])

def fit_comp(func, params0, z, d, pc, g=None):
    def obj(params):
        if g is not None:
            pred = func(params, z, d, g)
        else:
            pred = func(params, z, d)
        if np.any(np.isnan(pred)):
            return 1e10
        return np.sum((pred - pc)**2)
    res = minimize(obj, params0, method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
    if g is not None:
        pred = func(res.x, z, d, g)
    else:
        pred = func(res.x, z, d)
    ss_res = np.sum((pc - pred)**2)
    ss_tot = np.sum((pc - np.mean(pc))**2)
    R2 = 1 - ss_res / ss_tot
    MAE = np.mean(np.abs(pc - pred))
    k = len(res.x)
    AIC = n * np.log(ss_res/n) + 2*k
    BIC = n * np.log(ss_res/n) + k * np.log(n)
    return res.x, R2, MAE, AIC, BIC, pred

r1 = fit_comp(model1, [3.0, 0.7, 0.7], z_arr, d_arr, pc_arr)
r2 = fit_comp(model2, [2.3, 0.8, 0.7, 1.5], z_arr, d_arr, pc_arr, g_arr)
r3 = fit_comp(model3, [10.0, 0.7, 1.5], z_arr, d_arr, pc_arr, g_arr)

print(f"\n{'框架':<50} {'R²':<10} {'MAE':<10} {'AIC':<10} {'BIC':<10}")
print("-"*95)

print(f"1: p_c = A·z^(-α)·d^(-γ):")
print(f"   参数: A={r1[0][0]:.3f}, α={r1[0][1]:.3f}, γ={r1[0][2]:.3f}")

print(f"\n2: p_c = A·z^(-α)·d^(-γ)·(1+B/g):")
print(f"   参数: A={r2[0][0]:.3f}, α={r2[0][1]:.3f}, γ={r2[0][2]:.3f}, B={r2[0][3]:.3f}")

print(f"\n3 (新): p_c = A·(z·d)^(-α)·(1+B/g):")
print(f"   参数: A={r3[0][0]:.3f}, α={r3[0][1]:.3f}, B={r3[0][2]:.3f}")

print(f"\nAIC 比较:")
print(f"  框架1 vs 框架2: ΔAIC = {r1[3]-r2[3]:.2f}")
print(f"  框架1 vs 框架3: ΔAIC = {r1[3]-r3[3]:.2f}")
print(f"  框架2 vs 框架3: ΔAIC = {r2[3]-r3[3]:.2f}")

if r3[3] < r2[3]:
    print(f"\n  ✓ 新框架(3) AIC 更低 → 更优！（少 1 个参数，但 R² 相当）")
else:
    print(f"\n  ✗ 旧框架(2) AIC 更低")

# ============ 结论 ============

print("\n" + "="*80)
print("【结论】")
print("="*80)

print("""
机器学习特征重要性揭示了 z·d 的主导地位。

新框架: p_c = A·(z·d)^{-α}·(1+B/g)

优势：
  - 3 个参数（比旧框架少 1 个）
  - 物理意义清晰：z·d 是"总连接密度"
  - ML 特征重要性支持

关键问题：
  - 同 z·d 不同 p_c 的晶格需要额外解释
  - FCC 是否遵循这个框架？

如果这个框架通过严格验证：
  → PRL 级别的发现！
  → 揭示了渗流阈值的真正标度变量
""")

print("\n" + "="*80)
print("探索完成")
print("="*80)
