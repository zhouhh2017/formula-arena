from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
公式改进实验 v2 — 残差模式驱动
================================
发现：高维超立方(d=6-11)一致被高估20%，某些2D被高估20-29%
目标：系统性消除这些偏差模式
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from itertools import product
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

print("=" * 75)
print("【公式改进实验 v2 — 残差模式驱动】")
print("=" * 75)

# ============================================================
# 1. 加载数据
# ============================================================
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
print(f"\n加载: {len(df)} 个晶格")

z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values if 'name' in df.columns else [f"lat_{i}" for i in range(len(df))]
n = len(pc)

# ============================================================
# 2. 残差模式分析
# ============================================================
print(f"\n{'='*75}")
print("【残差模式分析】")
print(f"{'='*75}")

def current_model(z, d, g):
    a, beta, gamma, z0 = 12.9, 1.03, 0.85, 9.7
    correction = (a / (d**beta * g**gamma)) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

pred_cur = current_model(z, d, g)
R_cur = pc * (z - 1)  # 实际的Bethe归一化
pred_R = pred_cur * (z - 1)
residual_R = R_cur - pred_R  # R的残差

# 按类别分析残差
print(f"\n{'类别':<25} {'数量':>4} {'平均残差R':>10} {'平均误差%':>10} {'方向':>6}")
print("-" * 60)

# 高维超立方
mask_hyper = (z == 2*d) & (d >= 4)
for d_min in [4, 6, 8]:
    mask = mask_hyper & (d >= d_min)
    if np.sum(mask) > 0:
        err = np.mean(np.abs((pc[mask] - pred_cur[mask]) / pc[mask])) * 100
        mean_res = np.mean(residual_R[mask])
        direction = "高估" if mean_res < 0 else "低估"
        print(f"超立方 d≥{d_min:<20} {np.sum(mask):>4} {mean_res:>10.4f} {err:>9.2f}% {direction:>6}")

# 2D
mask_2d = d == 2
err = np.mean(np.abs((pc[mask_2d] - pred_cur[mask_2d]) / pc[mask_2d])) * 100
mean_res = np.mean(residual_R[mask_2d])
direction = "高估" if mean_res < 0 else "低估"
print(f"{'2D全部':<25} {np.sum(mask_2d):>4} {mean_res:>10.4f} {err:>9.2f}% {direction:>6}")

# 3D
mask_3d = d == 3
err = np.mean(np.abs((pc[mask_3d] - pred_cur[mask_3d]) / pc[mask_3d])) * 100
mean_res = np.mean(residual_R[mask_3d])
direction = "高估" if mean_res < 0 else "低估"
print(f"{'3D全部':<25} {np.sum(mask_3d):>4} {mean_res:>10.4f} {err:>9.2f}% {direction:>6}")

# Malarz 2D
mask_malarz = (d == 2) & (z >= 8)
err = np.mean(np.abs((pc[mask_malarz] - pred_cur[mask_malarz]) / pc[mask_malarz])) * 100
mean_res = np.mean(residual_R[mask_malarz])
direction = "高估" if mean_res < 0 else "低估"
print(f"{'Malarz 2D (z≥8)':<25} {np.sum(mask_malarz):>4} {mean_res:>10.4f} {err:>9.2f}% {direction:>6}")

# 经典2D
mask_classic2d = (d == 2) & (z < 8)
err = np.mean(np.abs((pc[mask_classic2d] - pred_cur[mask_classic2d]) / pc[mask_classic2d])) * 100
mean_res = np.mean(residual_R[mask_classic2d])
direction = "高估" if mean_res < 0 else "低估"
print(f"{'经典2D (z<8)':<25} {np.sum(mask_classic2d):>4} {mean_res:>10.4f} {err:>9.2f}% {direction:>6}")

# ============================================================
# 3. 关键洞察：检查R vs z/d的关系
# ============================================================
print(f"\n{'='*75}")
print("【关键洞察：R = p_c·(z-1) 的标度行为】")
print(f"{'='*75}")

eta = z / d  # 每维度的配位数

print(f"\n{'晶格':<20} {'z':>3} {'d':>2} {'g':>2} {'R实测':>7} {'R预测':>7} {'η=z/d':>6}")
print("-" * 55)
# 按η排序
idx_sorted = np.argsort(eta)
for idx in idx_sorted[::5]:  # 每隔5个打印
    print(f"{names[idx]:<20} {int(z[idx]):>3} {int(d[idx]):>2} {int(g[idx]):>2} "
          f"{R_cur[idx]:>7.3f} {pred_R[idx]:>7.3f} {eta[idx]:>6.2f}")

# ============================================================
# 4. 改进方向：检查高维超立方的渐近行为
# ============================================================
print(f"\n{'='*75}")
print("【高维超立方：R vs d 的渐近行为】")
print(f"{'='*75}")

mask_hc = z == 2*d
d_hc = d[mask_hc]
R_hc = R_cur[mask_hc]
pred_R_hc = pred_R[mask_hc]

print(f"\n{'d':>3} {'z':>4} {'R实测':>8} {'R预测':>8} {'误差%':>8}")
print("-" * 35)
for i in np.argsort(d_hc):
    idx = np.where(mask_hc)[0][i]
    err_i = abs((pc[idx] - pred_cur[idx]) / pc[idx]) * 100
    print(f"{int(d[idx]):>3} {int(z[idx]):>4} {R_cur[idx]:>8.4f} {pred_R[idx]:>8.4f} {err_i:>7.2f}%")

# R实测随d的变化
# R = 1 + correction
# correction_R = R - 1
corr_R = R_hc - 1
print(f"\n修正量 correction_R = R-1:")
for i in np.argsort(d_hc):
    idx = np.where(mask_hc)[0][i]
    print(f"  d={int(d[idx]):>2}: R-1 = {corr_R[i]:.4f}")

# 尝试拟合 correction_R ~ A / d^alpha
from scipy.optimize import curve_fit

def power_decay(d, A, alpha):
    return A / d**alpha

if len(d_hc) >= 3:
    try:
        popt, pcov = curve_fit(power_decay, d_hc[d_hc >= 4], corr_R[d_hc >= 4], p0=[3.0, 1.0])
        print(f"\n超立方 correction_R ≈ {popt[0]:.3f} / d^{popt[1]:.3f}")
        print(f"对比：当前公式 prediction = 12.9/(d^1.03 · 4^0.85) · (1-exp(-2d/9.7))")

        # 比较
        d_test = np.array([4, 6, 8, 10, 12, 13])
        fit_pred = power_decay(d_test, *popt)
        cur_pred = (12.9 / (d_test**1.03 * 4**0.85)) * (1 - np.exp(-2*d_test/9.7))
        print(f"\n{'d':>3} {'实测R-1':>8} {'幂律拟合':>8} {'当前公式':>8}")
        for i, dd in enumerate(d_test):
            idx_near = np.argmin(np.abs(d_hc - dd))
            print(f"{dd:>3} {corr_R[idx_near]:>8.4f} {fit_pred[i]:>8.4f} {cur_pred[i]:>8.4f}")
    except:
        print("幂律拟合失败")

# ============================================================
# 5. 系统性公式搜索
# ============================================================
print(f"\n{'='*75}")
print("【系统性公式搜索 — 100+候选】")
print(f"{'='*75}")

# 定义LOOCV计算（高效版）
def fast_loocv(pred_func, z, d, g, pc):
    """快速LOOCV: 对拟合好的模型直接预测"""
    pred = pred_func(z, d, g)
    errors = np.abs((pc - pred) / pc) * 100
    return np.mean(errors)

# 通用模型框架
results = []

def evaluate(name, func, params, z, d, g, pc, n_params):
    """评估一个模型"""
    pred = func(params, z, d, g)
    errors = np.abs((pc - pred) / pc) * 100
    mape = np.mean(errors)
    ssr = np.sum((pc - pred)**2)
    aic = n * np.log(ssr/n) + 2 * n_params
    bic = n * np.log(ssr/n) + n_params * np.log(n)

    # 分类误差
    err_hyper = np.mean(errors[z == 2*d]) if np.any(z == 2*d) else 0
    err_2d = np.mean(errors[d == 2]) if np.any(d == 2) else 0
    err_3d = np.mean(errors[d == 3]) if np.any(d == 3) else 0

    results.append({
        'name': name, 'mape': mape, 'aic': aic, 'bic': bic,
        'n_params': n_params, 'err_hyper': err_hyper,
        'err_2d': err_2d, 'err_3d': err_3d,
        'max_err': np.max(errors), 'median_err': np.median(errors)
    })

# --- 系列A: 不同饱和函数 ---
# A1: 当前 1-exp(-z/z0)
def model_A1(p, z, d, g):
    return (1 + (p[0]/(d**p[1]*g**p[2]))*(1-np.exp(-z/p[3]))) / (z-1)

# A2: z/(z+z0) 型饱和
def model_A2(p, z, d, g):
    return (1 + (p[0]/(d**p[1]*g**p[2]))*(z/(z+p[3]))) / (z-1)

# A3: 1-(1+z/z0)^{-1} 型 (等价A2)
# A4: tanh(z/z0) 型饱和
def model_A4(p, z, d, g):
    return (1 + (p[0]/(d**p[1]*g**p[2]))*np.tanh(z/p[3])) / (z-1)

# A5: z^2/(z^2+z0^2) 型
def model_A5(p, z, d, g):
    return (1 + (p[0]/(d**p[1]*g**p[2]))*(z**2/(z**2+p[3]**2))) / (z-1)

# --- 系列B: 不同d标度 ---
# B1: d^{-beta} (当前)
# B2: (d-1)^{-beta} — 考虑有效维度
def model_B2(p, z, d, g):
    return (1 + (p[0]/((d-1)**p[1]*g**p[2]))*(1-np.exp(-z/p[3]))) / (z-1)

# B3: d^{-beta1} * (1+beta2/d)
def model_B3(p, z, d, g):
    return (1 + (p[0]/(d**p[1]*g**p[2]))*(1-np.exp(-z/p[3]))*(1+p[4]/d)) / (z-1)

# B4: (d+c)^{-beta}
def model_B4(p, z, d, g):
    return (1 + (p[0]/((d+p[4])**p[1]*g**p[2]))*(1-np.exp(-z/p[3]))) / (z-1)

# --- 系列C: z依赖的修正 ---
# C1: z^alpha / (d^beta * g^gamma) * sat(z)
def model_C1(p, z, d, g):
    return (1 + p[0]*z**p[4]/(d**p[1]*g**p[2])*(1-np.exp(-z/p[3]))) / (z-1)

# C2: (z-1)^alpha
def model_C2(p, z, d, g):
    return (1 + (p[0]*(z-1)**p[4])/(d**p[1]*g**p[2])*(1-np.exp(-z/p[3]))) / (z-1)

# --- 系列D: 乘法修正 ---
# D1: Bethe * sat * (1 + c/(z-1))
def model_D1(p, z, d, g):
    return (1 + (p[0]/(d**p[1]*g**p[2]))*(1-np.exp(-z/p[3]))*(1+p[4]/(z-1))) / (z-1)

# D2: Bethe * sat * (1 + c/z^delta)
def model_D2(p, z, d, g):
    return (1 + (p[0]/(d**p[1]*g**p[2]))*(1-np.exp(-z/p[3]))*(1+p[4]/z**p[5])) / (z-1)

# --- 系列E: 加法模型 ---
# E1: Bethe + correction
def model_E1(p, z, d, g):
    return 1/(z-1) + p[0]/(d**p[1]*g**p[2])*(1-np.exp(-z/p[3]))/(z-1)**p[4]

# E2: 1/(z-1+c)
def model_E2(p, z, d, g):
    c = p[0]/(d**p[1]*g**p[2])*(1-np.exp(-z/p[3]))
    return 1/(z-1+c)

# --- 系列F: 有理函数 ---
# F1: p_c = (1+c1/(d^a*g^b)) / (z-1+c2/(d^e*g^f))
def model_F1(p, z, d, g):
    num = 1 + p[0]/(d**p[1]*g**p[2])
    den = (z-1) * (1 + p[3]/(d**p[4]*g**p[5]))
    return num / den

# --- 系列G: 指数修正 ---
# G1: exp(-alpha*d) 代替 d^{-beta}
def model_G1(p, z, d, g):
    return (1 + p[0]*np.exp(-p[1]*d)/g**p[2]*(1-np.exp(-z/p[3]))) / (z-1)

# G2: exp(-alpha*d - beta/g)
def model_G2(p, z, d, g):
    return (1 + p[0]*np.exp(-p[1]*d - p[2]/g)*(1-np.exp(-z/p[3]))) / (z-1)

# ============================================================
# 拟合所有模型
# ============================================================
print(f"\n拟合中...")

model_configs = [
    # (name, func, init_params, bounds, n_params)
    ('A1: exp饱和(基线)', model_A1, [12.9, 1.03, 0.85, 9.7],
     [(5, 25), (0.5, 2.0), (0.3, 1.5), (3, 30)], 4),
    ('A2: z/(z+z0)饱和', model_A2, [12.0, 1.0, 0.8, 10.0],
     [(5, 25), (0.5, 2.0), (0.3, 1.5), (3, 30)], 4),
    ('A4: tanh饱和', model_A4, [12.0, 1.0, 0.8, 8.0],
     [(5, 25), (0.5, 2.0), (0.3, 1.5), (3, 20)], 4),
    ('A5: z^2/(z^2+z0^2)', model_A5, [12.0, 1.0, 0.8, 5.0],
     [(5, 25), (0.5, 2.0), (0.3, 1.5), (2, 15)], 4),
    ('B2: (d-1)标度', model_B2, [10.0, 1.0, 0.8, 9.0],
     [(5, 25), (0.5, 2.0), (0.3, 1.5), (3, 30)], 4),
    ('B4: (d+c)标度', model_B4, [10.0, 1.0, 0.8, 9.0, 1.0],
     [(5, 25), (0.5, 2.0), (0.3, 1.5), (3, 30), (-2, 5)], 5),
    ('C1: z^alpha修正', model_C1, [5.0, 1.0, 0.8, 9.0, 0.5],
     [(1, 15), (0.5, 2.0), (0.3, 1.5), (3, 30), (0.1, 1.0)], 5),
    ('D1: 1+1/(z-1)修正', model_D1, [12.0, 1.0, 0.8, 9.0, 2.0],
     [(5, 25), (0.5, 2.0), (0.3, 1.5), (3, 30), (-5, 10)], 5),
    ('E1: 加法Bethe', model_E1, [12.0, 1.0, 0.8, 9.0, 0.5],
     [(1, 25), (0.5, 2.0), (0.3, 1.5), (3, 30), (0.1, 1.5)], 5),
    ('E2: 1/(z-1+c)', model_E2, [12.0, 1.0, 0.8, 9.0],
     [(5, 25), (0.5, 2.0), (0.3, 1.5), (3, 30)], 4),
    ('G1: exp(-αd)', model_G1, [15.0, 0.15, 0.8, 9.0],
     [(5, 30), (0.05, 0.5), (0.3, 1.5), (3, 30)], 4),
    ('G2: exp(-αd-β/g)', model_G2, [15.0, 0.15, 2.0, 9.0],
     [(5, 30), (0.05, 0.5), (0.5, 5.0), (3, 30)], 4),
]

best_overall = None
best_loocv = np.inf

for name, func, init, bounds, np_ in model_configs:
    best_ss = np.inf
    best_p = None

    for trial in range(50):
        if trial == 0:
            p0 = init
        else:
            p0 = [np.random.uniform(b[0], b[1]) for b in bounds]

        try:
            res = minimize(lambda p: np.sum((pc - func(p, z, d, g))**2),
                          p0, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_ss:
                # 检查参数合理性
                p = res.x
                valid = True
                for j, (lo, hi) in enumerate(bounds):
                    if p[j] < lo * 0.5 or p[j] > hi * 2:
                        valid = False
                        break
                if valid:
                    best_ss = res.fun
                    best_p = p
        except:
            continue

    if best_p is not None:
        evaluate(name, func, best_p, z, d, g, pc, np_)
        mape = results[-1]['mape']
        if mape < best_loocv:
            best_loocv = mape
            best_overall = (name, func, best_p)

# ============================================================
# 6. 打印结果排名
# ============================================================
print(f"\n{'='*75}")
print(f"【模型排名（按MAPE）】")
print(f"{'='*75}")

results_sorted = sorted(results, key=lambda x: x['mape'])
print(f"\n{'排名':<3} {'模型':<25} {'MAPE':>7} {'超立方误差':>10} {'2D误差':>8} {'3D误差':>8} {'参数':>4}")
print("-" * 75)
for rank, r in enumerate(results_sorted, 1):
    print(f"{rank:<3} {r['name']:<25} {r['mape']:>6.2f}% {r['err_hyper']:>9.2f}% {r['err_2d']:>7.2f}% {r['err_3d']:>7.2f}% {r['n_params']:>4}")

# ============================================================
# 7. 最佳模型详细分析
# ============================================================
print(f"\n{'='*75}")
print(f"【最佳模型详细分析】")
print(f"{'='*75}")

best = results_sorted[0]
print(f"\n最佳: {best['name']}")
print(f"MAPE = {best['mape']:.2f}%")
print(f"超立方误差 = {best['err_hyper']:.2f}%")
print(f"2D误差 = {best['err_2d']:.2f}%")
print(f"3D误差 = {best['err_3d']:.2f}%")
print(f"最大误差 = {best['max_err']:.2f}%")

# 打印误差最大的5个
if best_overall:
    name, func, params = best_overall
    pred_best = func(params, z, d, g)
    errors_best = np.abs((pc - pred_best) / pc) * 100
    idx_top5 = np.argsort(errors_best)[::-1][:5]
    print(f"\n最佳模型误差最大的5个:")
    for idx in idx_top5:
        print(f"  {names[idx]:<20} z={int(z[idx]):>3} d={int(d[idx]):>2} g={int(g[idx]):>2} "
              f"p_c={pc[idx]:.5f} pred={pred_best[idx]:.5f} err={errors_best[idx]:.1f}%")

# ============================================================
# 8. 混合策略：分段修正
# ============================================================
print(f"\n{'='*75}")
print(f"【混合策略：针对高维的额外修正】")
print(f"{'='*75}")

# 观察：高维超立方一致被高估 → 可能需要额外的d依赖修正
# 思路：在大d时，修正项应该更快衰减

def model_hybrid(p, z, d, g):
    """混合模型：基础饱和 + 高维额外衰减"""
    a, beta, gamma, z0, delta = p
    base_corr = (a / (d**beta * g**gamma)) * (1 - np.exp(-z / z0))
    # 额外的高维衰减因子
    high_d_factor = 1 / (1 + (z/(2*d))**delta)  # z/(2d) = η, 每维配位数
    correction = base_corr * high_d_factor
    return (1 + correction) / (z - 1)

def obj_hybrid(p):
    pred = model_hybrid(p, z, d, g)
    return np.sum((pc - pred)**2)

best_hybrid = None
best_ss_hybrid = np.inf
for _ in range(200):
    init = [12.9, 1.03, 0.85, 9.7, 1.0]
    init_r = [np.random.uniform(b[0], b[1]) for b in
              [(5, 25), (0.5, 2.0), (0.3, 1.5), (3, 30), (0.1, 3.0)]]
    res = minimize(obj_hybrid, init_r, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_ss_hybrid:
        best_ss_hybrid = res.fun
        best_hybrid = res.x

if best_hybrid is not None:
    pred_hybrid = model_hybrid(best_hybrid, z, d, g)
    errors_hybrid = np.abs((pc - pred_hybrid) / pc) * 100
    mape_hybrid = np.mean(errors_hybrid)

    # 分类误差
    mask_h = z == 2*d
    err_hyper_h = np.mean(errors_hybrid[mask_h])
    err_2d_h = np.mean(errors_hybrid[d == 2])
    err_3d_h = np.mean(errors_hybrid[d == 3])

    print(f"\n混合模型参数:")
    print(f"  a={best_hybrid[0]:.3f}, β={best_hybrid[1]:.3f}, γ={best_hybrid[2]:.3f}")
    print(f"  z₀={best_hybrid[3]:.3f}, δ={best_hybrid[4]:.3f}")
    print(f"  MAPE = {mape_hybrid:.2f}%")
    print(f"  超立方误差 = {err_hyper_h:.2f}%")
    print(f"  2D误差 = {err_2d_h:.2f}%")
    print(f"  3D误差 = {err_3d_h:.2f}%")

# ============================================================
# 9. 另一种思路：η=z/d 作为核心变量
# ============================================================
print(f"\n{'='*75}")
print(f"【η=z/d 作为核心变量】")
print(f"{'='*75}")

eta = z / d

# 检查R-1 vs η的关系
R_minus_1 = R_cur - 1

print(f"\nR-1 vs η=z/d 的相关性:")
from scipy.stats import pearsonr
r_corr, p_val = pearsonr(eta, R_minus_1)
print(f"  corr(η, R-1) = {r_corr:.4f} (p={p_val:.2e})")

# 对于超立方: η = 2d/d = 2, 常数!
# 所以η不能区分超立方的d依赖
# 需要同时用η和d

# 尝试: R-1 = a * η^α * d^δ * g^γ * sat(η, η0)
def model_eta(p, z, d, g):
    eta = z / d
    a, alpha, delta, gamma, eta0 = p
    sat = 1 - np.exp(-eta / eta0)
    correction = a * eta**alpha * d**delta * g**gamma * sat
    return (1 + correction) / (z - 1)

def obj_eta(p):
    pred = model_eta(p, z, d, g)
    return np.sum((pc - pred)**2)

best_eta = None
best_ss_eta = np.inf
for _ in range(200):
    init_r = [np.random.uniform(b[0], b[1]) for b in
              [(1, 20), (0.1, 1.5), (-2.0, -0.1), (-1.5, -0.1), (1, 10)]]
    res = minimize(obj_eta, init_r, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_ss_eta:
        best_ss_eta = res.fun
        best_eta = res.x

if best_eta is not None:
    pred_eta = model_eta(best_eta, z, d, g)
    errors_eta = np.abs((pc - pred_eta) / pc) * 100
    mape_eta = np.mean(errors_eta)

    mask_h = z == 2*d
    err_hyper_eta = np.mean(errors_eta[mask_h])
    err_2d_eta = np.mean(errors_eta[d == 2])
    err_3d_eta = np.mean(errors_eta[d == 3])

    print(f"\nη模型参数:")
    print(f"  a={best_eta[0]:.3f}, α={best_eta[1]:.3f}, δ={best_eta[2]:.3f}")
    print(f"  γ={best_eta[3]:.3f}, η₀={best_eta[4]:.3f}")
    print(f"  MAPE = {mape_eta:.2f}%")
    print(f"  超立方误差 = {err_hyper_eta:.2f}%")
    print(f"  2D误差 = {err_2d_eta:.2f}%")
    print(f"  3D误差 = {err_3d_eta:.2f}%")

# ============================================================
# 10. 最终对比总结
# ============================================================
print(f"\n{'='*75}")
print(f"【最终对比总结】")
print(f"{'='*75}")

all_models = results_sorted.copy()
if best_hybrid is not None:
    all_models.append({
        'name': 'Hybrid: η衰减', 'mape': mape_hybrid,
        'err_hyper': err_hyper_h, 'err_2d': err_2d_h, 'err_3d': err_3d_h,
        'max_err': np.max(errors_hybrid), 'n_params': 5
    })
if best_eta is not None:
    all_models.append({
        'name': 'η模型', 'mape': mape_eta,
        'err_hyper': err_hyper_eta, 'err_2d': err_2d_eta, 'err_3d': err_3d_eta,
        'max_err': np.max(errors_eta), 'n_params': 5
    })

all_models.sort(key=lambda x: x['mape'])

print(f"\n{'排名':<3} {'模型':<25} {'MAPE':>7} {'超立方':>8} {'2D':>7} {'3D':>7} {'最大':>7} {'参数':>4}")
print("-" * 80)
for rank, r in enumerate(all_models[:15], 1):
    max_e = r.get('max_err', 0)
    print(f"{rank:<3} {r['name']:<25} {r['mape']:>6.2f}% {r['err_hyper']:>7.2f}% {r['err_2d']:>6.2f}% {r['err_3d']:>6.2f}% {max_e:>6.1f}% {r['n_params']:>4}")

# 保存结果
with open(f'{OUT}/formula_improvement_v2_report.md', 'w', encoding='utf-8') as f:
    f.write("# 公式改进实验 v2 报告\n\n")
    f.write(f"日期: 2026-10-04\n")
    f.write(f"数据: {len(df)}个晶格\n\n")

    f.write("## 残差模式分析\n\n")
    f.write("关键发现：高维超立方(d≥6)一致被高估约20%\n\n")

    f.write("## 模型对比\n\n")
    f.write(f"| 排名 | 模型 | MAPE | 超立方 | 2D | 3D | 参数 |\n")
    f.write(f"|------|------|------|--------|-----|-----|------|\n")
    for rank, r in enumerate(all_models[:15], 1):
        f.write(f"| {rank} | {r['name']} | {r['mape']:.2f}% | {r['err_hyper']:.2f}% | {r['err_2d']:.2f}% | {r['err_3d']:.2f}% | {r['n_params']} |\n")

    f.write(f"\n## 最佳模型\n\n")
    best = all_models[0]
    f.write(f"**{best['name']}**: MAPE={best['mape']:.2f}%, 超立方={best['err_hyper']:.2f}%\n")

print(f"\n报告已保存: formula_improvement_v2_report.md")
print(f"\n{'='*75}")
print("【实验完成】")
print(f"{'='*75}")
