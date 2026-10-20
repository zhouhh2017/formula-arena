from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
公式发现实验 v2 — 从头来过
================================

核心改进（相对于v1的教训）:
  1. 用 Bethe-normalized R = p_c·(z-1) 代替原始 p_c
  2. 引入物理约束（z→∞ 时 R→1, d→∞ 时 R→1）
  3. 符号回归（遗传编程）自动搜索解析表达式
  4. 分段建模：自动检测 regime 边界
  5. 严格交叉验证（LOOCV）+ AIC/BIC 模型选择
  6. 多种候选模型族的公平竞争

数据: complete_dataset_v2.csv
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

# ============================================================
# 0. 加载数据
# ============================================================
df = pd.read_csv(f'{OUT}/complete_dataset_v2.csv')
print("=" * 75)
print("【公式发现实验 v2 — 从头来过】")
print("=" * 75)
print(f"数据: {len(df)} 个晶格")
print(f"维度: d = {sorted(df['d'].unique())}")
print(f"z 范围: [{df['z'].min()}, {df['z'].max()}]")
print(f"g 值: {sorted(df['g'].unique())}")

# 排除有限图 (d=0)
df = df[df['d'] > 0].copy().reset_index(drop=True)
n = len(df)
print(f"排除有限图后: {n} 个")

z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values

# Bethe-normalized
R = pc * (z - 1)  # R = 1 是 Bethe 极限

# ============================================================
# 1. 数据探索 — 物理约束分析
# ============================================================
print(f"\n{'='*75}")
print("【1. 数据探索与物理约束】")
print(f"{'='*75}")

print(f"\nBethe-normalized R = p_c·(z-1):")
print(f"{'晶格':<20} {'z':>3} {'d':>2} {'g':>2} {'p_c':>8} {'R':>8} {'R-1':>8}")
print("-" * 60)
order = np.argsort(R)
for i in order:
    flag = " ⚠️" if abs(R[i] - 1) > 0.5 else ""
    print(f"{names[i]:<20} {z[i]:>3.0f} {d[i]:>2.0f} {g[i]:>2.0f} {pc[i]:>8.5f} {R[i]:>8.4f} {R[i]-1:>+8.4f}{flag}")

print(f"\nR 统计:")
print(f"  min={R.min():.4f} ({names[np.argmin(R)]})")
print(f"  max={R.max():.4f} ({names[np.argmax(R)]})")
print(f"  mean={R.mean():.4f}, std={R.std():.4f}")
print(f"  R<1 的晶格: {np.sum(R < 1)} 个")
print(f"  R>1 的晶格: {np.sum(R > 1)} 个")

# 物理约束 1: z→∞ 时 R→1
print(f"\n物理约束检验:")
print(f"  z ≥ 16 的晶格: R = {R[z>=16]} (应接近1)")
print(f"  z ≥ 20 的晶格: R = {R[z>=20]} (应更接近1)")

# 物理约束 2: R 应随 z 增大趋向 1
r_Rz, p_Rz = pearsonr(R, z)
print(f"  corr(R, z) = {r_Rz:.4f} (应为负 — z越大R越接近1)")

# ============================================================
# 2. 特征工程
# ============================================================
print(f"\n{'='*75}")
print("【2. 特征工程】")
print(f"{'='*75}")

# 候选特征
features = {
    'z': z,
    'd': d,
    'g': g,
    'z_inv': 1/z,
    'd_inv': 1/d,
    'g_inv': 1/g,
    'zd': z*d,
    'zd_inv': 1/(z*d),
    'z_minus_1': z - 1,
    'bethe': 1/(z-1),       # Bethe 近似
    'R_deviation': R - 1,   # R 偏离 Bethe 的程度
    'log_z': np.log(z),
    'log_d': np.log(d),
    'log_g': np.log(g),
    'log_zd': np.log(z*d),
    'z_d_ratio': z/d,
    'g_d': g*d,
    'z_g': z*g,
}

# 特征与 R-1 的相关性
print(f"\n各特征与 R-1 的相关系数 (Pearson):")
print(f"{'特征':<15} {'corr':>8} {'p-value':>10}")
print("-" * 35)
target = R - 1  # 偏离 Bethe 的量
corrs = {}
for fname, fvals in features.items():
    r, p = pearsonr(fvals, target)
    corrs[fname] = r
    print(f"{fname:<15} {r:>+8.4f} {p:>10.2e}")

# ============================================================
# 3. 参数化模型族 — 公平竞争
# ============================================================
print(f"\n{'='*75}")
print("【3. 参数化模型族 — 公平竞争】")
print(f"{'='*75}")

# 定义 LOOCV 评估函数
def evaluate_model(predict_func, params, z, d, g, pc, R, verbose=False):
    """评估模型: 全拟合 R², LOOCV MAE, AIC, BIC"""
    pred = predict_func(params, z, d, g)

    # 全拟合
    ss_res = np.sum((pc - pred)**2)
    ss_tot = np.sum((pc - np.mean(pc))**2)
    R2 = 1 - ss_res / ss_tot
    MAE = np.mean(np.abs(pc - pred))
    MAPE = np.mean(np.abs((pc - pred) / pc)) * 100

    # LOOCV
    loo = LeaveOneOut()
    loo_errs = []
    for train_idx, test_idx in loo.split(z):
        z_tr, d_tr, g_tr, pc_tr = z[train_idx], d[train_idx], g[train_idx], pc[train_idx]
        z_te, d_te, g_te = z[test_idx], d[test_idx], g[test_idx]

        def obj_loo(p):
            return np.sum((pc_tr - predict_func(p, z_tr, d_tr, g_tr))**2)

        res_loo = minimize(obj_loo, params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        pred_i = predict_func(res_loo.x, z_te, d_te, g_te)
        loo_errs.append(np.abs(pc[test_idx[0]] - pred_i[0]))

    loo_mae = np.mean(loo_errs)
    loo_mape = np.mean(np.array(loo_errs) / pc * 100)

    # AIC/BIC
    k = len(params)
    AIC = n * np.log(ss_res / n) + 2 * k
    BIC = n * np.log(ss_res / n) + k * np.log(n)

    if verbose:
        return R2, MAE, MAPE, loo_mae, loo_mape, AIC, BIC
    return R2, loo_mae, loo_mape, AIC, BIC


# --- 模型定义 ---
models = {}

# M1: 原始幂律 (v1公式)
def m1_pred(params, z, d, g):
    A, alpha, B = params
    return A * np.power(z*d, -alpha) * (1 + B/g)
models['M1: A(zd)^-α(1+B/g)'] = (m1_pred, [3.0, 0.8, 1.0])

# M2: z和d分离
def m2_pred(params, z, d, g):
    A, az, ad, B = params
    return A * np.power(z, -az) * np.power(d, -ad) * (1 + B/g)
models['M2: Az^-α d^-γ(1+B/g)'] = (m2_pred, [2.0, 0.8, 0.5, 1.0])

# M3: Bethe基线 + 修正
def m3_pred(params, z, d, g):
    a, b, c = params
    bethe = 1.0 / (z - 1)
    return bethe * (1 + a / (d**b) + c / g)
models['M3: Bethe·(1+a/d^b+c/g)'] = (m3_pred, [1.0, 0.5, 0.5])

# M4: 指数衰减
def m4_pred(params, z, d, g):
    A, alpha, B = params
    return A * np.exp(-alpha * z * d) * (1 + B/g)
models['M4: A·exp(-α·zd)·(1+B/g)'] = (m4_pred, [100.0, 0.05, 1.0])

# M5: 有理函数
def m5_pred(params, z, d, g):
    a, b, c, e = params
    return (a + b/g) / (c + z*d + e)
models['M5: (a+b/g)/(c+zd+e)'] = (m5_pred, [5.0, 2.0, 1.0, 0.1])

# M6: Bethe·exp修正
def m6_pred(params, z, d, g):
    a, b, c = params
    bethe = 1.0 / (z - 1)
    return bethe * (1 + a * np.exp(-b * z * d / g))
models['M6: Bethe·(1+a·exp(-b·zd/g))'] = (m6_pred, [1.0, 0.1, 1.0])

# M7: 幂律 + girth² 修正
def m7_pred(params, z, d, g):
    A, alpha, B, beta = params
    return A * np.power(z*d, -alpha) * (1 + B/g + beta/g**2)
models['M7: A(zd)^-α(1+B/g+C/g²)'] = (m7_pred, [3.0, 0.8, 1.0, 0.1])

# M8: Bethe + d^(-b) 修正（无girth）
def m8_pred(params, z, d, g):
    a, b = params
    bethe = 1.0 / (z - 1)
    return bethe * (1 + a / d**b)
models['M8: Bethe·(1+a/d^b)'] = (m8_pred, [5.0, 0.5])

# M9: 通用幂律 z^α · d^β · g^γ
def m9_pred(params, z, d, g):
    A, a, b, c = params
    return A * np.power(z, a) * np.power(d, b) * np.power(g, c)
models['M9: A·z^a·d^b·g^c'] = (m9_pred, [1.0, -0.8, -0.3, -0.2])

# M10: Bethe + zd/g 交互
def m10_pred(params, z, d, g):
    a, b, c = params
    bethe = 1.0 / (z - 1)
    return bethe * (1 + a * np.power(z*d/g, -b))
models['M10: Bethe·(1+a·(zd/g)^-b)'] = (m10_pred, [5.0, 0.5, 1.0])

# 拟合和评估所有模型
print(f"\n{'模型':<30} {'R²':>7} {'MAE':>7} {'MAPE%':>7} {'LOO':>7} {'LOO%':>7} {'AIC':>8} {'BIC':>8}")
print("=" * 95)

results = {}
for mname, (pred_func, init) in models.items():
    def obj(params):
        return np.sum((pc - pred_func(params, z, d, g))**2)

    # 多次随机重启
    best_res = None
    best_ss = np.inf
    for _ in range(50):
        init_r = [x * np.random.uniform(0.3, 3.0) for x in init]
        try:
            res = minimize(obj, init_r, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_ss:
                best_ss = res.fun
                best_res = res
        except:
            pass

    if best_res is None:
        print(f"{mname:<30} 拟合失败")
        continue

    params = best_res.x
    R2, MAE, MAPE, loo_mae, loo_mape, AIC, BIC = evaluate_model(
        pred_func, params, z, d, g, pc, R, verbose=True)

    results[mname] = {
        'params': params, 'R2': R2, 'MAE': MAE, 'MAPE': MAPE,
        'loo_mae': loo_mae, 'loo_mape': loo_mape, 'AIC': AIC, 'BIC': BIC,
        'pred_func': pred_func
    }

    print(f"{mname:<30} {R2:>7.4f} {MAE:>7.5f} {MAPE:>7.2f} {loo_mae:>7.5f} {loo_mape:>7.2f} {AIC:>8.1f} {BIC:>8.1f}")

# 找出最佳模型（按 LOOCV MAE）
best_model_name = min(results, key=lambda k: results[k]['loo_mae'])
print(f"\n最佳模型（LOOCV MAE 最小）: {best_model_name}")
print(f"  LOO MAE = {results[best_model_name]['loo_mae']:.5f}")
print(f"  LOO MAPE = {results[best_model_name]['loo_mape']:.2f}%")

# ============================================================
# 4. 符号回归（遗传编程）
# ============================================================
print(f"\n{'='*75}")
print("【4. 符号回归 — 遗传编程自动搜索】")
print(f"{'='*75}")

try:
    from gplearn.genetic import SymbolicRegressor
    from gplearn.functions import make_function
    from gplearn.fitness import make_fitness

    # 准备特征矩阵
    X = np.column_stack([z, d, g])
    y = pc

    # 保护除法
    def _protected_div(x1, x2):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(x2) < 1e-10, 1.0, x1 / x2)

    # 自定义函数
    def _bethe(x):
        return 1.0 / (x - 1 + 1e-10)

    def _safe_log(x):
        return np.log(np.abs(x) + 1e-10)

    # 适应度函数（优先 LOOCV 性能）
    def _mape_fitness(y, y_pred, w):
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100
        return mape

    sr = SymbolicRegressor(
        population_size=2000,
        generations=100,
        tournament_size=20,
        stopping_criteria=0.01,  # 1% MAPE
        metric='mean absolute error',
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'inv'],
        parsimony_coefficient=0.01,
        random_state=42,
        n_jobs=1,
        verbose=1,
        hall_of_fame=10,
    )

    print("\n运行遗传编程 (2000个体 × 100代)...")
    sr.fit(X, y)

    print(f"\n最佳符号回归结果:")
    print(f"  表达式: {sr._program}")
    pred_sr = sr.predict(X)
    R2_sr = r2_score(y, pred_sr)
    MAE_sr = mean_absolute_error(y, pred_sr)
    MAPE_sr = np.mean(np.abs((y - pred_sr) / y)) * 100
    print(f"  R² = {R2_sr:.4f}, MAE = {MAE_sr:.5f}, MAPE = {MAPE_sr:.2f}%")

    # Hall of fame
    print(f"\nHall of Fame (前5):")
    for i, prog in enumerate(sr._hall_of_fame[:5]):
        pred_hof = prog.predict(X)
        mape_hof = np.mean(np.abs((y - pred_hof) / y)) * 100
        print(f"  #{i+1}: {prog}  (MAPE={mape_hof:.2f}%, len={prog.length_})")

except Exception as e:
    print(f"符号回归失败: {e}")
    sr = None

# ============================================================
# 5. 分段建模
# ============================================================
print(f"\n{'='*75}")
print("【5. 分段建模 — 自动检测 Regime】")
print(f"{'='*75}")

# 按 z·d 排序
zd = z * d
sort_idx = np.argsort(zd)

# 尝试不同的分段数
def fit_piecewise(boundaries, z, d, g, pc, pred_func, init):
    """拟合分段模型"""
    boundaries = sorted(boundaries)
    n_pieces = len(boundaries) + 1
    pieces = []

    for i in range(n_pieces):
        if i == 0:
            mask = zd <= boundaries[0]
        elif i == n_pieces - 1:
            mask = zd > boundaries[-1]
        else:
            mask = (zd > boundaries[i-1]) & (zd <= boundaries[i])

        if np.sum(mask) < 3:
            return None, np.inf

        z_p, d_p, g_p, pc_p = z[mask], d[mask], g[mask], pc[mask]

        def obj_p(params):
            return np.sum((pc_p - pred_func(params, z_p, d_p, g_p))**2)

        res_p = minimize(obj_p, init, method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        pred_p = pred_func(res_p.x, z_p, d_p, g_p)
        ss_p = np.sum((pc_p - pred_p)**2)
        R2_p = 1 - ss_p / np.sum((pc_p - np.mean(pc_p))**2)
        pieces.append({
            'mask': mask, 'params': res_p.x, 'R2': R2_p,
            'n': np.sum(mask), 'ss': ss_p
        })

    total_ss = sum(p['ss'] for p in pieces)
    total_k = len(init) * n_pieces
    total_AIC = n * np.log(total_ss / n) + 2 * total_k
    total_BIC = n * np.log(total_ss / n) + total_k * np.log(n)

    return pieces, total_AIC, total_BIC, total_ss

# 测试 M1 的分段
print(f"\n使用 M1 (A(zd)^-α(1+B/g)) 的分段模型:")
print(f"{'分段':<30} {'k':>3} {'ss':>10} {'AIC':>10} {'BIC':>10}")
print("-" * 60)

# 不分段
def obj_m1(params):
    return np.sum((pc - m1_pred(params, z, d, g))**2)

res_m1 = minimize(obj_m1, [3.0, 0.8, 1.0], method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
ss_1seg = res_m1.fun
k_1seg = 3
aic_1seg = n * np.log(ss_1seg / n) + 2 * k_1seg
bic_1seg = n * np.log(ss_1seg / n) + k_1seg * np.log(n)
print(f"{'1段 (无分段)':<30} {k_1seg:>3} {ss_1seg:>10.6f} {aic_1seg:>10.1f} {bic_1seg:>10.1f}")

# 2段：尝试不同边界
best_2seg = None
best_2seg_bic = np.inf
for boundary in [15, 20, 25, 30, 35, 40]:
    pieces, aic, bic, ss = fit_piecewise([boundary], z, d, g, pc, m1_pred, [3.0, 0.8, 1.0])
    if pieces is not None:
        print(f"{'2段 zd≤'+str(boundary):<30} {6:>3} {ss:>10.6f} {aic:>10.1f} {bic:>10.1f}")
        if bic < best_2seg_bic:
            best_2seg_bic = bic
            best_2seg = (boundary, pieces)

# 3段
best_3seg = None
best_3seg_bic = np.inf
for b1 in [12, 15, 18, 20]:
    for b2 in [30, 35, 40, 50]:
        if b2 <= b1:
            continue
        pieces, aic, bic, ss = fit_piecewise([b1, b2], z, d, g, pc, m1_pred, [3.0, 0.8, 1.0])
        if pieces is not None:
            if bic < best_3seg_bic:
                best_3seg_bic = bic
                best_3seg = (b1, b2, pieces)

if best_3seg:
    b1, b2, pieces = best_3seg
    k_3seg = 3 * 3
    ss_3seg = sum(p['ss'] for p in pieces)
    bic_3seg = n * np.log(ss_3seg / n) + k_3seg * np.log(n)
    print(f"{'3段 zd≤'+str(b1)+','+str(b2):<30} {k_3seg:>3} {ss_3seg:>10.6f} {n*np.log(ss_3seg/n)+2*k_3seg:>10.1f} {bic_3seg:>10.1f}")

    print(f"\n3段模型详情:")
    for i, p in enumerate(pieces):
        A, alpha, B = p['params']
        print(f"  段{i+1}: n={p['n']}, A={A:.3f}, α={alpha:.3f}, B={B:.3f}, R²={p['R2']:.4f}")

# ============================================================
# 6. Bethe-normalized 建模 (R = p_c·(z-1))
# ============================================================
print(f"\n{'='*75}")
print("【6. Bethe-Normalized 建模: R = p_c·(z-1)】")
print(f"{'='*75}")

print(f"\n对 R-1（偏离 Bethe 的量）建模:")

# R_models
r_models = {}

# RM1: R-1 = a/(z·d^b)
def rm1_pred(params, z, d, g):
    a, b = params
    return 1/(z-1) * (1 + a / np.power(d, b))

# RM2: R-1 = a/(z·d^b) + c/g
def rm2_pred(params, z, d, g):
    a, b, c = params
    return 1/(z-1) * (1 + a / np.power(d, b) + c / g)

# RM3: R-1 = a·(z·d)^{-b}·g^{-c}
def rm3_pred(params, z, d, g):
    a, b, c = params
    return 1/(z-1) * (1 + a * np.power(z*d, -b) * np.power(g, -c))

# RM4: R-1 = a/(d^b·g^c)
def rm4_pred(params, z, d, g):
    a, b, c = params
    return 1/(z-1) * (1 + a / (np.power(d, b) * np.power(g, c)))

# RM5: R = 1 - 1/z + a/(z·d^b)  (高维极限公式)
def rm5_pred(params, z, d, g):
    a, b = params
    return 1 - 1/z + a / (z * np.power(d, b))

# RM6: R = 1 - 1/z + a/(z·d^b) + c/(z·g)
def rm6_pred(params, z, d, g):
    a, b, c = params
    return 1 - 1/z + a / (z * np.power(d, b)) + c / (z * g)

# RM7: R-1 = a/(z^b · d^c · g^e)
def rm7_pred(params, z, d, g):
    a, b, c, e = params
    return 1/(z-1) * (1 + a / (np.power(z, b) * np.power(d, c) * np.power(g, e)))

r_model_list = {
    'RM1: 1+a/d^b': (rm1_pred, [5.0, 0.5]),
    'RM2: 1+a/d^b+c/g': (rm2_pred, [5.0, 0.5, 1.0]),
    'RM3: 1+a(zd)^-b g^-c': (rm3_pred, [5.0, 0.5, 0.5]),
    'RM4: 1+a/(d^b g^c)': (rm4_pred, [5.0, 0.5, 0.5]),
    'RM5: 1-1/z+a/(zd^b)': (rm5_pred, [7.0, 0.42]),
    'RM6: 1-1/z+a/(zd^b)+c/(zg)': (rm6_pred, [7.0, 0.42, 1.0]),
    'RM7: 1+a/(z^b d^c g^e)': (rm7_pred, [10.0, 0.5, 0.3, 0.3]),
}

# 评估R模型（直接对 pc 预测）
print(f"\n{'模型':<35} {'R²(pc)':>8} {'MAE(pc)':>8} {'MAPE%':>8} {'LOO%':>8} {'AIC':>8} {'BIC':>8}")
print("=" * 95)

r_results = {}
for rmname, (pred_func, init) in r_model_list.items():
    def obj_rm(params):
        return np.sum((pc - pred_func(params, z, d, g))**2)

    best_res = None
    best_ss = np.inf
    for _ in range(50):
        init_r = [x * np.random.uniform(0.3, 3.0) for x in init]
        try:
            res = minimize(obj_rm, init_r, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_ss:
                best_ss = res.fun
                best_res = res
        except:
            pass

    if best_res is None:
        print(f"{rmname:<35} 拟合失败")
        continue

    params = best_res.x
    R2, MAE, MAPE, loo_mae, loo_mape, AIC, BIC = evaluate_model(
        pred_func, params, z, d, g, pc, R, verbose=True)

    r_results[rmname] = {
        'params': params, 'R2': R2, 'MAE': MAE, 'MAPE': MAPE,
        'loo_mae': loo_mae, 'loo_mape': loo_mape, 'AIC': AIC, 'BIC': BIC,
        'pred_func': pred_func
    }
    print(f"{rmname:<35} {R2:>8.4f} {MAE:>8.5f} {MAPE:>8.2f} {loo_mape:>8.2f} {AIC:>8.1f} {BIC:>8.1f}")

# 最佳R模型
best_rmodel = min(r_results, key=lambda k: r_results[k]['loo_mape'])
print(f"\n最佳 Bethe-normalized 模型（LOOCV MAPE 最小）: {best_rmodel}")

# ============================================================
# 7. 交叉预测测试
# ============================================================
print(f"\n{'='*75}")
print("【7. 交叉预测测试】")
print(f"{'='*75}")

# 按维度分组
groups = {
    'd=2': np.where(d == 2)[0],
    'd=3': np.where(d == 3)[0],
    'd≥4': np.where(d >= 4)[0],
}

# 用最佳模型做交叉预测
best_overall_name = best_model_name
best_overall_func = results[best_overall_name]['pred_func']

# 也用最佳R模型
best_r_func = r_results[best_rmodel]['pred_func']

print(f"\n交叉预测（M1类模型: {best_overall_name}）:")
print(f"{'训练→测试':<25} {'MAE':>8} {'MAPE%':>8} {'最大误差%':>10}")
print("-" * 55)

for train_name, train_idx in groups.items():
    for test_name, test_idx in groups.items():
        if train_name == test_name:
            continue
        if len(test_idx) < 2:
            continue

        z_tr, d_tr, g_tr, pc_tr = z[train_idx], d[train_idx], g[train_idx], pc[train_idx]

        def obj_cross(params):
            return np.sum((pc_tr - best_overall_func(params, z_tr, d_tr, g_tr))**2)

        init_p = results[best_overall_name]['params']
        res_cross = minimize(obj_cross, init_p, method='Nelder-Mead',
                            options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})

        pred_test = best_overall_func(res_cross.x, z[test_idx], d[test_idx], g[test_idx])
        mae_cross = np.mean(np.abs(pc[test_idx] - pred_test))
        mape_cross = np.mean(np.abs((pc[test_idx] - pred_test) / pc[test_idx])) * 100
        max_err = np.max(np.abs((pc[test_idx] - pred_test) / pc[test_idx])) * 100

        print(f"{train_name+'→'+test_name:<25} {mae_cross:>8.5f} {mape_cross:>8.2f} {max_err:>10.1f}")

# ============================================================
# 8. 综合评估与排名
# ============================================================
print(f"\n{'='*75}")
print("【8. 综合评估与排名】")
print(f"{'='*75}")

# 合并所有模型结果
all_results = {}
for k, v in results.items():
    all_results[k] = v
for k, v in r_results.items():
    all_results[k] = v

# 排名（按 LOOCV MAPE）
ranked = sorted(all_results.items(), key=lambda x: x[1]['loo_mape'])

print(f"\n所有模型按 LOOCV MAPE 排名:")
print(f"{'排名':>4} {'模型':<35} {'R²':>7} {'MAPE%':>7} {'LOO%':>7} {'AIC':>8} {'BIC':>8}")
print("=" * 80)
for i, (name, res) in enumerate(ranked):
    print(f"{i+1:>4} {name:<35} {res['R2']:>7.4f} {res['MAPE']:>7.2f} {res['loo_mape']:>7.2f} {res['AIC']:>8.1f} {res['BIC']:>8.1f}")

# ============================================================
# 9. 最佳模型详细分析
# ============================================================
print(f"\n{'='*75}")
print("【9. 最佳模型详细分析】")
print(f"{'='*75}")

# 全局最佳
best_name = ranked[0][0]
best_res_dict = ranked[0][1]
best_func = best_res_dict['pred_func']
best_params = best_res_dict['params']

print(f"\n全局最佳: {best_name}")
print(f"  参数: {best_params}")
print(f"  R² = {best_res_dict['R2']:.4f}")
print(f"  MAPE = {best_res_dict['MAPE']:.2f}%")
print(f"  LOOCV MAPE = {best_res_dict['loo_mape']:.2f}%")
print(f"  AIC = {best_res_dict['AIC']:.1f}")
print(f"  BIC = {best_res_dict['BIC']:.1f}")

# 逐点预测
pred_best = best_func(best_params, z, d, g)
errors = np.abs(pc - pred_best) / pc * 100

print(f"\n逐点预测:")
print(f"{'晶格':<20} {'p_c实际':>8} {'p_c预测':>8} {'误差%':>7}")
print("-" * 50)
sort_err = np.argsort(-errors)
for i in sort_err:
    flag = " ⚠️" if errors[i] > 15 else ""
    print(f"{names[i]:<20} {pc[i]:>8.5f} {pred_best[i]:>8.5f} {errors[i]:>7.2f}%{flag}")

print(f"\n误差统计:")
print(f"  平均误差: {np.mean(errors):.2f}%")
print(f"  中位误差: {np.median(errors):.2f}%")
print(f"  最大误差: {np.max(errors):.2f}%")
print(f"  误差 > 5%: {np.sum(errors > 5)} 个")
print(f"  误差 > 10%: {np.sum(errors > 10)} 个")
print(f"  误差 > 20%: {np.sum(errors > 20)} 个")

# ============================================================
# 10. Bootstrap 不确定性估计
# ============================================================
print(f"\n{'='*75}")
print("【10. Bootstrap 不确定性估计】")
print(f"{'='*75}")

n_boot = 500
boot_params = []
boot_loo = []
np.random.seed(42)

for b in range(n_boot):
    # 重采样
    idx = np.random.choice(n, size=n, replace=True)
    z_b, d_b, g_b, pc_b = z[idx], d[idx], g[idx], pc[idx]

    def obj_boot(params):
        return np.sum((pc_b - best_func(params, z_b, d_b, g_b))**2)

    res_boot = minimize(obj_boot, best_params, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    boot_params.append(res_boot.x)

boot_params = np.array(boot_params)
print(f"\n{best_name} 参数 Bootstrap ({n_boot} 次):")
for i in range(len(best_params)):
    p = boot_params[:, i]
    print(f"  param[{i}]: {np.mean(p):.4f} ± {np.std(p):.4f}  "
          f"95% CI: [{np.percentile(p, 2.5):.4f}, {np.percentile(p, 97.5):.4f}]")

# ============================================================
# 保存结果
# ============================================================
print(f"\n{'='*75}")
print("【完成 — 保存结果】")
print(f"{'='*75}")

# 保存综合结果
summary = []
for name, res in ranked:
    summary.append({
        'model': name,
        'R2': res['R2'],
        'MAPE': res['MAPE'],
        'LOOCV_MAPE': res['loo_mape'],
        'AIC': res['AIC'],
        'BIC': res['BIC'],
        'params': str(res['params']),
    })
df_summary = pd.DataFrame(summary)
df_summary.to_csv(f'{OUT}/formula_discovery_v2_results.csv', index=False)
print(f"结果保存至: formula_discovery_v2_results.csv")
print(f"\n{'='*75}")
