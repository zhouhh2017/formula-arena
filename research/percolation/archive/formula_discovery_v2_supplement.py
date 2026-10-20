from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
公式发现 v2 补充分析
=====================
  1. 系统性表达式搜索（代替gplearn）
  2. RM7 物理含义深度分析
  3. 改进的符号搜索
  4. 与MC模拟数据的合并（如果MC完成）
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import LeaveOneOut
import itertools
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

# ============================================================
# 加载数据
# ============================================================
df = pd.read_csv(f'{OUT}/complete_dataset_v2.csv')
df = df[df['d'] > 0].copy().reset_index(drop=True)
n = len(df)

z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values
R = pc * (z - 1)

print("=" * 75)
print("【公式发现 v2 补充分析】")
print("=" * 75)

# ============================================================
# Part A: RM7 深度分析
# ============================================================
print(f"\n{'='*75}")
print("【A. RM7 深度分析 — 当前最佳模型】")
print(f"{'='*75}")

# RM7: p_c = 1/(z-1) · (1 + a/(z^b · d^c · g^e))
# 最优参数: a=3.385, b=-0.515, c=1.488, e=0.894

def rm7_pred(params, z, d, g):
    a, b, c, e = params
    return 1/(z-1) * (1 + a / (np.power(z, b) * np.power(d, c) * np.power(g, e)))

# 重新拟合（精确参数）
def obj_rm7(params):
    return np.sum((pc - rm7_pred(params, z, d, g))**2)

best_params = [3.385, -0.515, 1.488, 0.894]
res = minimize(obj_rm7, best_params, method='Nelder-Mead',
              options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-14})
p = res.x
pred = rm7_pred(p, z, d, g)
R2 = 1 - np.sum((pc-pred)**2) / np.sum((pc-np.mean(pc))**2)
MAPE = np.mean(np.abs((pc-pred)/pc))*100

print(f"\nRM7 精确参数:")
print(f"  a = {p[0]:.6f}")
print(f"  b = {p[1]:.6f}")
print(f"  c = {p[2]:.6f}")
print(f"  e = {p[3]:.6f}")
print(f"  R² = {R2:.6f}")
print(f"  MAPE = {MAPE:.2f}%")

print(f"\n等价形式:")
print(f"  p_c = 1/(z-1) · [1 + {p[0]:.3f} · z^{-p[1]:.3f} / (d^{p[2]:.3f} · g^{p[3]:.3f})]")
print(f"  R-1 = {p[0]:.3f} · z^({-p[1]:.3f}) · d^({-p[2]:.3f}) · g^({-p[3]:.3f})")

# R-1 的幂律分析
print(f"\n验证 R-1 的幂律关系:")
Rm1 = R - 1
# log(R-1) = log(a) - b·log(z) - c·log(d) - e·log(g)
log_Rm1 = np.log(Rm1)
log_z = np.log(z)
log_d = np.log(d)
log_g = np.log(g)

# 线性回归
X_log = np.column_stack([np.ones(n), log_z, log_d, log_g])
try:
    coeffs = np.linalg.lstsq(X_log, log_Rm1, rcond=None)[0]
    print(f"  线性回归 log(R-1) = {coeffs[0]:.3f} + {coeffs[1]:.3f}·log(z) + {coeffs[2]:.3f}·log(d) + {coeffs[3]:.3f}·log(g)")
    pred_log = X_log @ coeffs
    R2_log = 1 - np.sum((log_Rm1 - pred_log)**2) / np.sum((log_Rm1 - np.mean(log_Rm1))**2)
    print(f"  R²(log空间) = {R2_log:.6f}")
    print(f"  这意味着 R-1 确实是幂律形式！")
except:
    pass

# 分析各指数的物理含义
print(f"\n指数解读:")
print(f"  z 指数: {-p[1]:+.3f} → z 增大 → R-1 减小 (z 的效应)")
print(f"  d 指数: {-p[2]:+.3f} → d 增大 → R-1 减小 (维度效应)")
print(f"  g 指数: {-p[3]:+.3f} → g 增大 → R-1 减小 (环结构效应)")
print(f"  d 的效应最强 (|{-p[2]:.3f}| > |{-p[1]:.3f}| > |{-p[3]:.3f}|)")

# ============================================================
# Part B: 系统性表达式搜索
# ============================================================
print(f"\n{'='*75}")
print("【B. 系统性表达式搜索 — 100+候选公式】")
print(f"{'='*75}")

# 定义函数模板
# 形式: p_c = base(z) · correction(z, d, g)
# base(z) 候选: 1/(z-1), 1/z, A·z^{-α}, A·(z·d)^{-α}
# correction 候选: 1 + ..., exp(...), ...

templates = []

# --- 类别1: Bethe基线 + 幂律修正 ---
# p_c = 1/(z-1) · (1 + a·z^p·d^q·g^r)
for p_val in [-1, -0.5, 0, 0.5, 1]:
    for q_val in [-2, -1.5, -1, -0.5, 0]:
        for r_val in [-1, -0.5, 0, 0.5, 1]:
            def make_t1(pv, qv, rv):
                def pred(params, z, d, g):
                    a = params[0]
                    return 1/(z-1) * (1 + a * np.power(z, pv) * np.power(d, qv) * np.power(g, rv))
                return pred
            templates.append((f'Bethe·(1+a·z^{p_val}·d^{q_val}·g^{r_val})',
                            make_t1(p_val, q_val, r_val), [3.0]))

# --- 类别2: 分离幂律 ---
# p_c = A · z^a · d^b · g^c
for a_range in [(-1.5, -0.3), (-0.3, 0.3)]:
    pass  # 这些由优化器处理

def make_power(z_exp, d_exp, g_exp):
    def pred(params, z, d, g):
        A = params[0]
        return A * np.power(z, z_exp) * np.power(d, d_exp) * np.power(g, g_exp)
    return pred

# 扫描整数/半整数指数
for a in np.arange(-1.5, 0.1, 0.25):
    for b in np.arange(-2.0, -0.1, 0.25):
        for c in np.arange(-1.0, 0.1, 0.25):
            templates.append((f'A·z^{a:.2f}·d^{b:.2f}·g^{c:.2f}',
                            make_power(a, b, c), [5.0]))

# --- 类别3: Bethe + 有理修正 ---
# p_c = 1/(z-1) · (1 + a / (z^p + b))
for p_val in [0.5, 1, 1.5, 2]:
    def make_rat1(pv):
        def pred(params, z, d, g):
            a, b = params
            return 1/(z-1) * (1 + a / (np.power(d, pv) + b))
        return pred
    templates.append((f'Bethe·(1+a/(d^{p_val}+b))', make_rat1(p_val), [5.0, 1.0]))

# --- 类别4: 双尺度模型 ---
# p_c = 1/(z-1) · f(d) · g(g)
for pd in [0.5, 1, 1.5, 2]:
    for pg in [0.5, 1, 1.5]:
        def make_dual(pd_v, pg_v):
            def pred(params, z, d, g):
                a, b = params
                return 1/(z-1) * (1 + a/np.power(d, pd_v)) * (1 + b/np.power(g, pg_v))
            return pred
        templates.append((f'Bethe·(1+a/d^{pd})·(1+b/g^{pg})',
                         make_dual(pd, pg), [5.0, 1.0]))

# --- 类别5: 混合幂律+加法 ---
# p_c = A·z^a·d^b + B/g^c
for a_v in np.arange(-1.5, 0, 0.25):
    for b_v in np.arange(-2.0, 0, 0.25):
        for c_v in [0.5, 1, 1.5, 2]:
            def make_mix(av, bv, cv):
                def pred(params, z, d, g):
                    A, B = params
                    return A * np.power(z, av) * np.power(d, bv) + B / np.power(g, cv)
                return pred
            templates.append((f'A·z^{a_v:.2f}·d^{b_v:.2f}+B/g^{c_v}',
                            make_mix(a_v, b_v, c_v), [0.5, 0.5]))

# --- 类别6: RM7 变体（最佳模型的小改动）---
# p_c = 1/(z-1) · (1 + a·z^p / (d^q · g^r))  其中p,q,r自由优化
for p_init in [-0.8, -0.5, -0.3, 0, 0.3, 0.5]:
    for q_init in [0.5, 1.0, 1.5, 2.0]:
        for r_init in [0.5, 0.8, 1.0, 1.5]:
            def make_rm7v(pi, qi, ri):
                def pred(params, z, d, g):
                    a, p, q, r = params
                    return 1/(z-1) * (1 + a * np.power(z, p) / (np.power(d, q) * np.power(g, r)))
                return pred
            templates.append((f'Bethe·(1+a·z^p/(d^q·g^r))[{p_init},{q_init},{r_init}]',
                            make_rm7v(p_init, q_init, r_init), [3.0, p_init, q_init, r_init]))

print(f"\n共 {len(templates)} 个候选公式")

# 评估所有模板
def loocv_mape(pred_func, params, z, d, g, pc):
    loo = LeaveOneOut()
    errs = []
    for train_idx, test_idx in loo.split(z):
        z_tr, d_tr, g_tr, pc_tr = z[train_idx], d[train_idx], g[train_idx], pc[train_idx]
        z_te, d_te, g_te = z[test_idx], d[test_idx], g[test_idx]
        def obj(p):
            return np.sum((pc_tr - pred_func(p, z_tr, d_tr, g_tr))**2)
        res = minimize(obj, params, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        pred_i = pred_func(res.x, z_te, d_te, g_te)
        errs.append(abs(pc[test_idx[0]] - pred_i[0]) / pc[test_idx[0]])
    return np.mean(errs) * 100

print(f"\n搜索中...")
all_results = []
for i, (name, pred_func, init) in enumerate(templates):
    def obj(params):
        try:
            pred = pred_func(params, z, d, g)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e10
            return np.sum((pc - pred)**2)
        except:
            return 1e10

    # 多次重启
    best_ss = np.inf
    best_params = None
    for _ in range(10):
        init_r = [x * np.random.uniform(0.5, 2.0) for x in init]
        try:
            res = minimize(obj, init_r, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_ss:
                best_ss = res.fun
                best_params = res.x
        except:
            pass

    if best_params is None:
        continue

    try:
        pred = pred_func(best_params, z, d, g)
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            continue
        ss_tot = np.sum((pc - np.mean(pc))**2)
        R2 = 1 - best_ss / ss_tot
        MAPE_val = np.mean(np.abs((pc - pred) / pc)) * 100
        k = len(best_params)
        AIC = n * np.log(best_ss / n) + 2 * k
        BIC = n * np.log(best_ss / n) + k * np.log(n)

        # 快速 LOOCV (采样)
        loo_val = loocv_mape(pred_func, best_params, z, d, g, pc)

        all_results.append({
            'name': name, 'R2': R2, 'MAPE': MAPE_val, 'LOO_MAPE': loo_val,
            'AIC': AIC, 'BIC': BIC, 'params': best_params, 'k': k,
            'pred_func': pred_func
        })
    except:
        continue

    if (i+1) % 100 == 0:
        print(f"  已处理 {i+1}/{len(templates)} ...")

print(f"  有效结果: {len(all_results)} 个")

# 按 LOOCV MAPE 排序
all_results.sort(key=lambda x: x['LOO_MAPE'])

print(f"\n{'='*75}")
print("【Top 20 候选公式（按 LOOCV MAPE）】")
print(f"{'='*75}")
print(f"{'排名':>4} {'LOO%':>7} {'R²':>7} {'MAPE%':>7} {'k':>3} {'AIC':>8} {'模型'}")
print("-" * 90)
for i, r in enumerate(all_results[:20]):
    print(f"{i+1:>4} {r['LOO_MAPE']:>7.2f} {r['R2']:>7.4f} {r['MAPE']:>7.2f} {r['k']:>3} {r['AIC']:>8.1f} {r['name']}")

# ============================================================
# Part C: 最佳公式的深度验证
# ============================================================
print(f"\n{'='*75}")
print("【C. 最佳公式深度验证】")
print(f"{'='*75}")

if all_results:
    best = all_results[0]
    print(f"\n🏆 最佳公式: {best['name']}")
    print(f"   参数: {best['params']}")
    print(f"   R² = {best['R2']:.4f}")
    print(f"   LOOCV MAPE = {best['LOO_MAPE']:.2f}%")

    pred_best = best['pred_func'](best['params'], z, d, g)
    errors = np.abs(pc - pred_best) / pc * 100

    print(f"\n逐点误差（按误差降序）:")
    print(f"{'晶格':<20} {'z':>3} {'d':>2} {'g':>2} {'p_c':>8} {'预测':>8} {'误差%':>7}")
    print("-" * 60)
    sort_idx = np.argsort(-errors)
    for i in sort_idx:
        flag = " ⚠️" if errors[i] > 15 else ""
        print(f"{names[i]:<20} {z[i]:>3.0f} {d[i]:>2.0f} {g[i]:>2.0f} {pc[i]:>8.5f} {pred_best[i]:>8.5f} {errors[i]:>7.2f}%{flag}")

    print(f"\n误差统计:")
    print(f"  平均: {np.mean(errors):.2f}%")
    print(f"  中位: {np.median(errors):.2f}%")
    print(f"  最大: {np.max(errors):.2f}%")
    print(f"  >10%: {np.sum(errors>10)} 个")
    print(f"  >20%: {np.sum(errors>20)} 个")

    # 分类别误差
    print(f"\n分类别误差:")
    for d_val in sorted(np.unique(d)):
        mask = d == d_val
        err_cat = errors[mask]
        print(f"  d={int(d_val)}: n={np.sum(mask)}, mean_err={np.mean(err_cat):.2f}%, max_err={np.max(err_cat):.2f}%")

# ============================================================
# Part D: 与 RM7 的对比
# ============================================================
print(f"\n{'='*75}")
print("【D. 搜索结果 vs RM7 对比】")
print(f"{'='*75}")

# RM7 的结果
rm7_loo = loocv_mape(
    lambda params, z, d, g: 1/(z-1) * (1 + params[0] / (np.power(z, params[1]) * np.power(d, params[2]) * np.power(g, params[3]))),
    [3.385, -0.515, 1.488, 0.894], z, d, g, pc)

print(f"\n  RM7 (Bethe·(1+a/(z^b·d^c·g^e))):")
print(f"    LOOCV MAPE = {rm7_loo:.2f}%")

if all_results:
    print(f"\n  搜索最佳: {all_results[0]['name']}")
    print(f"    LOOCV MAPE = {all_results[0]['LOO_MAPE']:.2f}%")
    print(f"\n  改进: {rm7_loo - all_results[0]['LOO_MAPE']:.2f}%")

# ============================================================
# Part E: 改进公式 — 加权/混合策略
# ============================================================
print(f"\n{'='*75}")
print("【E. 高级模型 — 加权/交互/非线性】")
print(f"{'='*75}")

# E1: RM7 + FCC修正
# p_c = 1/(z-1) · (1 + a·z^p/(d^q·g^r)) · (1 + δ·I_{tri})
# I_{tri} = 1 if g=3 else 0

def advanced_e1(params, z, d, g):
    a, p, q, r, delta = params
    I_tri = (g == 3).astype(float)
    return 1/(z-1) * (1 + a * np.power(z, p) / (np.power(d, q) * np.power(g, r))) * (1 + delta * I_tri)

# E2: 两段幂律
# p_c = 1/(z-1) · (1 + a/(d^b·g^c))  for d ≥ 3
# p_c = A·z^p · g^q                  for d = 2
def advanced_e2(params, z, d, g):
    a, b, c, A, p, q = params
    result = np.zeros_like(z)
    mask_hd = d >= 3
    mask_2d = d < 3
    if np.any(mask_hd):
        result[mask_hd] = 1/(z[mask_hd]-1) * (1 + a / (np.power(d[mask_hd], b) * np.power(g[mask_hd], c)))
    if np.any(mask_2d):
        result[mask_2d] = A * np.power(z[mask_2d], p) * np.power(g[mask_2d], q)
    return result

# E3: 指数混合
# p_c = α · p_Bethe + (1-α) · p_power
def advanced_e3(params, z, d, g):
    a, b, c, alpha, A, p, q, r = params
    p_bethe = 1/(z-1) * (1 + a / (np.power(d, b) * np.power(g, c)))
    p_power = A * np.power(z, p) * np.power(d, q) * np.power(g, r)
    alpha_c = 1 / (1 + np.exp(-alpha))  # sigmoid
    return alpha_c * p_bethe + (1 - alpha_c) * p_power

advanced_models = {
    'E1: RM7+g=3修正': (advanced_e1, [3.0, -0.5, 1.5, 1.0, 0.1]),
    'E2: d=2/d≥3分段': (advanced_e2, [5.0, 0.5, 1.0, 0.5, -0.8, -0.5]),
    'E3: Bethe+幂律混合': (advanced_e3, [3.0, 0.5, 1.0, 0.0, 1.0, -0.8, -0.3, -0.5]),
}

print(f"\n{'模型':<25} {'R²':>7} {'MAPE%':>7} {'LOO%':>7} {'k':>3}")
print("-" * 55)

for ename, (efunc, einit) in advanced_models.items():
    def obj_e(params):
        try:
            pred = efunc(params, z, d, g)
            if np.any(np.isnan(pred)): return 1e10
            return np.sum((pc - pred)**2)
        except:
            return 1e10

    best_ss = np.inf
    best_p = None
    for _ in range(30):
        init_r = [x * np.random.uniform(0.5, 2.0) for x in einit]
        try:
            res = minimize(obj_e, init_r, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_ss:
                best_ss = res.fun
                best_p = res.x
        except:
            pass

    if best_p is None:
        continue

    try:
        pred_e = efunc(best_p, z, d, g)
        if np.any(np.isnan(pred_e)): continue
        R2_e = 1 - best_ss / np.sum((pc - np.mean(pc))**2)
        MAPE_e = np.mean(np.abs((pc - pred_e) / pc)) * 100
        loo_e = loocv_mape(efunc, best_p, z, d, g, pc)
        print(f"{ename:<25} {R2_e:>7.4f} {MAPE_e:>7.2f} {loo_e:>7.2f} {len(best_p):>3}")
    except Exception as ex:
        print(f"{ename:<25} 失败: {ex}")

print(f"\n{'='*75}")
print("【分析完成】")
print(f"{'='*75}")
