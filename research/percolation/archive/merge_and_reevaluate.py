from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
MC 补充数据合并 + 公式重新评估
================================
等 MC 模拟完成后运行此脚本：
  1. 合并新 MC 数据到完整数据集
  2. 用新数据重新评估所有公式
  3. 验证公式的泛化能力
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import LeaveOneOut
import warnings, os
warnings.filterwarnings('ignore')

OUT = ROOT

# ============================================================
# 1. 合并数据
# ============================================================
print("=" * 75)
print("【数据合并与公式重新评估】")
print("=" * 75)

# 加载基础数据集
df_base = pd.read_csv(f'{OUT}/complete_dataset_v2.csv')
print(f"基础数据集: {len(df_base)} 个晶格")

# 检查 MC 补充数据
mc_file = f'{OUT}/mc_supplement_dataset.csv'
if os.path.exists(mc_file):
    df_mc = pd.read_csv(mc_file)
    print(f"MC 补充数据: {len(df_mc)} 个晶格")

    # 合并（去重）
    df_all = pd.concat([df_base, df_mc], ignore_index=True)
    df_all = df_all.drop_duplicates(subset='name', keep='last')
    df_all = df_all[df_all['d'] > 0].copy().reset_index(drop=True)

    print(f"合并后: {len(df_all)} 个晶格")
    print(f"新增晶格:")
    new_names = set(df_mc['name']) - set(df_base['name'])
    for name in new_names:
        row = df_all[df_all['name'] == name].iloc[0]
        print(f"  {name}: z={int(row['z'])}, d={int(row['d'])}, g={int(row['g'])}, pc={row['pc']:.4f}")
else:
    print("MC 补充数据尚未生成，使用基础数据集")
    df_all = df_base.copy()

# 保存合并后的数据集
df_all.to_csv(f'{OUT}/complete_dataset_v3.csv', index=False)

n = len(df_all)
z = df_all['z'].values.astype(float)
d = df_all['d'].values.astype(float)
g = df_all['g'].values.astype(float)
pc = df_all['pc'].values
names = df_all['name'].values

print(f"\n数据概况:")
print(f"  总晶格: {n}")
print(f"  维度: {sorted(np.unique(d.astype(int)))}")
print(f"  z 范围: [{int(z.min())}, {int(z.max())}]")
print(f"  g 值: {sorted(np.unique(g.astype(int)))}")

# ============================================================
# 2. 重新评估核心公式
# ============================================================
print(f"\n{'='*75}")
print("【公式重新评估】")
print(f"{'='*75}")

def formula_v2(params, z, d, g):
    a, alpha, beta, gamma = params
    return 1/(z-1) * (1 + a * np.power(z, alpha) / (np.power(d, beta) * np.power(g, gamma)))

# 全局优化
def obj(params):
    try:
        pred = formula_v2(params, z, d, g)
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            return 1e10
        return np.sum((pc - pred)**2)
    except:
        return 1e10

print(f"\n全局优化 (differential evolution)...")
bounds = [(0.1, 20), (0.1, 1.0), (0.5, 2.5), (0.1, 1.5)]
res_de = differential_evolution(obj, bounds, seed=42, maxiter=2000, tol=1e-14,
                                 polish=True, mutation=(0.5, 1.5), recombination=0.9)
p_opt = res_de.x

pred = formula_v2(p_opt, z, d, g)
R2 = 1 - np.sum((pc-pred)**2)/np.sum((pc-np.mean(pc))**2)
MAPE = np.mean(np.abs((pc-pred)/pc))*100

# LOOCV
loo = LeaveOneOut()
loo_errs = []
for tr_idx, te_idx in loo.split(z):
    def obj_loo(params):
        try:
            pred = formula_v2(params, z[tr_idx], d[tr_idx], g[tr_idx])
            return np.sum((pc[tr_idx] - pred)**2)
        except:
            return 1e10
    res_loo = minimize(obj_loo, p_opt, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
    try:
        pred_i = formula_v2(res_loo.x, z[te_idx], d[te_idx], g[te_idx])
        loo_errs.append(abs(pc[te_idx[0]] - pred_i[0]) / pc[te_idx[0]])
    except:
        loo_errs.append(1.0)
loo_mape = np.mean(loo_errs) * 100

k = 4
AIC = n * np.log(np.sum((pc-pred)**2)/n) + 2*k
BIC = n * np.log(np.sum((pc-pred)**2)/n) + k * np.log(n)

print(f"\n最优参数:")
print(f"  a = {p_opt[0]:.6f}")
print(f"  α = {p_opt[1]:.6f}")
print(f"  β = {p_opt[2]:.6f}")
print(f"  γ = {p_opt[3]:.6f}")
print(f"  R² = {R2:.6f}")
print(f"  MAPE = {MAPE:.2f}%")
print(f"  LOOCV MAPE = {loo_mape:.2f}%")
print(f"  AIC = {AIC:.1f}")
print(f"  BIC = {BIC:.1f}")

# ============================================================
# 3. 固定指数的 1 参数公式评估
# ============================================================
print(f"\n{'='*75}")
print("【固定指数公式评估】")
print(f"{'='*75}")

fixed_formulas = [
    ('v2a: z^{0.5}/d^{1.5}/g^{1.0}', 0.5, 1.5, 1.0),
    ('v2b: z^{0.5}/d^{1.5}/g^{0.5}', 0.5, 1.5, 0.5),
    ('v2c: z^{0.6}/d^{1.5}/g^{0.5}', 0.6, 1.5, 0.5),
    ('v2d: z^{0.5}/d^{1.6}/g^{0.5}', 0.5, 1.6, 0.5),
    ('v2e: z^{0.5}/d^{3/2}/√g',      0.5, 1.5, 0.5),
]

print(f"\n{'公式':<30} {'a':>7} {'R²':>7} {'MAPE%':>7} {'LOO%':>7} {'AIC':>8}")
print("-" * 70)

for fname, alpha_f, beta_f, gamma_f in fixed_formulas:
    def pred_fixed(params, z, d, g):
        a = params[0]
        return 1/(z-1) * (1 + a * np.power(z, alpha_f) / (np.power(d, beta_f) * np.power(g, gamma_f)))

    def obj_fixed(params):
        try:
            pred = pred_fixed(params, z, d, g)
            return np.sum((pc - pred)**2)
        except:
            return 1e10

    res_f = minimize(obj_fixed, [3.0], method='Nelder-Mead',
                    options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
    a_f = res_f.x[0]
    pred_f = pred_fixed([a_f], z, d, g)
    R2_f = 1 - np.sum((pc-pred_f)**2)/np.sum((pc-np.mean(pc))**2)
    MAPE_f = np.mean(np.abs((pc-pred_f)/pc))*100

    loo_errs_f = []
    for tr_idx, te_idx in loo.split(z):
        def obj_loo_f(p):
            return np.sum((pc[tr_idx] - pred_fixed(p, z[tr_idx], d[tr_idx], g[tr_idx]))**2)
        res_lf = minimize(obj_loo_f, [a_f], method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        try:
            pred_i = pred_fixed(res_lf.x, z[te_idx], d[te_idx], g[te_idx])
            loo_errs_f.append(abs(pc[te_idx[0]] - pred_i[0]) / pc[te_idx[0]])
        except:
            loo_errs_f.append(1.0)
    loo_f = np.mean(loo_errs_f) * 100

    AIC_f = n * np.log(np.sum((pc-pred_f)**2)/n) + 2
    print(f"{fname:<30} {a_f:>7.3f} {R2_f:>7.4f} {MAPE_f:>7.2f} {loo_f:>7.2f} {AIC_f:>8.1f}")

# ============================================================
# 4. 逐点误差（新数据验证）
# ============================================================
print(f"\n{'='*75}")
print("【逐点预测误差（最佳 1-参数公式）】")
print(f"{'='*75}")

# 使用最佳 1-参数公式
best_1p = ('v2c', 0.6, 1.5, 0.5)
def pred_best(params, z, d, g):
    a = params[0]
    return 1/(z-1) * (1 + a * np.power(z, best_1p[1]) / (np.power(d, best_1p[2]) * np.power(g, best_1p[3])))

def obj_best(params):
    return np.sum((pc - pred_best(params, z, d, g))**2)

res_best = minimize(obj_best, [2.0], method='Nelder-Mead',
                   options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
a_best = res_best.x[0]
pred_best_vals = pred_best([a_best], z, d, g)
errors = np.abs(pc - pred_best_vals) / pc * 100

print(f"\n公式: p_c = 1/(z-1) · (1 + {a_best:.3f} · z^{best_1p[1]}/(d^{best_1p[2]}·g^{best_1p[3]}))")
print(f"\n{'晶格':<20} {'z':>3} {'d':>2} {'g':>2} {'p_c':>8} {'预测':>8} {'误差%':>7} {'来源'}")
print("-" * 70)

sort_idx = np.argsort(-errors)
for i in sort_idx:
    source = df_all.iloc[i]['source']
    is_new = source != df_base.iloc[0]['source'] if len(df_base) > 0 else False
    flag = " ⚠️" if errors[i] > 15 else ""
    new_flag = " 🆕" if names[i] in new_names else ""
    print(f"{names[i]:<20} {z[i]:>3.0f} {d[i]:>2.0f} {g[i]:>2.0f} {pc[i]:>8.5f} {pred_best_vals[i]:>8.5f} {errors[i]:>7.2f}%{flag}{new_flag}")

print(f"\n误差统计:")
print(f"  平均: {np.mean(errors):.2f}%")
print(f"  中位: {np.median(errors):.2f}%")
print(f"  最大: {np.max(errors):.2f}%")
print(f"  >10%: {np.sum(errors>10)} 个")
print(f"  >20%: {np.sum(errors>20)} 个")

if new_names:
    print(f"\n新 MC 数据的误差:")
    for name in new_names:
        idx = np.where(names == name)[0]
        if len(idx) > 0:
            i = idx[0]
            print(f"  {name}: 误差 = {errors[i]:.2f}%")

print(f"\n{'='*75}")
print("【合并与评估完成】")
print(f"{'='*75}")
