"""
深入验证 II：B 参数稳定性 + 模型复杂度 + 分组交叉验证 (v2)
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============ 数据 ============

lattices_all = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'pc': 0.697040},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'pc': 0.592746},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'pc': 0.500000},
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'pc': 0.652703},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'pc': 0.311600},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'pc': 0.245800},
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'pc': 0.431900},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'pc': 0.196737},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'pc': 0.140810},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'pc': 0.109460},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'g': 4, 'pc': 0.089270},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'g': 4, 'pc': 0.075050},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'g': 4, 'pc': 0.064530},
    {'name': 'MC_sq+[1,2]', 'z': 8, 'd': 2, 'g': 3, 'pc': 0.4046},
    {'name': 'MC_sq+[1,4]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2338},
    {'name': 'MC_sq+[1,3]', 'z': 8, 'd': 2, 'g': 4, 'pc': 0.3399},
    {'name': 'MC_sq+[1,2,3]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2834},
]

n = len(lattices_all)
z = np.array([lat['z'] for lat in lattices_all])
d = np.array([lat['d'] for lat in lattices_all])
g = np.array([lat['g'] for lat in lattices_all])
pc = np.array([lat['pc'] for lat in lattices_all])

# ============ 标度律 ============

def scaling_zdg(params, z, d, g):
    A, alpha, gamma, B = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * (1 + B / g)

def scaling_zd(params, z, d):
    A, alpha, gamma = params
    return A * np.power(z, -alpha) * np.power(d, -gamma)

def robust_fit(formula, params0, z, d, pc, g=None, max_restarts=5):
    """Robust fitting with multiple random restarts"""
    best_result = None
    best_loss = np.inf

    # Try multiple starting points
    starts = [params0]
    np.random.seed(0)
    for _ in range(max_restarts - 1):
        perturbation = np.random.uniform(-0.3, 0.3, size=len(params0))
        starts.append(params0 * (1 + perturbation))

    for start in starts:
        def objective(p):
            if g is not None:
                pred = formula(p, z, d, g)
            else:
                pred = formula(p, z, d)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e10
            return np.sum((pred - pc) ** 2)

        result = minimize(objective, start, method='Nelder-Mead',
                         options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})

        if result.fun < best_loss:
            pred = formula(result.x, z, d, g) if g is not None else formula(result.x, z, d)
            if not np.any(np.isnan(pred)):
                best_loss = result.fun
                best_result = result

    if best_result is None:
        return None

    if g is not None:
        pred = formula(best_result.x, z, d, g)
    else:
        pred = formula(best_result.x, z, d)

    if np.any(np.isnan(pred)):
        return None

    ss_res = np.sum((pc - pred) ** 2)
    ss_tot = np.sum((pc - np.mean(pc)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(pc - pred))
    k = len(best_result.x)

    aic = len(pc) * np.log(ss_res / len(pc)) + 2 * k
    bic = len(pc) * np.log(ss_res / len(pc)) + k * np.log(len(pc))

    return {'params': best_result.x, 'R2': r2, 'MAE': mae, 'AIC': aic, 'BIC': bic}

# ============ 基准拟合 ============

print("="*80)
print("【基准拟合】")
print("="*80)

result_zd = robust_fit(scaling_zd, [3.0, 0.7, 0.8], z, d, pc)
result_zdg = robust_fit(scaling_zdg, [2.3, 0.8, 0.7, 1.5], z, d, pc, g)

print(f"\n模型1 (3参数): p_c = A·z^{{-α}}·d^{{-γ}}")
print(f"  R² = {result_zd['R2']:.6f}, MAE = {result_zd['MAE']:.6f}")
print(f"  AIC = {result_zd['AIC']:.4f}, BIC = {result_zd['BIC']:.4f}")
print(f"  参数: A={result_zd['params'][0]:.4f}, α={result_zd['params'][1]:.4f}, γ={result_zd['params'][2]:.4f}")

print(f"\n模型2 (4参数): p_c = A·z^{{-α}}·d^{{-γ}}·(1+B/g)")
print(f"  R² = {result_zdg['R2']:.6f}, MAE = {result_zdg['MAE']:.6f}")
print(f"  AIC = {result_zdg['AIC']:.4f}, BIC = {result_zdg['BIC']:.4f}")
print(f"  参数: A={result_zdg['params'][0]:.4f}, α={result_zdg['params'][1]:.4f}, "
      f"γ={result_zdg['params'][2]:.4f}, B={result_zdg['params'][3]:.4f}")

# ============ 1. AIC/BIC 比较 ============

print("\n" + "="*80)
print("【1. AIC/BIC 模型比较】")
print("="*80)

delta_aic = result_zd['AIC'] - result_zdg['AIC']
delta_bic = result_zd['BIC'] - result_zdg['BIC']

print(f"\nΔAIC (3参数-4参数) = {delta_aic:.4f}")
print(f"ΔBIC (3参数-4参数) = {delta_bic:.4f}")

if delta_aic > 10:
    print("  ✓✓✓ 强证据支持 4 参数模型 (ΔAIC > 10)")
elif delta_aic > 2:
    print("  ✓✓ 显著证据支持 4 参数模型 (ΔAIC > 2)")
elif delta_aic > 0:
    print("  ✓ 弱证据支持 4 参数模型 (ΔAIC > 0)")
else:
    print("  ✗ 不支持 4 参数模型（3 参数更优）")

if delta_bic > 0:
    print(f"  ✓ BIC 也支持 4 参数模型")
else:
    print(f"  ✗ BIC 偏好 3 参数模型")

import math
try:
    w_aic = 1 / (1 + math.exp(-delta_aic / 2))
    print(f"\nAIC 权重 (4参数模型): {w_aic:.4f}")
    if w_aic > 0.9:
        print("  ✓ 4 参数模型概率 > 90%")
except:
    if delta_aic > 0:
        print(f"\nAIC 权重 (4参数模型): ~1.000")

# ============ 2. Bootstrap B 参数分布 ============

print("\n" + "="*80)
print("【2. Bootstrap B 参数稳定性（1000 次）】")
print("="*80)

n_boot = 1000
np.random.seed(42)
b_values = []
fail_count = 0

for b_idx in range(n_boot):
    indices = np.random.choice(n, size=n, replace=True)

    # 确保至少有一个 g=6 和一个 g=3 的点
    g_boot = g[indices]
    if len(np.unique(g_boot)) < 2:
        fail_count += 1
        continue

    z_b, d_b, g_b, pc_b = z[indices], d[indices], g[indices], pc[indices]

    result_b = robust_fit(scaling_zdg, [2.3, 0.8, 0.7, 1.5], z_b, d_b, pc_b, g_b, max_restarts=3)
    if result_b is not None:
        b_values.append(result_b['params'][3])
    else:
        fail_count += 1

b_values = np.array(b_values)
print(f"\n成功: {len(b_values)}/{n_boot}, 失败: {fail_count}")

if len(b_values) > 0:
    # 去掉极端值
    q01 = np.percentile(b_values, 1)
    q99 = np.percentile(b_values, 99)
    mask = (b_values >= q01) & (b_values <= q99)
    b_clean = b_values[mask]

    print(f"\nB 的分布（1%-99% 截断）:")
    print(f"  有效样本: {len(b_clean)}")
    print(f"  中位数: {np.median(b_clean):.4f}")
    print(f"  均值: {np.mean(b_clean):.4f}")
    print(f"  std: {np.std(b_clean):.4f}")
    print(f"  CV: {np.std(b_clean)/abs(np.median(b_clean))*100:.1f}%")
    print(f"  95% CI: [{np.percentile(b_clean, 2.5):.4f}, {np.percentile(b_clean, 97.5):.4f}]")

    # 原始分布
    print(f"\nB 的原始分布:")
    print(f"  中位数: {np.median(b_values):.4f}")
    print(f"  均值: {np.mean(b_values):.4f}")
    print(f"  std: {np.std(b_values):.4f}")
    print(f"  min: {np.min(b_values):.4f}, max: {np.max(b_values):.4f}")
    print(f"  5%: {np.percentile(b_values, 5):.4f}, 95%: {np.percentile(b_values, 95):.4f}")

# ============ 3. 按 girth 分组交叉验证 ============

print("\n" + "="*80)
print("【3. 按 girth 分组交叉验证】")
print("="*80)

g_values = sorted(set(g))
print(f"\ngirth 分布:")
for gv in g_values:
    names = [lattices_all[i]['name'] for i in range(n) if g[i] == gv]
    print(f"  g={gv}: {len(names)} 个 → {names}")

gcv_results = {}

for g_test_val in g_values:
    train_mask = g != g_test_val
    test_mask = g == g_test_val

    if np.sum(train_mask) < 5:
        continue

    z_tr, d_tr, g_tr, pc_tr = z[train_mask], d[train_mask], g[train_mask], pc[train_mask]
    z_te, d_te, g_te, pc_te = z[test_mask], d[test_mask], g[test_mask], pc[test_mask]

    res = robust_fit(scaling_zdg, [2.3, 0.8, 0.7, 1.5], z_tr, d_tr, pc_tr, g_tr)
    if res is None:
        continue

    pred_te = scaling_zdg(res['params'], z_te, d_te, g_te)
    errors = np.abs(pc_te - pred_te)
    mae_g = np.mean(errors)
    names_te = [lattices_all[i]['name'] for i in range(n) if g[i] == g_test_val]

    gcv_results[g_test_val] = {
        'mae': mae_g,
        'params': res['params'],
        'names': names_te,
        'preds': pred_te,
        'trues': pc_te
    }

    print(f"\n  测试 g={g_test_val} ({len(names_te)} 个: {names_te}):")
    print(f"    参数: A={res['params'][0]:.3f}, α={res['params'][1]:.3f}, "
          f"γ={res['params'][2]:.3f}, B={res['params'][3]:.3f}")
    print(f"    MAE = {mae_g:.6f}")
    for i, nm in enumerate(names_te):
        err = errors[i]
        print(f"      {nm}: 真实={pc_te[i]:.5f}, 预测={pred_te[i]:.5f}, 误差={err:.5f} ({err/pc_te[i]:.1%})")

# ============ 4. 参数敏感性 ============

print("\n" + "="*80)
print("【4. 参数敏感性分析（±50%）】")
print("="*80)

params_opt = result_zdg['params']
param_names = ['A', 'α', 'γ', 'B']
sensitivities = {}

for p_idx in range(4):
    orig = params_opt[p_idx]
    maes_pos = []
    maes_neg = []
    perturb_range = np.linspace(0, 0.5, 20)

    for frac in perturb_range:
        pp = params_opt.copy()
        pp[p_idx] = orig * (1 + frac)
        pred = scaling_zdg(pp, z, d, g)
        maes_pos.append(np.mean(np.abs(pc - pred)))

        pp = params_opt.copy()
        pp[p_idx] = orig * (1 - frac)
        pred = scaling_zdg(pp, z, d, g)
        maes_neg.append(np.mean(np.abs(pc - pred)))

    maes_pos = np.array(maes_pos)
    maes_neg = np.array(maes_neg)
    base_mae = maes_pos[0]

    # Sensitivity = max MAE change per 10% parameter change
    max_change = max(np.max(maes_pos) - base_mae, np.max(maes_neg) - base_mae)
    sensitivity = max_change / base_mae / 5 * 10  # per 10% change

    sensitivities[p_idx] = sensitivity

    print(f"\n  {param_names[p_idx]} (={orig:.4f}):")
    print(f"    基准 MAE = {base_mae:.6f}")
    print(f"    +50% → MAE = {maes_pos[-1]:.6f} ({(maes_pos[-1]-base_mae)/base_mae*100:+.1f}%)")
    print(f"    -50% → MAE = {maes_neg[-1]:.6f} ({(maes_neg[-1]-base_mae)/base_mae*100:+.1f}%)")
    print(f"    敏感度 = {sensitivity:.3f} (每 10% 参数变化 → MAE 变化百分比)")

print(f"\n  敏感度排序（最敏感 → 最不敏感）:")
sorted_idx = sorted(sensitivities.keys(), key=lambda i: sensitivities[i], reverse=True)
for idx in sorted_idx:
    print(f"    {param_names[idx]}: {sensitivities[idx]:.3f}")

# ============ 5. Jacobian 条件数 ============

print("\n" + "="*80)
print("【5. 参数可识别性（Jacobian 条件数）】")
print("="*80)

eps = 1e-6
J = np.zeros((n, 4))

for j in range(4):
    p_plus = params_opt.copy(); p_plus[j] += eps
    p_minus = params_opt.copy(); p_minus[j] -= eps
    pred_plus = scaling_zdg(p_plus, z, d, g)
    pred_minus = scaling_zdg(p_minus, z, d, g)
    J[:, j] = (pred_plus - pred_minus) / (2 * eps)

# Scale columns by parameter value for relative condition number
for j in range(4):
    J[:, j] *= params_opt[j]

U, S, Vt = np.linalg.svd(J)
cond = S[0] / S[-1]

print(f"\nJacobian 奇异值: {['%.2e' % s for s in S]}")
print(f"条件数 = {cond:.2f}")

if cond < 30:
    print("  ✓ 条件数优秀（< 30）— 参数强可识别")
elif cond < 100:
    print("  ✓ 条件数良好（< 100）— 参数可识别")
elif cond < 1000:
    print("  ⚠️ 条件数中等（100-1000）— 参数弱可识别")
else:
    print("  ✗ 条件数差（> 1000）— 参数不可识别！")

# Correlation
print(f"\n参数相关性矩阵:")
corr = np.corrcoef(J.T)
header = "        " + "".join(f"{pn:>10}" for pn in param_names)
print(header)
for i, pn in enumerate(param_names):
    row = f"  {pn:<6}" + "".join(f"{corr[i,j]:10.4f}" for j in range(4))
    print(row)

max_corr = np.max(np.abs(corr[np.triu_indices(4, k=1)]))
print(f"\n最大参数间相关性: {max_corr:.4f}")
if max_corr > 0.95:
    print("  ✗ 存在严重共线性！")
elif max_corr > 0.8:
    print("  ⚠️ 存在较强共线性")
else:
    print("  ✓ 共线性不严重")

# ============ 6. 综合评估 ============

print("\n" + "="*80)
print("【6. 综合可靠性评估】")
print("="*80)

strengths = []
issues = []

# R²
if result_zdg['R2'] > 0.99:
    strengths.append(f"全数据集 R² = {result_zdg['R2']:.6f} > 0.99")
else:
    issues.append(f"全数据集 R² = {result_zdg['R2']:.6f} < 0.99")

# AIC/BIC
if delta_aic > 2:
    strengths.append(f"ΔAIC = {delta_aic:.1f} > 2（4 参数显著优于 3 参数）")
elif delta_aic > 0:
    strengths.append(f"ΔAIC = {delta_aic:.1f} > 0（4 参数略优）")
else:
    issues.append(f"ΔAIC = {-delta_aic:.1f}（3 参数更优 → 4 参数过拟合风险）")

if delta_bic > 0:
    strengths.append(f"ΔBIC = {delta_bic:.1f} > 0（BIC 也支持 4 参数）")
else:
    issues.append(f"ΔBIC = {-delta_bic:.1f} < 0（BIC 偏好 3 参数）")

# Condition number
if cond < 100:
    strengths.append(f"条件数 = {cond:.1f} < 100（参数可识别）")
elif cond < 1000:
    issues.append(f"条件数 = {cond:.1f}（100-1000，参数弱可识别）")
else:
    issues.append(f"条件数 = {cond:.1f} > 1000（参数不可识别）")

# Correlation
if max_corr < 0.8:
    strengths.append(f"最大参数相关性 = {max_corr:.3f} < 0.8（无共线性）")
elif max_corr < 0.95:
    issues.append(f"最大参数相关性 = {max_corr:.3f}（0.8-0.95，中度共线性）")
else:
    issues.append(f"最大参数相关性 = {max_corr:.3f} > 0.95（严重共线性）")

# Bootstrap B stability
if len(b_values) > 0:
    b_clean = b_values[(b_values >= q01) & (b_values <= q99)]
    b_cv = np.std(b_clean) / abs(np.median(b_clean)) * 100
    if b_cv < 30:
        strengths.append(f"B 变异系数 = {b_cv:.1f}% < 30%（Bootstrap 稳定）")
    elif b_cv < 100:
        issues.append(f"B 变异系数 = {b_cv:.1f}%（30-100%，中度不稳定）")
    else:
        issues.append(f"B 变异系数 = {b_cv:.1f}% > 100%（高度不稳定）")

# Group CV
gcv_maes = [v['mae'] for v in gcv_results.values()]
if gcv_maes:
    mean_gcv_mae = np.mean(gcv_maes)
    if mean_gcv_mae < 0.05:
        strengths.append(f"分组 CV MAE = {mean_gcv_mae:.4f} < 0.05（泛化良好）")
    elif mean_gcv_mae < 0.10:
        strengths.append(f"分组 CV MAE = {mean_gcv_mae:.4f} < 0.10（泛化可接受）")
    else:
        issues.append(f"分组 CV MAE = {mean_gcv_mae:.4f} > 0.10（泛化能力不足）")

# Sample size vs parameters
ratio = n / 4
if ratio > 10:
    strengths.append(f"数据/参数比 = {n}/4 = {ratio:.1f} > 10（充分）")
elif ratio > 5:
    strengths.append(f"数据/参数比 = {n}/4 = {ratio:.1f} > 5（足够）")
else:
    issues.append(f"数据/参数比 = {n}/4 = {ratio:.1f} < 5（可能过拟合）")

print(f"\n✓ 强度 ({len(strengths)}):")
for s in strengths:
    print(f"  {s}")

print(f"\n✗ 问题 ({len(issues)}):")
if issues:
    for iss in issues:
        print(f"  {iss}")
else:
    print("  无重大问题")

score = len(strengths) / (len(strengths) + len(issues)) * 100 if (len(strengths) + len(issues)) > 0 else 0

print(f"\n可靠性评分: {score:.0f}%")
if score >= 80 and not any("不可识别" in i for i in issues):
    print("结论: ★★★ 发现高度可靠")
elif score >= 60:
    print("结论: ★★☆ 发现基本可靠，但存在需要注意的问题")
elif score >= 40:
    print("结论: ★☆☆ 发现有价值但需谨慎解读")
else:
    print("结论: ☆☆☆ 发现需要更多数据验证")

print("\n" + "="*80)
print("验证完成")
print("="*80)
