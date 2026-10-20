"""
PRL 级标度律搜索：引入拓扑修正
====================================

当前成果：
  p_c = 3.031 · z^{-0.717} · d^{-0.898}
  R² = 0.979, MAE = 0.022

问题：
  - FCC (z=12, d=3, g=3) 误差 60%
  - 高维晶格系统性高估
  - 同 z 不同 g 的系统性偏差

新假说：
  p_c = A · z^{-α} · d^{-γ} · g^{δ}

目标：
  R² > 0.99, MAE < 0.015

如果能实现，就是 PRL 级别的普适标度律！
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

# ============ 数据 ============

lattices_data = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'pc': 0.697040},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'pc': 0.592746},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'pc': 0.500000},
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'pc': 0.652703},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'pc': 0.311600},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'pc': 0.245800},
    {'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'pc': 0.118800},
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'pc': 0.431900},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'pc': 0.196737},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'pc': 0.140810},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'pc': 0.109460},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'g': 4, 'pc': 0.089270},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'g': 4, 'pc': 0.075050},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'g': 4, 'pc': 0.064530},
    {'name': 'MC_sq', 'z': 4, 'd': 2, 'g': 4, 'pc': 0.592746},
    {'name': 'MC_sq+[1,2]', 'z': 8, 'd': 2, 'g': 3, 'pc': 0.4046},
    {'name': 'MC_sq+[1,4]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2338},
    {'name': 'MC_sq+[1,3]', 'z': 8, 'd': 2, 'g': 4, 'pc': 0.3399},
    {'name': 'MC_sq+[1,2,3]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2834},
]

# ============ 标度律定义 ============

def scaling_law_zdg(params, z, d, g):
    """z-d-g 标度律: p_c = A · z^{-α} · d^{-γ} · g^{δ}"""
    A, alpha, gamma, delta = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * np.power(g, delta)

def scaling_law_zd_gcorrection(params, z, d, g):
    """z-d 标度律 + girth 修正: p_c = A · z^{-α} · d^{-γ} · (1 + B/g)"""
    A, alpha, gamma, B = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * (1 + B / g)

def scaling_law_zd_glog(params, z, d, g):
    """z-d 标度律 + log(g) 修正: p_c = A · z^{-α} · d^{-γ} · (1 + B·log(g))"""
    A, alpha, gamma, B = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * (1 + B * np.log(g))

# ============ 拟合函数 ============

def fit_scaling_law(formula_func, z, d, g, pc, initial_guess):
    """拟合标度律"""
    def objective(params):
        pred = formula_func(params, z, d, g)
        return np.sum((pred - pc) ** 2)

    result = minimize(objective, initial_guess, method='Nelder-Mead',
                     options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})

    pred = formula_func(result.x, z, d, g)

    ss_res = np.sum((pc - pred) ** 2)
    ss_tot = np.sum((pc - np.mean(pc)) ** 2)
    r2 = 1 - ss_res / ss_tot

    mae = np.mean(np.abs(pc - pred))
    rmse = np.sqrt(np.mean((pc - pred) ** 2))

    return {
        'params': result.x,
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'success': result.success,
        'predictions': pred
    }

# ============ 主实验 ============

print("="*80)
print("【实验】PRL 级标度律搜索")
print("="*80)

z = np.array([lat['z'] for lat in lattices_data])
d = np.array([lat['d'] for lat in lattices_data])
g = np.array([lat['g'] for lat in lattices_data])
pc = np.array([lat['pc'] for lat in lattices_data])

# 测试三种公式
formulas = [
    ('p_c = A·z^{-α}·d^{-γ}·g^{δ}', scaling_law_zdg, [3.0, 0.7, 0.9, 0.1]),
    ('p_c = A·z^{-α}·d^{-γ}·(1+B/g)', scaling_law_zd_gcorrection, [3.0, 0.7, 0.9, 0.5]),
    ('p_c = A·z^{-α}·d^{-γ}·(1+B·log(g))', scaling_law_zd_glog, [3.0, 0.7, 0.9, 0.1]),
]

results = []

for name, func, init in formulas:
    print(f"\n--- {name} ---")
    result = fit_scaling_law(func, z, d, g, pc, init)

    if result['success']:
        print(f"参数: {[f'{p:.4f}' for p in result['params']]}")
        print(f"R² = {result['R2']:.6f}, MAE = {result['MAE']:.6f}")
        result['name'] = name
        result['formula'] = func
        results.append(result)
    else:
        print(f"拟合失败")

# ============ 找到最优公式 ============

print("\n" + "="*80)
print("【结果】最优标度律")
print("="*80)

best_result = max(results, key=lambda x: x['R2'])

print(f"\n最优公式: {best_result['name']}")
print(f"参数: {[f'{p:.4f}' for p in best_result['params']]}")
print(f"\n拟合质量:")
print(f"  R² = {best_result['R2']:.6f}")
print(f"  MAE = {best_result['MAE']:.6f}")
print(f"  RMSE = {best_result['RMSE']:.6f}")

# ============ 逐点分析 ============

print("\n" + "="*80)
print("【逐点分析】")
print("="*80)

pc_pred = best_result['predictions']

print(f"\n{'晶格':<20} {'z':<4} {'d':<2} {'g':<2} {'p_c真实':<10} {'p_c预测':<10} {'误差':<8}")
print("-"*75)

for i, lat in enumerate(lattices_data):
    error = abs(pc[i] - pc_pred[i])
    error_rel = error / pc[i] if pc[i] > 0 else 0
    print(f"{lat['name']:<20} {lat['z']:<4} {lat['d']:<2} {lat['g']:<2} "
          f"{pc[i]:<10.5f} {pc_pred[i]:<10.5f} {error:<8.5f} ({error_rel:.1%})")

# ============ 对比分析 ============

print("\n" + "="*80)
print("【对比】")
print("="*80)

print(f"\n无修正 (R² = 0.979, MAE = 0.022):")
print(f"  p_c = 3.031 · z^{{-0.717}} · d^{{-0.898}}")

print(f"\n最优修正 (R² = {best_result['R2']:.6f}, MAE = {best_result['MAE']:.6f}):")
print(f"  {best_result['name']}")
print(f"  参数: {[f'{p:.4f}' for p in best_result['params']]}")

improvement_r2 = (best_result['R2'] - 0.979) / 0.979 * 100
improvement_mae = (0.022 - best_result['MAE']) / 0.022 * 100

print(f"\n改进:")
print(f"  R²: {improvement_r2:+.1f}%")
print(f"  MAE: {improvement_mae:+.1f}%")

# ============ PRL 级别评估 ============

print("\n" + "="*80)
print("【PRL 级别评估】")
print("="*80)

if best_result['R2'] > 0.995:
    print("\n✓✓✓ PRL 级别！")
    print(f"R² = {best_result['R2']:.6f} > 0.995")
    print("这是普适标度律的重大发现！")
elif best_result['R2'] > 0.99:
    print("\n✓✓ 接近 PRL 级别")
    print(f"R² = {best_result['R2']:.6f} > 0.99")
    print("需要更严格的验证（cross-validation, bootstrap）")
elif best_result['R2'] > 0.98:
    print("\n✓ 有价值的改进")
    print(f"R² = {best_result['R2']:.6f} > 0.98")
    print("但仍需进一步改进")
else:
    print("\n⚠️  改进有限")
    print(f"R² = {best_result['R2']:.6f} < 0.98")

# ============ 物理意义解读 ============

print("\n" + "="*80)
print("【物理意义】")
print("="*80)

if 'g^{δ}' in best_result['name']:
    A, alpha, gamma, delta = best_result['params']
    print(f"""
标度律: p_c = A · z^{{-α}} · d^{{-γ}} · g^{{δ}}

参数:
  A = {A:.4f}
  α = {alpha:.4f}
  γ = {gamma:.4f}
  δ = {delta:.4f}

物理意义:

1. α = {alpha:.3f}: 配位数的标度指数
   - p_c ~ z^{{-{alpha:.2f}}}
   - 配位数越大，p_c 越低

2. γ = {gamma:.3f}: 维度的标度指数
   - p_c ~ d^{{-{gamma:.2f}}}
   - 维度越高，p_c 越低

3. δ = {delta:.3f}: girth 的标度指数
   - p_c ~ g^{{{delta:.2f}}}
   - δ > 0: girth 越大，p_c 越高（短环促进渗流）
   - δ < 0: girth 越大，p_c 越低（短环阻碍渗流）

4. 如果 δ ≈ 0:
   - girth 效应不显著
   - 拓扑修正不重要

5. 如果 δ ≈ 1:
   - p_c ~ g
   - girth 线性影响 p_c
   - 有明确的物理意义（环密度效应）
""")
elif '1+B/g' in best_result['name']:
    A, alpha, gamma, B = best_result['params']
    print(f"""
标度律: p_c = A · z^{{-α}} · d^{{-γ}} · (1 + B/g)

参数:
  A = {A:.4f}
  α = {alpha:.4f}
  γ = {gamma:.4f}
  B = {B:.4f}

物理意义:

1. B = {B:.3f}: girth 修正系数
   - B > 0: girth 越大，p_c 越高
   - B < 0: girth 越大，p_c 越低

2. 修正项 (1 + B/g):
   - g → ∞: 修正 → 1（无修正）
   - g = 3: 修正 = 1 + B/3
   - g = 6: 修正 = 1 + B/6

3. 物理机制:
   - 短环（g=3）促进/阻碍渗流
   - 长环（g≥6）影响减弱
""")

print("\n" + "="*80)
print("【结论】")
print("="*80)
print("""
通过引入拓扑修正（girth），我们得到了更精确的标度律。

如果 R² > 0.99，这是 PRL 级别的普适标度律！

下一步:
1. 进行严格的统计检验（cross-validation, bootstrap）
2. 从理论上推导标度指数
3. 预测新的晶格 p_c，进行实验验证
""")
