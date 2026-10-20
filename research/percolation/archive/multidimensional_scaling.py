"""
多维标度律搜索：引入维度修正
================================

问题：单一标度律 p_c = A · z_eff^{-α} 的 R² = 0.869，不够精确

观察：高维晶格被系统性高估（误差 26-30%），说明需要维度修正

新假说：
  p_c = A · z_eff(β)^{-α} · d^{-γ}

实验计划：
1. 扫描 (α, β, γ) 参数空间
2. 找到最优参数，使得 R² 最大
3. 如果 R² > 0.95，可能是 PRL 级别
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr
import itertools

# ============ 数据准备 ============

# 晶格数据（简化版）
lattices_data = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'zones': [], 'pc': 0.697040},
    {'name': 'square', 'z': 4, 'd': 2, 'zones': [1], 'pc': 0.592746},
    {'name': 'triangular', 'z': 6, 'd': 2, 'zones': [1, 2], 'pc': 0.500000},
    {'name': 'kagome', 'z': 4, 'd': 2, 'zones': [1], 'pc': 0.652703},
    {'name': 'sc', 'z': 6, 'd': 3, 'zones': [1], 'pc': 0.311600},
    {'name': 'bcc', 'z': 8, 'd': 3, 'zones': [1], 'pc': 0.245800},
    {'name': 'fcc', 'z': 12, 'd': 3, 'zones': [1, 2], 'pc': 0.118800},
    {'name': 'diamond', 'z': 4, 'd': 3, 'zones': [], 'pc': 0.431900},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'zones': [1], 'pc': 0.196737},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'zones': [1], 'pc': 0.140810},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'zones': [1], 'pc': 0.109460},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'zones': [1], 'pc': 0.089270},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'zones': [1], 'pc': 0.075050},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'zones': [1], 'pc': 0.064530},
    {'name': 'MC_sq', 'z': 4, 'd': 2, 'zones': [1], 'pc': 0.592746},
    {'name': 'MC_sq+[1,2]', 'z': 8, 'd': 2, 'zones': [1, 2], 'pc': 0.4046},
    {'name': 'MC_sq+[1,4]', 'z': 12, 'd': 2, 'zones': [1, 4], 'pc': 0.2338},
    {'name': 'MC_sq+[1,3]', 'z': 8, 'd': 2, 'zones': [1, 3], 'pc': 0.3399},
    {'name': 'MC_sq+[1,2,3]', 'z': 12, 'd': 2, 'zones': [1, 2, 3], 'pc': 0.2834},
]

# 配位区距离
zone_distances = {
    1: 1.0, 2: np.sqrt(2), 3: 2.0, 4: np.sqrt(5), 5: np.sqrt(8)
}
zone_counts = {1: 4, 2: 4, 3: 4, 4: 8, 5: 4}

def compute_z_eff(lattice, beta):
    """计算 z_eff(β)"""
    z_eff = 0.0
    if not lattice['zones']:
        z_eff = lattice['z'] / (1.0 ** beta)
    else:
        for zone in lattice['zones']:
            if zone in zone_distances:
                dist = zone_distances[zone]
                count = zone_counts.get(zone, 4)
                z_eff += count / (dist ** beta)
    return z_eff

# ============ 多维标度律定义 ============

def scaling_law_nd(params, z_eff, d):
    """多维标度律: p_c = A · z_eff^{-α} · d^{-γ}"""
    A, alpha, gamma = params
    return A * np.power(z_eff, -alpha) * np.power(d, -gamma)

def fit_scaling_law_nd(z_eff_array, d_array, pc_array):
    """拟合多维标度律"""
    def objective(params):
        pred = scaling_law_nd(params, z_eff_array, d_array)
        return np.sum((pred - pc_array) ** 2)

    result = minimize(objective, [2.5, 1.0, 0.5], method='Nelder-Mead',
                     options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-8})

    A, alpha, gamma = result.x
    pred = scaling_law_nd(result.x, z_eff_array, d_array)

    ss_res = np.sum((pc_array - pred) ** 2)
    ss_tot = np.sum((pc_array - np.mean(pc_array)) ** 2)
    r2 = 1 - ss_res / ss_tot

    mae = np.mean(np.abs(pc_array - pred))

    return {
        'A': A, 'alpha': alpha, 'gamma': gamma,
        'R2': r2, 'MAE': mae, 'success': result.success
    }

# ============ 主实验：扫描 β，拟合 (α, γ) ============

print("="*80)
print("【实验】多维标度律搜索")
print("="*80)

beta_values = np.linspace(0.0, 2.5, 51)

results = []

print(f"\n{'β':<6} {'A':<8} {'α':<8} {'γ':<8} {'R²':<10} {'MAE':<10}")
print("-"*60)

for beta in beta_values:
    # 计算 z_eff
    z_eff_array = np.array([compute_z_eff(lat, beta) for lat in lattices_data])
    d_array = np.array([lat['d'] for lat in lattices_data])
    pc_array = np.array([lat['pc'] for lat in lattices_data])

    # 拟合多维标度律
    result = fit_scaling_law_nd(z_eff_array, d_array, pc_array)

    if result['success']:
        result['beta'] = beta
        results.append(result)
        print(f"{beta:<6.2f} {result['A']:<8.3f} {result['alpha']:<8.3f} {result['gamma']:<8.3f} "
              f"{result['R2']:<10.6f} {result['MAE']:<10.6f}")

# ============ 找到最优参数 ============

print("\n" + "="*80)
print("【结果】最优多维标度律")
print("="*80)

best_result = max(results, key=lambda x: x['R2'])
best_beta = best_result['beta']

print(f"\n最优参数:")
print(f"  β = {best_beta:.2f}")
print(f"  A = {best_result['A']:.4f}")
print(f"  α = {best_result['alpha']:.4f}")
print(f"  γ = {best_result['gamma']:.4f}")

print(f"\n拟合质量:")
print(f"  R² = {best_result['R2']:.6f}")
print(f"  MAE = {best_result['MAE']:.6f}")

# ============ 逐点分析 ============

print("\n" + "="*80)
print("【逐点分析】")
print("="*80)

z_eff_array = np.array([compute_z_eff(lat, best_beta) for lat in lattices_data])
d_array = np.array([lat['d'] for lat in lattices_data])
pc_array = np.array([lat['pc'] for lat in lattices_data])

params = [best_result['A'], best_result['alpha'], best_result['gamma']]
pc_pred = scaling_law_nd(params, z_eff_array, d_array)

print(f"\n{'晶格':<20} {'z':<4} {'d':<2} {'z_eff':<8} {'p_c真实':<10} {'p_c预测':<10} {'误差':<8}")
print("-"*75)

for i, lat in enumerate(lattices_data):
    error = abs(pc_array[i] - pc_pred[i])
    error_rel = error / pc_array[i] if pc_array[i] > 0 else 0
    print(f"{lat['name']:<20} {lat['z']:<4} {lat['d']:<2} {z_eff_array[i]:<8.2f} "
          f"{pc_array[i]:<10.5f} {pc_pred[i]:<10.5f} {error:<8.5f} ({error_rel:.1%})")

# ============ 对比分析 ============

print("\n" + "="*80)
print("【对比】")
print("="*80)

print(f"\n之前的公式:")
print(f"  单一标度律: p_c = A · z_eff^{-α}")
print(f"    R² = 0.869, MAE = 0.059")

print(f"\n多维标度律:")
print(f"  p_c = {best_result['A']:.3f} · z_eff({best_beta:.2f})^{-best_result['alpha']:.3f} · d^{-best_result['gamma']:.3f}")
print(f"    R² = {best_result['R2']:.6f}, MAE = {best_result['MAE']:.6f}")

improvement_r2 = (best_result['R2'] - 0.869) / 0.869 * 100
improvement_mae = (0.059 - best_result['MAE']) / 0.059 * 100

print(f"\n改进:")
print(f"  R²: {improvement_r2:+.1f}%")
print(f"  MAE: {improvement_mae:+.1f}%")

# ============ 结论 ============

print("\n" + "="*80)
print("【结论】")
print("="*80)

if best_result['R2'] > 0.99:
    print("\n✓✓✓ 重大发现！R² > 0.99")
    print("这是 PRL 级别的普适标度律！")
elif best_result['R2'] > 0.95:
    print("\n✓✓ 重要发现！R² > 0.95")
    print("接近 PRL 级别，但需要更严格的验证")
elif best_result['R2'] > 0.90:
    print("\n✓ 有价值的发现！R² > 0.90")
    print("但仍需改进")
else:
    print("\n⚠️  标度律不够精确")
    print("需要考虑更多物理量（girth, 聚类系数等）")

# ============ 物理意义解读 ============

print("\n" + "="*80)
print("【物理意义】")
print("="*80)

print(f"""
多维标度律:
  p_c = A · z_eff(β)^{{-α}} · d^{{-γ}}

参数:
  A = {best_result['A']:.4f}
  α = {best_result['alpha']:.4f}
  β = {best_beta:.2f}
  γ = {best_result['gamma']:.4f}

物理意义:

1. α = {best_result['alpha']:.3f}: z_eff 的标度指数
   - 描述 p_c 对有效配位数的敏感性
   - α > 1: 高度敏感
   - α < 1: 不敏感

2. β = {best_beta:.2f}: 距离衰减指数
   - β ≈ 0: 不考虑距离
   - β ≈ 1: 类似电导
   - β ≈ 2: 类似偶极相互作用
   - β > 2: 超短程

3. γ = {best_result['gamma']:.3f}: 维度的标度指数
   - 描述 p_c 对维度的依赖性
   - γ > 0: 维度越高，p_c 越低（符合直觉）
   - γ 的物理意义：维度如何影响渗流阈值

4. 如果 β ≈ 2 且 γ ≈ 1:
   - p_c ~ z_eff(2)^{-α} · d^{-1}
   - z_eff ~ 1/dist²（偶极相互作用）
   - 维度效应类似 1/d（平均场）
   - 这可能有深刻的物理意义！
""")
