from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
第2轮迭代：模拟数据反馈改进公式
================================

流程：
1. 将第1轮的模拟数据加入训练集
2. 重新拟合3个物理公式
3. 预测2个新晶格（目标C、D）
4. 用蒙特卡洛模拟验证
5. 对比第1轮 vs 第2轮的改进

新预测目标：
- 目标C: 正方+第3配位区 (z=12, g=4, 无三角形)
- 目标D: 正方+第1+2+3区 (z=12, g=3, 有三角形)
"""

import numpy as np
import sys
sys.path.insert(0, ROOT)

from scipy.optimize import minimize
from monte_carlo_simulator import finite_size_scaling

# ============ 扩展训练集 ============

# 原始52个数据点（简化版，只保留关键物理量）
original_data = [
    # 标准晶格
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'n_tri': 0, 'pc': 0.697040},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'n_tri': 0, 'pc': 0.592746},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'n_tri': 6, 'pc': 0.500000},
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'n_tri': 2, 'pc': 0.652703},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'n_tri': 0, 'pc': 0.311600},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'n_tri': 0, 'pc': 0.245800},
    {'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'n_tri': 8, 'pc': 0.118800},
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'n_tri': 0, 'pc': 0.431900},
    # 高维
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'n_tri': 0, 'pc': 0.196737},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'n_tri': 0, 'pc': 0.140810},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'n_tri': 0, 'pc': 0.109460},
    # 部分复杂邻域（从52个点中选取代表性的）
    {'name': 'sq_2', 'z': 8, 'd': 2, 'g': 3, 'n_tri': 4, 'pc': 0.2310},  # 估算值
    {'name': 'sq_3', 'z': 12, 'd': 2, 'g': 4, 'n_tri': 0, 'pc': 0.1850},  # 估算值
    {'name': 'sq_4', 'z': 12, 'd': 2, 'g': 3, 'n_tri': 8, 'pc': 0.1650},  # 估算值
    {'name': 'sq_5', 'z': 20, 'd': 2, 'g': 3, 'n_tri': 12, 'pc': 0.1250},  # 估算值
]

# 加入第1轮的模拟数据
simulated_data = [
    {'name': 'sq_12_sim', 'z': 8, 'd': 2, 'g': 3, 'n_tri': 4, 'pc': 0.4046,
     'source': 'monte_carlo_round1'},
    {'name': 'sq_14_sim', 'z': 16, 'd': 2, 'g': 3, 'n_tri': 12, 'pc': 0.2338,
     'source': 'monte_carlo_round1'},
]

# 扩展训练集
extended_data = original_data + simulated_data

print(f"扩展训练集: {len(extended_data)} 个数据点")
print(f"  原始数据: {len(original_data)}")
print(f"  模拟数据: {len(simulated_data)}")

# ============ 计算派生物理量 ============

def compute_derived_quantities(data):
    """计算 z_eff, C 等"""
    for item in data:
        z = item['z']
        n_tri = item['n_tri']
        item['z_eff'] = z - 2 * n_tri / z if z > 0 else z
        item['C'] = 2 * n_tri / (z * (z - 1)) if z > 1 else 0
    return data

extended_data = compute_derived_quantities(extended_data)

# ============ 公式定义 ============

def formula_comprehensive(params, z, d, g, C):
    """综合公式"""
    A, alpha, beta, B, D = params
    return A / (z**alpha * d**beta) * (1 + B * C + D / g)

def formula_basic_powerlaw(params, z, d):
    """基础幂律"""
    A, alpha, beta = params
    return A / (z**alpha * d**beta)

def formula_effective_z(params, z_eff, d):
    """有效配位数"""
    A, alpha, beta = params
    return A / ((z_eff - 1)**alpha * d**beta)

# ============ 拟合函数 ============

def fit_formula(formula_func, data, param_names, initial_guess):
    """拟合公式参数"""
    def objective(params):
        error = 0
        for item in data:
            if formula_func == formula_comprehensive:
                pred = formula_func(params, item['z'], item['d'], item['g'], item['C'])
            elif formula_func == formula_basic_powerlaw:
                pred = formula_func(params, item['z'], item['d'])
            elif formula_func == formula_effective_z:
                pred = formula_func(params, item['z_eff'], item['d'])
            else:
                raise ValueError(f"Unknown formula: {formula_func}")

            error += (pred - item['pc'])**2
        return np.sqrt(error / len(data))

    result = minimize(objective, initial_guess, method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})

    return result.x

# ============ 重新拟合 ============

print("\n" + "="*80)
print("【第2轮迭代】重新拟合公式（加入模拟数据）")
print("="*80)

# 综合公式
print("\n--- 综合公式 ---")
params_comp_init = np.array([2.01, 1.02, 0.70, -1.60, 4.89])
params_comp_new = fit_formula(formula_comprehensive, extended_data,
                               ['A', 'alpha', 'beta', 'B', 'D'], params_comp_init)
print(f"旧参数: {[f'{p:.2f}' for p in params_comp_init]}")
print(f"新参数: {[f'{p:.2f}' for p in params_comp_new]}")

# 基础幂律
print("\n--- 基础幂律 ---")
params_basic_init = np.array([3.96, 0.90, 0.83])
params_basic_new = fit_formula(formula_basic_powerlaw, extended_data,
                                ['A', 'alpha', 'beta'], params_basic_init)
print(f"旧参数: {[f'{p:.2f}' for p in params_basic_init]}")
print(f"新参数: {[f'{p:.2f}' for p in params_basic_new]}")

# 有效配位数
print("\n--- 有效配位数 ---")
params_eff_init = np.array([1.73, 0.69, 0.61])
params_eff_new = fit_formula(formula_effective_z, extended_data,
                              ['A', 'alpha', 'beta'], params_eff_init)
print(f"旧参数: {[f'{p:.2f}' for p in params_eff_init]}")
print(f"新参数: {[f'{p:.2f}' for p in params_eff_new]}")

# ============ 第2轮预测目标 ============

round2_targets = [
    {
        'id': 'C',
        'name': '正方+第3配位区',
        'z': 12,
        'd': 2,
        'g': 4,  # 只有(±2,0),(0,±2)，最短环是4
        'n_tri': 0,  # 无三角形
        'neighbor_zones': [1, 3],  # 第1区+第3区
        'description': '无三角形的高配位',
        'p_guess': 0.20
    },
    {
        'id': 'D',
        'name': '正方+第1+2+3区',
        'z': 12,
        'd': 2,
        'g': 3,  # 包含对角线，最短环是3
        'n_tri': 8,  # 估算：第2区引入4个，第3区引入4个
        'neighbor_zones': [1, 2, 3],  # 第1+2+3区
        'description': '复杂组合（高z+高C）',
        'p_guess': 0.18
    },
]

# ============ 第2轮预测 ============

print("\n" + "="*80)
print("【第2轮迭代】预测新晶格")
print("="*80)

def make_predictions_round2(target, params_comp, params_basic, params_eff):
    """用更新后的公式预测"""
    z = target['z']
    d = target['d']
    g = target['g']
    n_tri = target['n_tri']

    z_eff = z - 2 * n_tri / z if z > 0 else z
    C = 2 * n_tri / (z * (z - 1)) if z > 1 else 0

    predictions = {
        'comprehensive': formula_comprehensive(params_comp, z, d, g, C),
        'basic_powerlaw': formula_basic_powerlaw(params_basic, z, d),
        'effective_z': formula_effective_z(params_eff, z_eff, d)
    }

    return predictions, z_eff, C

for target in round2_targets:
    print(f"\n--- 目标 {target['id']}: {target['name']} ---")
    print(f"物理量: z={target['z']}, d={target['d']}, g={target['g']}, n_tri={target['n_tri']}")

    # 第1轮预测（用旧参数）
    pred_r1, _, _ = make_predictions_round2(target, params_comp_init, params_basic_init, params_eff_init)

    # 第2轮预测（用新参数）
    pred_r2, z_eff, C = make_predictions_round2(target, params_comp_new, params_basic_new, params_eff_new)

    print(f"\n第1轮预测（旧公式）:")
    for name, val in pred_r1.items():
        print(f"  {name}: p_c = {val:.4f}")

    print(f"\n第2轮预测（新公式）:")
    for name, val in pred_r2.items():
        print(f"  {name}: p_c = {val:.4f}")

    print(f"\n预测变化:")
    for name in pred_r1.keys():
        delta = pred_r2[name] - pred_r1[name]
        print(f"  {name}: {pred_r1[name]:.4f} → {pred_r2[name]:.4f} (Δ={delta:+.4f})")

    target['predictions_r2'] = pred_r2
    target['z_eff'] = z_eff
    target['C'] = C

# ============ 第2轮模拟验证 ============

print("\n" + "="*80)
print("【第2轮迭代】蒙特卡洛模拟验证")
print("="*80)

for target in round2_targets:
    print(f"\n--- 目标 {target['id']}: {target['name']} ---")

    L_values = [30, 40, 50, 60]
    p_c_sim, error = finite_size_scaling(
        '2d_square', L_values,
        neighbor_zones=target['neighbor_zones'],
        p_guess=target['p_guess'],
        n_samples=80
    )

    print(f"\n模拟结果: p_c = {p_c_sim:.4f} ± {error:.4f}")

    # 对比第2轮预测
    print(f"\n第2轮预测 vs 模拟:")
    for name, p_pred in target['predictions_r2'].items():
        error_abs = abs(p_pred - p_c_sim)
        error_rel = error_abs / p_c_sim if p_c_sim > 0 else 0
        print(f"  {name:20s}: 预测={p_pred:.4f}, 误差={error_abs:.4f} ({error_rel:.1%})")

    target['p_c_sim'] = p_c_sim
    target['sim_error'] = error

# ============ 对比第1轮 vs 第2轮 ============

print("\n" + "="*80)
print("【对比】第1轮 vs 第2轮")
print("="*80)

# 第1轮的结果（从之前的实验）
round1_results = {
    'A': {
        'name': '正方+第2配位区',
        'p_c_sim': 0.4046,
        'predictions': {'comprehensive': 0.3563, 'basic_powerlaw': 0.3428, 'effective_z': 0.3292}
    },
    'B': {
        'name': '正方+第4配位区',
        'p_c_sim': 0.2338,
        'predictions': {'comprehensive': 0.1807, 'basic_powerlaw': 0.1837, 'effective_z': 0.1881}
    }
}

print("\n--- 第1轮结果（旧公式） ---")
for target_id, result in round1_results.items():
    print(f"\n目标 {target_id}: {result['name']}")
    print(f"  模拟 p_c = {result['p_c_sim']:.4f}")
    for name, p_pred in result['predictions'].items():
        error = abs(p_pred - result['p_c_sim'])
        print(f"  {name:20s}: 预测={p_pred:.4f}, 误差={error:.4f} ({error/result['p_c_sim']:.1%})")

print("\n--- 第2轮结果（新公式） ---")
for target in round2_targets:
    print(f"\n目标 {target['id']}: {target['name']}")
    print(f"  模拟 p_c = {target['p_c_sim']:.4f}")
    for name, p_pred in target['predictions_r2'].items():
        error = abs(p_pred - target['p_c_sim'])
        print(f"  {name:20s}: 预测={p_pred:.4f}, 误差={error:.4f} ({error/target['p_c_sim']:.1%})")

# ============ 改进分析 ============

print("\n" + "="*80)
print("【改进分析】")
print("="*80)

print("\n第1轮平均误差:")
errors_r1 = []
for result in round1_results.values():
    for p_pred in result['predictions'].values():
        errors_r1.append(abs(p_pred - result['p_c_sim']) / result['p_c_sim'])
print(f"  平均相对误差: {np.mean(errors_r1):.1%}")

print("\n第2轮平均误差:")
errors_r2 = []
for target in round2_targets:
    for p_pred in target['predictions_r2'].values():
        errors_r2.append(abs(p_pred - target['p_c_sim']) / target['p_c_sim'])
print(f"  平均相对误差: {np.mean(errors_r2):.1%}")

improvement = (np.mean(errors_r1) - np.mean(errors_r2)) / np.mean(errors_r1) * 100
print(f"\n改进幅度: {improvement:.1f}%")

if improvement > 0:
    print("✓ 第2轮公式更准确")
else:
    print("✗ 第2轮公式没有改进")

print("\n" + "="*80)
print("【结论】")
print("="*80)
print("1. 加入模拟数据后，公式参数发生了变化")
print("2. 新公式在目标C、D上的预测需要验证")
print("3. 对比第1轮vs第2轮，评估改进效果")
print("4. 如果改进显著，说明迭代反馈是有效的")
