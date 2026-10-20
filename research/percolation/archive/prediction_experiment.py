"""
预测-验证实验设计
================

策略：选择3个数据集中没有的晶格，用公式预测 p_c，然后蒙特卡洛验证

目标晶格：
1. 【保守】正方晶格+第3配位区 (z=12, d=2, g=4)
   - 类似已知数据的外推
   - 预期：公式间一致性好

2. 【中等】正方晶格+第5配位区 (z=20, d=2, g=3)
   - 中等外推
   - 预期：公式间有分歧，但模拟可行

3. 【激进】3D 体心立方+第2近邻 (z=14, d=3, g=4)
   - 新物理区域
   - 预期：公式间分歧大，但物理意义重要

物理量估算：
- 正方+第3区: z=4+8=12, n_tri=0, n_sq=1+?, g=4
- 正方+第5区: z=4+16=20, n_tri=?, n_sq=?, g=3
- BCC+2NN: z=8+6=14, n_tri=0, n_sq=?, g=4
"""

import numpy as np
from scipy.optimize import minimize

# ============ 物理公式定义 ============

def formula_comprehensive(params, z, d, g, C):
    """综合公式: p_c = A / (z^α · d^β) · (1 + B·C + D/g)"""
    A, alpha, beta, B, D = params
    return A / (z**alpha * d**beta) * (1 + B * C + D / g)

def formula_basic_powerlaw(params, z, d):
    """基础幂律: p_c = A / (z^α · d^β)"""
    A, alpha, beta = params
    return A / (z**alpha * d**beta)

def formula_effective_z(params, z_eff, d):
    """有效配位数公式: p_c = A / ((z_eff-1)^α · d^β)"""
    A, alpha, beta = params
    return A / ((z_eff - 1)**alpha * d**beta)

# ============ 拟合参数（从之前实验中获取）============

# 综合公式参数（从 LIMITATION_SOLUTION_REPORT.md）
params_comprehensive = np.array([2.01, 1.02, 0.70, -1.60, 4.89])

# 基础幂律参数
params_basic = np.array([3.96, 0.90, 0.83])

# 有效配位数公式参数
params_effective = np.array([1.73, 0.69, 0.61])

# ============ 预测目标 ============

prediction_targets = [
    {
        'name': '正方+第3配位区',
        'z': 12,
        'd': 2,
        'g': 4,
        'n_tri': 0,
        'description': '保守预测：中等外推',
        'difficulty': 'easy'
    },
    {
        'name': '正方+第5配位区',
        'z': 20,
        'd': 2,
        'g': 3,
        'n_tri': 10,  # 估算：复杂邻域引入三角形
        'description': '中等预测：较大外推',
        'difficulty': 'medium'
    },
    {
        'name': 'BCC+第2近邻',
        'z': 14,
        'd': 3,
        'g': 4,
        'n_tri': 0,
        'description': '激进预测：3D高配位',
        'difficulty': 'hard'
    }
]

# ============ 预测函数 ============

def make_predictions(target):
    """用3个公式预测同一个晶格"""
    z = target['z']
    d = target['d']
    g = target['g']
    n_tri = target['n_tri']

    # 计算派生物理量
    z_eff = z - 2 * n_tri / z if z > 0 else z
    C = 2 * n_tri / (z * (z - 1)) if z > 1 else 0

    predictions = {}

    # 综合公式
    try:
        pc_comp = formula_comprehensive(params_comprehensive, z, d, g, C)
        predictions['comprehensive'] = pc_comp
    except:
        predictions['comprehensive'] = np.nan

    # 基础幂律
    try:
        pc_basic = formula_basic_powerlaw(params_basic, z, d)
        predictions['basic_powerlaw'] = pc_basic
    except:
        predictions['basic_powerlaw'] = np.nan

    # 有效配位数公式
    try:
        pc_eff = formula_effective(params_effective, z_eff, d)
        predictions['effective_z'] = pc_eff
    except:
        predictions['effective_z'] = np.nan

    return predictions, z_eff, C

# ============ 主程序 ============

def main():
    print("="*80)
    print("预测-验证实验：公式预测 + 蒙特卡洛验证")
    print("="*80)

    print("\n【实验设计】")
    print("选择3个数据集中没有的晶格，用3个公式预测 p_c，然后蒙特卡洛验证")

    print("\n" + "="*80)
    print("【预测阶段】")
    print("="*80)

    results = []

    for target in prediction_targets:
        print(f"\n--- 目标: {target['name']} ---")
        print(f"物理量: z={target['z']}, d={target['d']}, g={target['g']}, n_tri={target['n_tri']}")
        print(f"难度: {target['difficulty']} - {target['description']}")

        predictions, z_eff, C = make_predictions(target)

        print(f"\n预测结果:")
        print(f"  综合公式:      p_c = {predictions['comprehensive']:.4f}")
        print(f"  基础幂律:      p_c = {predictions['basic_powerlaw']:.4f}")
        print(f"  有效配位数:    p_c = {predictions['effective_z']:.4f}")

        # 计算预测一致性
        valid_preds = [v for v in predictions.values() if not np.isnan(v)]
        if len(valid_preds) >= 2:
            mean_pred = np.mean(valid_preds)
            std_pred = np.std(valid_preds)
            consistency = 1 - std_pred / mean_pred if mean_pred > 0 else 0
            print(f"\n  预测一致性: mean={mean_pred:.4f}, std={std_pred:.4f}, 一致性={consistency:.2%}")

        results.append({
            'target': target,
            'predictions': predictions,
            'z_eff': z_eff,
            'C': C
        })

    print("\n" + "="*80)
    print("【下一步】")
    print("="*80)
    print("用蒙特卡洛模拟验证这些预测")
    print("脚本: monte_carlo_verification.py")

    return results

if __name__ == '__main__':
    main()
