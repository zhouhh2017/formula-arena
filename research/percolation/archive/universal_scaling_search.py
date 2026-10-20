"""
普适标度律搜索：从拟合到推导
==============================

目标：找到 p_c = f(z_eff) 的精确标度形式

核心假说：
  p_c = A · z_eff(β)^{-α}

其中：
  z_eff(β) = Σ_i count_i / dist_i^β

如果能找到精确的 (α, β)，就是 PRL 级别的发现。

实验计划：
1. 构建完整的晶格数据库（标准 + 复杂邻域 + 模拟数据）
2. 计算每个晶格的 z_eff(β)（β 从 0 到 2 扫描）
3. 对每个 β，拟合 p_c = A · z_eff^{-α}
4. 找到最优的 β，使得 R² 最大
5. 如果 R² > 0.99，进行严格的统计检验
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# ============ 完整晶格数据库 ============

# 标准晶格（来自文献）
standard_lattices = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'pc': 0.697040,
     'zones': [], 'source': 'Stauffer1994'},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'pc': 0.592746,
     'zones': [1], 'source': 'Stauffer1994'},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'pc': 0.500000,
     'zones': [1, 2], 'source': 'Stauffer1994'},  # 6近邻 = 区1+区2
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'pc': 0.652703,
     'zones': [1], 'source': 'Stauffer1994'},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'pc': 0.311600,
     'zones': [1], 'source': 'Stauffer1994'},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'pc': 0.245800,
     'zones': [1], 'source': 'Stauffer1994'},
    {'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'pc': 0.118800,
     'zones': [1, 2], 'source': 'Stauffer1994'},  # 12近邻 = 区1+区2
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'pc': 0.431900,
     'zones': [], 'source': 'Stauffer1994'},
    # 高维
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'pc': 0.196737,
     'zones': [1], 'source': 'literature'},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'pc': 0.140810,
     'zones': [1], 'source': 'literature'},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'pc': 0.109460,
     'zones': [1], 'source': 'literature'},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'g': 4, 'pc': 0.089270,
     'zones': [1], 'source': 'literature'},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'g': 4, 'pc': 0.075050,
     'zones': [1], 'source': 'literature'},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'g': 4, 'pc': 0.064530,
     'zones': [1], 'source': 'literature'},
]

# 蒙特卡洛模拟数据（2D 正方复杂邻域）
mc_lattices = [
    {'name': 'MC_sq', 'z': 4, 'd': 2, 'g': 4, 'pc': 0.592746,
     'zones': [1], 'source': 'MC'},  # 标准正方
    {'name': 'MC_sq+[1,2]', 'z': 8, 'd': 2, 'g': 3, 'pc': 0.4046,
     'zones': [1, 2], 'source': 'MC_round1'},
    {'name': 'MC_sq+[1,4]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2338,
     'zones': [1, 4], 'source': 'MC_round1'},
    {'name': 'MC_sq+[1,3]', 'z': 8, 'd': 2, 'g': 4, 'pc': 0.3399,
     'zones': [1, 3], 'source': 'MC_round2'},
    {'name': 'MC_sq+[1,2,3]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2834,
     'zones': [1, 2, 3], 'source': 'MC_round2'},
]

# 合并所有数据
all_lattices = standard_lattices + mc_lattices

print(f"晶格数据库: {len(all_lattices)} 个数据点")
print(f"  标准晶格: {len(standard_lattices)}")
print(f"  模拟数据: {len(mc_lattices)}")

# ============ 连接距离定义 ============

# 2D 正方晶格的配位区距离
zone_distances_2d = {
    1: 1.0,           # (±1,0), (0,±1)
    2: np.sqrt(2),    # (±1,±1)
    3: 2.0,           # (±2,0), (0,±2)
    4: np.sqrt(5),    # (±2,±1), (±1,±2)
    5: np.sqrt(8),    # (±2,±2)
}

# 3D 简单立方的配位区距离
zone_distances_3d = {
    1: 1.0,           # (±1,0,0), (0,±1,0), (0,0,±1) → 6近邻
    2: np.sqrt(2),    # (±1,±1,0) → 12次近邻
    3: np.sqrt(3),    # (±1,±1,±1) → 8第三近邻
}

# 每个配位区的邻居数
zone_counts_2d = {1: 4, 2: 4, 3: 4, 4: 8, 5: 4}
zone_counts_3d = {1: 6, 2: 12, 3: 8}

# ============ 计算 z_eff ============

def compute_z_eff(lattice, beta):
    """计算有效配位数 z_eff(β) = Σ count_i / dist_i^β"""
    z_eff = 0.0

    if lattice['d'] == 2:
        zone_distances = zone_distances_2d
        zone_counts = zone_counts_2d
    elif lattice['d'] == 3:
        zone_distances = zone_distances_3d
        zone_counts = zone_counts_3d
    else:
        # 高维：假设只有区1（最近邻）
        zone_distances = {1: 1.0}
        zone_counts = {1: lattice['z']}

    # 如果 zones 为空，使用 z 推断
    if not lattice['zones']:
        # 对于没有明确 zones 的晶格，假设所有邻居都是区1
        if lattice['d'] == 2:
            z_eff = lattice['z'] / (1.0 ** beta)
        elif lattice['d'] == 3:
            z_eff = lattice['z'] / (1.0 ** beta)
        else:
            z_eff = lattice['z'] / (1.0 ** beta)
    else:
        for zone in lattice['zones']:
            if zone in zone_distances and zone in zone_counts:
                dist = zone_distances[zone]
                count = zone_counts[zone]
                z_eff += count / (dist ** beta)

    return z_eff

# ============ 标度律拟合 ============

def scaling_law(z_eff, A, alpha):
    """标度律 p_c = A · z_eff^{-α}"""
    return A * np.power(z_eff, -alpha)

def fit_scaling_law(z_eff_array, pc_array):
    """拟合标度律参数 (A, α)"""
    try:
        popt, pcov = curve_fit(scaling_law, z_eff_array, pc_array,
                               p0=[1.0, 1.0], maxfev=10000)
        A, alpha = popt

        # 计算拟合质量
        pc_pred = scaling_law(z_eff_array, A, alpha)
        ss_res = np.sum((pc_array - pc_pred) ** 2)
        ss_tot = np.sum((pc_array - np.mean(pc_array)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # 计算误差
        mae = np.mean(np.abs(pc_array - pc_pred))
        rmse = np.sqrt(np.mean((pc_array - pc_pred) ** 2))

        return {
            'A': A,
            'alpha': alpha,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============ 主实验：扫描 β 寻找最优标度律 ============

print("\n" + "="*80)
print("【实验】扫描 β 寻找最优标度律")
print("="*80)

beta_values = np.linspace(0.0, 2.0, 41)  # β 从 0 到 2，41 个点

results = []

print(f"\n{'β':<6} {'z_eff范围':<20} {'A':<8} {'α':<8} {'R²':<10} {'MAE':<10}")
print("-"*70)

for beta in beta_values:
    # 计算所有晶格的 z_eff(β)
    z_eff_array = np.array([compute_z_eff(lat, beta) for lat in all_lattices])
    pc_array = np.array([lat['pc'] for lat in all_lattices])

    # 拟合标度律
    result = fit_scaling_law(z_eff_array, pc_array)

    if result['success']:
        result['beta'] = beta
        result['z_eff_min'] = z_eff_array.min()
        result['z_eff_max'] = z_eff_array.max()
        results.append(result)

        print(f"{beta:<6.2f} [{result['z_eff_min']:.2f}, {result['z_eff_max']:.2f}]  "
              f"{result['A']:<8.3f} {result['alpha']:<8.3f} {result['R2']:<10.6f} {result['MAE']:<10.6f}")
    else:
        print(f"{beta:<6.2f} 拟合失败: {result['error']}")

# ============ 找到最优 β ============

print("\n" + "="*80)
print("【结果】最优标度律")
print("="*80)

best_result = max(results, key=lambda x: x['R2'])
best_beta = best_result['beta']

print(f"\n最优 β = {best_beta:.2f}")
print(f"标度律参数:")
print(f"  A = {best_result['A']:.4f}")
print(f"  α = {best_result['alpha']:.4f}")
print(f"\n拟合质量:")
print(f"  R² = {best_result['R2']:.6f}")
print(f"  MAE = {best_result['MAE']:.6f}")
print(f"  RMSE = {best_result['RMSE']:.6f}")

# ============ 逐点分析 ============

print("\n" + "="*80)
print("【逐点分析】预测 vs 真实")
print("="*80)

z_eff_array = np.array([compute_z_eff(lat, best_beta) for lat in all_lattices])
pc_array = np.array([lat['pc'] for lat in all_lattices])
pc_pred = scaling_law(z_eff_array, best_result['A'], best_result['alpha'])

print(f"\n{'晶格':<25} {'z':<4} {'d':<2} {'z_eff':<8} {'p_c真实':<10} {'p_c预测':<10} {'误差':<8}")
print("-"*80)

for i, lat in enumerate(all_lattices):
    error = abs(pc_array[i] - pc_pred[i])
    error_rel = error / pc_array[i] if pc_array[i] > 0 else 0
    print(f"{lat['name']:<25} {lat['z']:<4} {lat['d']:<2} {z_eff_array[i]:<8.2f} "
          f"{pc_array[i]:<10.5f} {pc_pred[i]:<10.5f} {error:<8.5f} ({error_rel:.1%})")

# ============ 统计检验 ============

print("\n" + "="*80)
print("【统计检验】")
print("="*80)

# Pearson 相关
r_pearson, p_pearson = pearsonr(z_eff_array, pc_array)
print(f"\nPearson 相关: r = {r_pearson:.4f}, p = {p_pearson:.6f}")

# Spearman 相关
r_spearman, p_spearman = spearmanr(z_eff_array, pc_array)
print(f"Spearman 相关: ρ = {r_spearman:.4f}, p = {p_spearman:.6f}")

# 残差分析
residuals = pc_array - pc_pred
print(f"\n残差统计:")
print(f"  均值: {np.mean(residuals):.6f}")
print(f"  标准差: {np.std(residuals):.6f}")
print(f"  最大正值: {np.max(residuals):.6f}")
print(f"  最大负值: {np.min(residuals):.6f}")

# ============ 与之前的公式对比 ============

print("\n" + "="*80)
print("【对比】标度律 vs 之前的公式")
print("="*80)

print(f"\n之前的公式 (52个数据点):")
print(f"  基础幂律: p_c = 3.96 / (z^0.90 · d^0.83)")
print(f"    LOOCV MAE ≈ 0.034")
print(f"\n标度律 ({len(all_lattices)}个数据点):")
print(f"  p_c = {best_result['A']:.3f} · z_eff({best_beta:.2f})^{-best_result['alpha']:.3f}")
print(f"    MAE = {best_result['MAE']:.4f}")

if best_result['MAE'] < 0.034:
    improvement = (0.034 - best_result['MAE']) / 0.034 * 100
    print(f"\n✓ 改进: {improvement:.1f}%")
else:
    degradation = (best_result['MAE'] - 0.034) / 0.034 * 100
    print(f"\n✗ 退化: {degradation:.1f}%")

# ============ 物理意义解读 ============

print("\n" + "="*80)
print("【物理意义】")
print("="*80)

print(f"""
标度律: p_c = A · z_eff(β)^{{-α}}

其中:
  z_eff(β) = Σ count_i / dist_i^β
  A = {best_result['A']:.4f}
  α = {best_result['alpha']:.4f}
  β = {best_beta:.2f}

物理意义:

1. β 的物理意义:
   - β = 0: z_eff = z（不考虑距离）
   - β > 0: 远距离连接权重降低
   - β < 0: 远距离连接权重增加（反直觉）

   当前最优 β = {best_beta:.2f}
   → 距离越远的连接，贡献{'越低' if best_beta > 0 else '越高'}

2. α 的物理意义:
   - 标度指数，描述 p_c 对 z_eff 的敏感性
   - α > 1: p_c 对 z_eff 高度敏感
   - α < 1: p_c 对 z_eff 不敏感

   当前 α = {best_result['alpha']:.4f}
   → p_c ~ z_eff^{{-{best_result['alpha']:.2f}}}

3. 如果 β ≈ 1:
   - z_eff ~ 1/dist
   - 类似"有效电导"或"有效耦合"
   - 有明确的物理意义（类似紧束缚模型）

4. 如果 β ≈ 2:
   - z_eff ~ 1/dist²
   - 类似偶极-偶极相互作用
   - 长程关联的标度

5. 如果 β < 0（反直觉）:
   - 远距离连接更重要
   - 可能揭示新的物理机制
   - 需要理论解释
""")

# ============ 结论 ============

print("\n" + "="*80)
print("【结论】")
print("="*80)

if best_result['R2'] > 0.99:
    print("\n✓✓✓ 重大发现！")
    print(f"标度律 R² = {best_result['R2']:.6f} > 0.99")
    print("这可能是 PRL 级别的发现！")
    print("\n下一步:")
    print("1. 进行更严格的统计检验（bootstrap, cross-validation）")
    print("2. 从理论推导 α 和 β 的值")
    print("3. 预测新的晶格 p_c，进行实验验证")
elif best_result['R2'] > 0.95:
    print("\n✓ 重要发现")
    print(f"标度律 R² = {best_result['R2']:.6f} > 0.95")
    print("有潜力达到 PRL 级别，但需要进一步改进")
    print("\n下一步:")
    print("1. 引入更多物理量（维度修正、聚类修正）")
    print("2. 分段建模（2D vs 3D）")
else:
    print("\n⚠️  标度律不够精确")
    print(f"标度律 R² = {best_result['R2']:.6f} < 0.95")
    print("需要更复杂的模型")
    print("\n下一步:")
    print("1. 考虑多维标度律 p_c = f(z_eff, d, g, ...)")
    print("2. 放弃单一标度律，采用分段模型")
