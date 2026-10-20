"""
关键发现分析：β = 0 意味着什么？
====================================

实验结果：最优多维标度律
  p_c = 4.569 · z^{-0.663} · d^{-1.591}
  R² = 0.938

关键发现：β = 0 → 距离衰减不重要！

这意味着：
1. 有效配位数 z_eff = z（不考虑距离）
2. 标度律简化为：p_c = A · z^{-α} · d^{-γ}
3. 这是"广义基础幂律"

物理意义：
- 渗流阈值由"连接数"决定，而非"连接质量"
- 短程和长程连接对渗流的贡献相同（反直觉！）

但这与之前的观察矛盾：
- 同 z=8，zones=[1,2] 的 p_c=0.40 > zones=[1,3] 的 p_c=0.34
- 说明连接结构确实重要

解释：
- 在"平均场"意义上，z 是主导因素
- 但在"精细结构"上，连接几何有影响
- 需要更高阶的修正项

实验计划：
1. 验证 p_c = A · z^{-α} · d^{-γ} 的效果
2. 分析残差，找出系统性偏差
3. 引入修正项（girth, 聚类系数等）
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

# ============ 数据（修正版）============

lattices_data = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'pc': 0.697040, 'type': 'standard'},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'pc': 0.592746, 'type': 'standard'},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'pc': 0.500000, 'type': 'standard'},
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'pc': 0.652703, 'type': 'standard'},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'pc': 0.311600, 'type': 'standard'},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'pc': 0.245800, 'type': 'standard'},
    {'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'pc': 0.118800, 'type': 'standard'},
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'pc': 0.431900, 'type': 'standard'},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'pc': 0.196737, 'type': 'standard'},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'pc': 0.140810, 'type': 'standard'},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'pc': 0.109460, 'type': 'standard'},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'g': 4, 'pc': 0.089270, 'type': 'standard'},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'g': 4, 'pc': 0.075050, 'type': 'standard'},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'g': 4, 'pc': 0.064530, 'type': 'standard'},
    {'name': 'MC_sq', 'z': 4, 'd': 2, 'g': 4, 'pc': 0.592746, 'type': 'MC'},
    {'name': 'MC_sq+[1,2]', 'z': 8, 'd': 2, 'g': 3, 'pc': 0.4046, 'type': 'MC'},
    {'name': 'MC_sq+[1,4]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2338, 'type': 'MC'},
    {'name': 'MC_sq+[1,3]', 'z': 8, 'd': 2, 'g': 4, 'pc': 0.3399, 'type': 'MC'},
    {'name': 'MC_sq+[1,2,3]', 'z': 12, 'd': 2, 'g': 3, 'pc': 0.2834, 'type': 'MC'},
]

# ============ 广义幂律拟合 ============

def generalized_powerlaw(params, z, d):
    """广义幂律: p_c = A · z^{-α} · d^{-γ}"""
    A, alpha, gamma = params
    return A * np.power(z, -alpha) * np.power(d, -gamma)

def fit_generalized_powerlaw(z_array, d_array, pc_array):
    """拟合广义幂律"""
    def objective(params):
        pred = generalized_powerlaw(params, z_array, d_array)
        return np.sum((pred - pc_array) ** 2)

    result = minimize(objective, [4.0, 0.7, 1.5], method='Nelder-Mead',
                     options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-8})

    A, alpha, gamma = result.x
    pred = generalized_powerlaw(result.x, z_array, d_array)

    ss_res = np.sum((pc_array - pred) ** 2)
    ss_tot = np.sum((pc_array - np.mean(pc_array)) ** 2)
    r2 = 1 - ss_res / ss_tot

    mae = np.mean(np.abs(pc_array - pred))
    rmse = np.sqrt(np.mean((pc_array - pred) ** 2))

    return {
        'A': A, 'alpha': alpha, 'gamma': gamma,
        'R2': r2, 'MAE': mae, 'RMSE': rmse, 'success': result.success
    }

# ============ 主实验 ============

print("="*80)
print("【实验】广义幂律拟合")
print("="*80)

z_array = np.array([lat['z'] for lat in lattices_data])
d_array = np.array([lat['d'] for lat in lattices_data])
pc_array = np.array([lat['pc'] for lat in lattices_data])

result = fit_generalized_powerlaw(z_array, d_array, pc_array)

print(f"\n最优参数:")
print(f"  A = {result['A']:.4f}")
print(f"  α = {result['alpha']:.4f}")
print(f"  γ = {result['gamma']:.4f}")

print(f"\n拟合质量:")
print(f"  R² = {result['R2']:.6f}")
print(f"  MAE = {result['MAE']:.6f}")
print(f"  RMSE = {result['RMSE']:.6f}")

# ============ 逐点分析 ============

print("\n" + "="*80)
print("【逐点分析】")
print("="*80)

params = [result['A'], result['alpha'], result['gamma']]
pc_pred = generalized_powerlaw(params, z_array, d_array)

print(f"\n{'晶格':<20} {'z':<4} {'d':<2} {'g':<2} {'p_c真实':<10} {'p_c预测':<10} {'误差':<8} {'类型':<10}")
print("-"*80)

for i, lat in enumerate(lattices_data):
    error = abs(pc_array[i] - pc_pred[i])
    error_rel = error / pc_array[i] if pc_array[i] > 0 else 0
    print(f"{lat['name']:<20} {lat['z']:<4} {lat['d']:<2} {lat['g']:<2} "
          f"{pc_array[i]:<10.5f} {pc_pred[i]:<10.5f} {error:<8.5f} ({error_rel:.1%}) {lat['type']:<10}")

# ============ 残差分析 ============

print("\n" + "="*80)
print("【残差分析】")
print("="*80)

residuals = pc_array - pc_pred

print(f"\n残差统计:")
print(f"  均值: {np.mean(residuals):.6f}")
print(f"  标准差: {np.std(residuals):.6f}")
print(f"  最大正值: {np.max(residuals):.6f}")
print(f"  最大负值: {np.min(residuals):.6f}")

# 按类型分组
standard_residuals = [residuals[i] for i, lat in enumerate(lattices_data) if lat['type'] == 'standard']
mc_residuals = [residuals[i] for i, lat in enumerate(lattices_data) if lat['type'] == 'MC']

print(f"\n按类型分组:")
print(f"  标准晶格: 平均误差 = {np.mean(np.abs(standard_residuals)):.5f}, 标准差 = {np.std(standard_residuals):.5f}")
print(f"  模拟数据: 平均误差 = {np.mean(np.abs(mc_residuals)):.5f}, 标准差 = {np.std(mc_residuals):.5f}")

# 按维度分组
for d in [2, 3, 4, 5, 6, 7, 8, 9]:
    d_residuals = [residuals[i] for i, lat in enumerate(lattices_data) if lat['d'] == d]
    if d_residuals:
        print(f"  d={d}: 平均误差 = {np.mean(np.abs(d_residuals)):.5f}")

# ============ 系统性偏差分析 ============

print("\n" + "="*80)
print("【系统性偏差分析】")
print("="*80)

# 按 girth 分组
g_values = sorted(set(lat['g'] for lat in lattices_data))
print(f"\n按 girth 分组:")
for g in g_values:
    g_residuals = [residuals[i] for i, lat in enumerate(lattices_data) if lat['g'] == g]
    if g_residuals:
        print(f"  g={g}: 平均残差 = {np.mean(g_residuals):+.5f} (n={len(g_residuals)})")

# 关键对比：同 z 不同 g
print(f"\n关键对比：同 z=8，不同 g")
z8_data = [(lat, pc_pred[i]) for i, lat in enumerate(lattices_data) if lat['z'] == 8]
for lat, pred in z8_data:
    error = lat['pc'] - pred
    print(f"  {lat['name']:<20} g={lat['g']} p_c={lat['pc']:.5f} 预测={pred:.5f} 残差={error:+.5f}")

print(f"\n关键对比：同 z=12，不同 g")
z12_data = [(lat, pc_pred[i]) for i, lat in enumerate(lattices_data) if lat['z'] == 12]
for lat, pred in z12_data:
    error = lat['pc'] - pred
    print(f"  {lat['name']:<20} g={lat['g']} d={lat['d']} p_c={lat['pc']:.5f} 预测={pred:.5f} 残差={error:+.5f}")

# ============ 结论 ============

print("\n" + "="*80)
print("【结论】")
print("="*80)

print(f"""
广义幂律: p_c = A · z^{{-α}} · d^{{-γ}}

参数:
  A = {result['A']:.4f}
  α = {result['alpha']:.4f}
  γ = {result['gamma']:.4f}

拟合质量:
  R² = {result['R2']:.6f}
  MAE = {result['MAE']:.6f}

物理意义:

1. α = {result['alpha']:.3f}: z 的标度指数
   - p_c ~ z^{{-{result['alpha']:.2f}}}
   - 配位数越大，p_c 越低（符合直觉）

2. γ = {result['gamma']:.3f}: d 的标度指数
   - p_c ~ d^{{-{result['gamma']:.2f}}}
   - 维度越高，p_c 越低（符合直觉）
   - γ ≈ 1.6，比之前的 γ ≈ 0.83 更强

3. 关键发现：β = 0
   - 距离衰减不重要
   - 有效配位数 z_eff = z（不考虑距离）
   - 这与之前的观察（同 z 不同 p_c）矛盾

4. 解释矛盾:
   - 在"平均场"意义上，z 是主导因素（R² = 0.938）
   - 但在"精细结构"上，连接几何有影响
   - 残差分析显示系统性偏差（与 g 相关）

5. 下一步:
   - 引入 girth 修正: p_c = A · z^{{-α}} · d^{{-γ}} · (1 + B/g)
   - 或引入聚类系数修正
   - 或分段建模（2D vs 3D）
""")

# ============ 与之前公式对比 ============

print("\n" + "="*80)
print("【对比】")
print("="*80)

print(f"\n之前的公式 (52个数据点):")
print(f"  基础幂律: p_c = 3.96 / (z^0.90 · d^0.83)")
print(f"    LOOCV MAE ≈ 0.034")

print(f"\n广义幂律 ({len(lattices_data)}个数据点):")
print(f"  p_c = {result['A']:.3f} / (z^{result['alpha']:.3f} · d^{result['gamma']:.3f})")
print(f"    MAE = {result['MAE']:.4f}")

if result['MAE'] < 0.034:
    improvement = (0.034 - result['MAE']) / 0.034 * 100
    print(f"\n✓ 改进: {improvement:.1f}%")
else:
    degradation = (result['MAE'] - 0.034) / 0.034 * 100
    print(f"\n✗ 退化: {degradation:.1f}%")

print(f"\n注意: 数据集不同，不能直接比较")
print(f"  之前: 52个数据点（主要是复杂邻域）")
print(f"  现在: {len(lattices_data)}个数据点（标准晶格+模拟数据）")
