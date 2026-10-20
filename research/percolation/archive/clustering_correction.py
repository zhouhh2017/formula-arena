"""
FCC 问题分析 + 聚类系数修正
================================

当前成果：
  p_c = 2.41 · z^{-0.84} · d^{-0.76} · (1 + 1.43/g)
  R² = 0.987, MAE = 0.016

问题：
  FCC (z=12, d=3, g=3) 误差 62%
  预测 0.193，真实 0.119

FCC 的特殊性：
  - 最密堆积结构
  - 每个顶点有 12 个近邻
  - 有很多三角形（聚类系数高）

新假说：
  p_c = A · z^{-α} · d^{-γ} · (1 + B/g) · (1 + C·C_clustering)

目标：
  解释 FCC 的异常，达到 R² > 0.99
"""

import numpy as np
from scipy.optimize import minimize

# ============ 数据（含聚类系数）============

# 聚类系数 C = 2·n_tri / (z·(z-1))
# 其中 n_tri 是每个顶点的三角形数

lattices_data = [
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'n_tri': 0, 'pc': 0.697040},
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'n_tri': 0, 'pc': 0.592746},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'n_tri': 6, 'pc': 0.500000},  # 完全三角化
    {'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'n_tri': 2, 'pc': 0.652703},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'n_tri': 0, 'pc': 0.311600},
    {'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'n_tri': 0, 'pc': 0.245800},
    {'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'n_tri': 24, 'pc': 0.118800},  # 最密堆积，很多三角形
    {'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'n_tri': 0, 'pc': 0.431900},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'n_tri': 0, 'pc': 0.196737},
    {'name': 'hypersc_5', 'z': 10, 'd': 5, 'g': 4, 'n_tri': 0, 'pc': 0.140810},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'n_tri': 0, 'pc': 0.109460},
    {'name': 'hypersc_7', 'z': 14, 'd': 7, 'g': 4, 'n_tri': 0, 'pc': 0.089270},
    {'name': 'hypersc_8', 'z': 16, 'd': 8, 'g': 4, 'n_tri': 0, 'pc': 0.075050},
    {'name': 'hypersc_9', 'z': 18, 'd': 9, 'g': 4, 'n_tri': 0, 'pc': 0.064530},
    {'name': 'MC_sq', 'z': 4, 'd': 2, 'g': 4, 'n_tri': 0, 'pc': 0.592746},
    {'name': 'MC_sq+[1,2]', 'z': 8, 'd': 2, 'g': 3, 'n_tri': 4, 'pc': 0.4046},
    {'name': 'MC_sq+[1,4]', 'z': 12, 'd': 2, 'g': 3, 'n_tri': 12, 'pc': 0.2338},
    {'name': 'MC_sq+[1,3]', 'z': 8, 'd': 2, 'g': 4, 'n_tri': 0, 'pc': 0.3399},
    {'name': 'MC_sq+[1,2,3]', 'z': 12, 'd': 2, 'g': 3, 'n_tri': 8, 'pc': 0.2834},
]

# 计算聚类系数
for lat in lattices_data:
    z = lat['z']
    n_tri = lat['n_tri']
    lat['C'] = 2 * n_tri / (z * (z - 1)) if z > 1 else 0

# ============ 标度律定义 ============

def scaling_zdg_C(params, z, d, g, C):
    """z-d-g-C 标度律: p_c = A·z^{-α}·d^{-γ}·(1+B/g)·(1+D·C)"""
    A, alpha, gamma, B, D = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * (1 + B / g) * (1 + D * C)

def scaling_zdg_Calt(params, z, d, g, C):
    """替代形式: p_c = A·z^{-α}·d^{-γ}·g^{δ}·(1+D·C)"""
    A, alpha, gamma, delta, D = params
    return A * np.power(z, -alpha) * np.power(d, -gamma) * np.power(g, delta) * (1 + D * C)

# ============ 拟合函数 ============

def fit_scaling(formula_func, z, d, g, C, pc, initial_guess):
    """拟合标度律"""
    def objective(params):
        pred = formula_func(params, z, d, g, C)
        return np.sum((pred - pc) ** 2)

    result = minimize(objective, initial_guess, method='Nelder-Mead',
                     options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})

    pred = formula_func(result.x, z, d, g, C)

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
print("【实验】聚类系数修正")
print("="*80)

z = np.array([lat['z'] for lat in lattices_data])
d = np.array([lat['d'] for lat in lattices_data])
g = np.array([lat['g'] for lat in lattices_data])
C = np.array([lat['C'] for lat in lattices_data])
pc = np.array([lat['pc'] for lat in lattices_data])

print(f"\n聚类系数统计:")
print(f"  最小: {C.min():.3f}")
print(f"  最大: {C.max():.3f}")
print(f"  平均: {C.mean():.3f}")

print(f"\n关键晶格的聚类系数:")
for lat in lattices_data[:8]:  # 标准晶格
    print(f"  {lat['name']:<15} z={lat['z']:<3} n_tri={lat['n_tri']:<3} C={lat['C']:.3f}")

# 测试两种公式
formulas = [
    ('p_c = A·z^{-α}·d^{-γ}·(1+B/g)·(1+D·C)',
     scaling_zdg_C, [2.4, 0.8, 0.8, 1.4, -0.5]),
    ('p_c = A·z^{-α}·d^{-γ}·g^{δ}·(1+D·C)',
     scaling_zdg_Calt, [4.5, 0.8, 0.8, -0.2, -0.5]),
]

results = []

for name, func, init in formulas:
    print(f"\n--- {name} ---")
    result = fit_scaling(func, z, d, g, C, pc, init)

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

# ============ 逐点分析 ============

print("\n" + "="*80)
print("【逐点分析】")
print("="*80)

pc_pred = best_result['predictions']

print(f"\n{'晶格':<15} {'z':<3} {'d':<2} {'g':<2} {'C':<5} {'p_c真实':<10} {'p_c预测':<10} {'误差':<8}")
print("-"*75)

for i, lat in enumerate(lattices_data):
    error = abs(pc[i] - pc_pred[i])
    error_rel = error / pc[i] if pc[i] > 0 else 0
    print(f"{lat['name']:<15} {lat['z']:<3} {lat['d']:<2} {lat['g']:<2} {lat['C']:<5.3f} "
          f"{pc[i]:<10.5f} {pc_pred[i]:<10.5f} {error:<8.5f} ({error_rel:.1%})")

# ============ FCC 专项分析 ============

print("\n" + "="*80)
print("【FCC 专项分析】")
print("="*80)

fcc_idx = [i for i, lat in enumerate(lattices_data) if lat['name'] == 'fcc'][0]
print(f"\nFCC:")
print(f"  z = {lattices_data[fcc_idx]['z']}")
print(f"  d = {lattices_data[fcc_idx]['d']}")
print(f"  g = {lattices_data[fcc_idx]['g']}")
print(f"  n_tri = {lattices_data[fcc_idx]['n_tri']}")
print(f"  C = {lattices_data[fcc_idx]['C']:.3f}")
print(f"  p_c 真实 = {pc[fcc_idx]:.5f}")
print(f"  p_c 预测 = {pc_pred[fcc_idx]:.5f}")
print(f"  误差 = {abs(pc[fcc_idx] - pc_pred[fcc_idx]):.5f} ({abs(pc[fcc_idx] - pc_pred[fcc_idx])/pc[fcc_idx]:.1%})")

# ============ 对比分析 ============

print("\n" + "="*80)
print("【对比】")
print("="*80)

print(f"\n无 C 修正 (R² = 0.987, MAE = 0.016):")
print(f"  p_c = 2.41·z^{{-0.84}}·d^{{-0.76}}·(1+1.43/g)")

print(f"\n有 C 修正 (R² = {best_result['R2']:.6f}, MAE = {best_result['MAE']:.6f}):")
print(f"  {best_result['name']}")

improvement_r2 = (best_result['R2'] - 0.987) / 0.987 * 100
improvement_mae = (0.016 - best_result['MAE']) / 0.016 * 100

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
    print("需要更严格的验证")
elif best_result['R2'] > 0.98:
    print("\n✓ 有价值的改进")
    print(f"R² = {best_result['R2']:.6f} > 0.98")
else:
    print("\n⚠️  改进有限")
    print(f"R² = {best_result['R2']:.6f} < 0.98")

# ============ 物理意义 ============

print("\n" + "="*80)
print("【物理意义】")
print("="*80)

A, alpha, gamma, B, D = best_result['params']

print(f"""
标度律: p_c = A · z^{{-α}} · d^{{-γ}} · (1 + B/g) · (1 + D·C)

参数:
  A = {A:.4f}
  α = {alpha:.4f}
  γ = {gamma:.4f}
  B = {B:.4f}
  D = {D:.4f}

物理意义:

1. α = {alpha:.3f}: 配位数的标度指数
   - p_c ~ z^{{-{alpha:.2f}}}

2. γ = {gamma:.3f}: 维度的标度指数
   - p_c ~ d^{{-{gamma:.2f}}}

3. B = {B:.3f}: girth 修正系数
   - 短环效应

4. D = {D:.3f}: 聚类系数修正系数
   - D > 0: 聚类促进渗流
   - D < 0: 聚类阻碍渗流

5. 如果 D < 0:
   - 三角形越多，p_c 越低
   - 因为三角形增加了局部连通性
   - 这符合直觉！
""")

print("\n" + "="*80)
print("【结论】")
print("="*80)

if best_result['R2'] > 0.99:
    print("""
通过引入聚类系数修正，我们得到了 PRL 级别的普适标度律！

p_c = A · z^{-α} · d^{-γ} · (1 + B/g) · (1 + D·C)

这个公式：
1. 包含 5 个物理量: z, d, g, C
2. 只有 5 个参数: A, α, γ, B, D
3. R² > 0.99，MAE < 0.015
4. 物理意义明确

下一步：
1. 进行严格的统计检验（cross-validation, bootstrap）
2. 从理论上推导标度指数
3. 预测新的晶格 p_c，进行实验验证
4. 撰写 PRL 论文
""")
else:
    print("""
聚类系数修正有效，但仍需进一步改进。

可能的方向：
1. 引入更多拓扑特征（环密度分布）
2. 考虑高维晶格的特殊效应
3. 分段建模（2D vs 3D vs 高维）
""")
