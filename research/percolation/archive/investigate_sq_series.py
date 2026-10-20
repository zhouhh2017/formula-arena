from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
深入调查2D sq系列 (Malarz 2023) 的数据质量和公式适用性
"""

import numpy as np
import pandas as pd

OUT = ROOT

df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values
sources = df['source'].values

def model_exp(params, z, d, g):
    a, alpha, gamma, z0 = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

params = np.array([12.3346, 0.3368, 0.8477, 9.6171])
pred = model_exp(params, z, d, g)
errors = np.abs((pc - pred) / pc) * 100
residuals = (pc - pred) / pc * 100

print("=" * 90)
print("  2D sq系列深入调查")
print("=" * 90)

# sq系列完整数据
mask_sq = (d == 2) & (pd.Series(names).str.contains('sq-', na=False))
idx_sq = np.where(mask_sq)[0]

print(f"\n所有2D sq系列数据 ({len(idx_sq)}个):")
print(f"{'#':>2} {'晶格':<20} {'z':>4} {'g':>2} {'p_c实测':>9} {'p_c预测':>9} {'残差%':>7} {'R实测':>8} {'R预测':>8}")
print("-" * 90)

# 按z和g排序
idx_sorted = idx_sq[np.lexsort((g[idx_sq], z[idx_sq]))]

for rank, idx in enumerate(idx_sorted, 1):
    R_real = pc[idx] * (z[idx] - 1)
    R_pred = pred[idx] * (z[idx] - 1)
    print(f"{rank:>2} {names[idx]:<20} {int(z[idx]):>4} {int(g[idx]):>2} "
          f"{pc[idx]:>9.5f} {pred[idx]:>9.5f} {residuals[idx]:>+7.2f} "
          f"{R_real:>8.4f} {R_pred:>8.4f}")

# 分析R = p_c × (z-1)的模式
print(f"\n{'='*90}")
print("R = p_c × (z-1) 分析")
print(f"{'='*90}")

# 按(z, g)分组看R
print(f"\n按(z, g)分组的R值:")
for z_val in sorted(set(z[mask_sq].astype(int))):
    for g_val in [3, 4]:
        mask_sub = mask_sq & (z.astype(int) == z_val) & (g.astype(int) == g_val)
        if mask_sub.sum() > 0:
            R_vals = pc[mask_sub] * (z_val - 1)
            R_pred_val = pred[mask_sub][0] * (z_val - 1)
            print(f"  z={z_val:>3}, g={g_val}: R实测={np.mean(R_vals):.4f}±{np.std(R_vals):.4f}, "
                  f"R预测={R_pred_val:.4f}, n={mask_sub.sum()}")

# 检查Malarz 2023数据与其他2D数据的一致性
print(f"\n{'='*90}")
print("与其他2D数据源的对比")
print(f"{'='*90}")

# 非sq的2D数据
mask_2d_other = (d == 2) & ~(pd.Series(names).str.contains('sq-', na=False))
idx_2d_other = np.where(mask_2d_other)[0]

print(f"\n非sq的2D数据 ({len(idx_2d_other)}个):")
print(f"{'#':>2} {'晶格':<22} {'z':>4} {'g':>2} {'p_c实测':>9} {'p_c预测':>9} {'残差%':>7} {'来源':<30}")
print("-" * 95)

for rank, idx in enumerate(idx_2d_other, 1):
    print(f"{rank:>2} {names[idx]:<22} {int(z[idx]):>4} {int(g[idx]):>2} "
          f"{pc[idx]:>9.5f} {pred[idx]:>9.5f} {residuals[idx]:>+7.2f} {sources[idx]:<30}")

# 检查sq系列中同一girth不同z的行为
print(f"\n{'='*90}")
print("sq系列 g=3 的 p_c vs z 行为")
print(f"{'='*90}")

mask_sq_g3 = mask_sq & (g == 3)
idx_sq_g3 = np.where(mask_sq_g3)[0]
idx_sq_g3_sorted = idx_sq_g3[np.argsort(z[idx_sq_g3])]

print(f"\n{'z':>4} {'p_c实测':>9} {'R实测':>8} {'p_c预测':>9} {'R预测':>8} {'残差%':>7}")
print("-" * 55)
for idx in idx_sq_g3_sorted:
    R_real = pc[idx] * (z[idx] - 1)
    R_pred = pred[idx] * (z[idx] - 1)
    print(f"{int(z[idx]):>4} {pc[idx]:>9.5f} {R_real:>8.4f} {pred[idx]:>9.5f} {R_pred:>8.4f} {residuals[idx]:>+7.2f}")

# 与经典2D数据的对比
print(f"\n{'='*90}")
print("与理论精确值的对比")
print(f"{'='*90}")

# 已知精确/高精度结果
exact_2d = {
    'triangular': (6, 2, 3, 0.5),
    'honeycomb': (3, 2, 6, 0.697043),
    'square': (4, 2, 4, 0.592746),
    'kagome': (4, 2, 3, 0.6527036),
}

print(f"\n{'晶格':<15} {'z':>3} {'g':>2} {'p_c(文献)':>10} {'p_c(预测)':>10} {'误差%':>7}")
print("-" * 55)
for name, (z_e, d_e, g_e, pc_e) in exact_2d.items():
    idx = np.where(names == name)[0]
    if len(idx) > 0:
        i = idx[0]
        print(f"{name:<15} {z_e:>3} {g_e:>2} {pc_e:>10.6f} {pred[i]:>10.6f} {errors[i]:>6.2f}%")

# 分析sq系列的命名规则
print(f"\n{'='*90}")
print("sq系列命名规则分析")
print(f"{'='*90}")
print("\nsq-X,Y,Z 表示在square晶格(配位数4)上添加X,Y,Z近邻连接")
print("例如 sq-2,6 表示添加第2和第6近邻 → z = 4+4+... ")
print("\n问题: sq-2,6 的 z=8 是否合理?")
print("  square: z=4 (4个最近邻)")
print("  +第2近邻: 4个 → z=8 ✓")
print("  +第6近邻: 4个 → z=12")
print("  所以 sq-2,6 应该有 z=4+4+4=12?")

# 检查sq系列的z是否自洽
print(f"\n检查sq系列的z值:")
for idx in idx_sq:
    name_str = str(names[idx])
    # 解析名称中的连接数
    parts = name_str.replace('sq-', '').split(',')
    n_extra = len(parts)
    z_expected_min = 4 + 4 * n_extra  # square + 每个额外连接+4
    print(f"  {name_str:<20} z={int(z[idx]):>3} (名称含{n_extra}个额外连接)")

# 核心问题：sq系列中 z=8, g=3 的物理含义
print(f"\n{'='*90}")
print("核心问题分析")
print(f"{'='*90}")

print(f"""
sq-2,6: z=8, g=3
  - 名称含义: square + 第2近邻 + 第6近邻
  - z=8: square(4) + 第2近邻(4) = 8, 第6近邻未加入?
  - 或者: square(4) + 某种组合 = 8
  - g=3: 最短环长度为3 (三角形)

  公式预测: p_c = 0.34279 (实测 0.26600)
  残差: -28.87% (预测远高于实测)

  R实测 = 0.266 × 7 = 1.862
  R预测 = 0.343 × 7 = 2.400

  问题: 公式预测R=2.40, 实测R=1.86
  这意味着对于 z=8, g=3, d=2, 实际p_c比公式预期更低
  可能的原因:
  1. g=3的三角形环在2D中特别有效地促进渗流
  2. 数据本身可能有误 (Malarz 2023的MC模拟可能有系统偏差)
  3. 公式在2D+小g+中等z的区域有根本性盲区
""")

# 对比：d=2, g=3, z=4 (kagome)
print(f"\n对比: d=2, g=3 的不同z")
mask_d2_g3 = (d == 2) & (g == 3)
idx_d2_g3 = np.where(mask_d2_g3)[0]
idx_d2_g3_sorted = idx_d2_g3[np.argsort(z[idx_d2_g3])]

print(f"{'晶格':<20} {'z':>4} {'p_c实测':>9} {'R实测':>8} {'p_c预测':>9} {'R预测':>8} {'残差%':>7}")
print("-" * 72)
for idx in idx_d2_g3_sorted:
    R_real = pc[idx] * (z[idx] - 1)
    R_pred = pred[idx] * (z[idx] - 1)
    print(f"{names[idx]:<20} {int(z[idx]):>4} {pc[idx]:>9.5f} {R_real:>8.4f} "
          f"{pred[idx]:>9.5f} {R_pred:>8.4f} {residuals[idx]:>+7.2f}")

print(f"\n{'='*90}")
print("调查完成")
print(f"{'='*90}")
