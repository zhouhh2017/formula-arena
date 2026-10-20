from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
诊断脚本：分析d=3系统性偏差和2D大误差的根因
"""

import numpy as np
import pandas as pd

OUT = ROOT

print("=" * 80)
print("  残差诊断：d=3系统性偏差 + 2D大误差")
print("=" * 80)

# 加载数据
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values
sources = df['source'].values

# 当前模型
def model_exp(params, z, d, g):
    a, alpha, gamma, z0 = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

params = np.array([12.3346, 0.3368, 0.8477, 9.6171])
pred = model_exp(params, z, d, g)
errors = np.abs((pc - pred) / pc) * 100
residuals = (pc - pred) / pc * 100  # 有符号

# ============================================================
# 1. d=3详细分析
# ============================================================
print("\n" + "=" * 80)
print("1. d=3 详细分析 (18个晶格)")
print("=" * 80)

mask_d3 = d == 3
idx_d3 = np.where(mask_d3)[0]

print(f"\n{'#':>2} {'晶格':<25} {'z':>4} {'g':>2} {'p_c实测':>9} {'p_c预测':>9} {'残差%':>7} {'来源':<30}")
print("-" * 95)

for rank, idx in enumerate(idx_d3, 1):
    print(f"{rank:>2} {names[idx]:<25} {int(z[idx]):>4} {int(g[idx]):>2} "
          f"{pc[idx]:>9.5f} {pred[idx]:>9.5f} {residuals[idx]:>+7.2f} {sources[idx]:<30}")

print(f"\nd=3统计:")
print(f"  残差均值: {np.mean(residuals[mask_d3]):+.2f}%")
print(f"  残差标准差: {np.std(residuals[mask_d3]):.2f}%")
print(f"  MAPE: {np.mean(errors[mask_d3]):.2f}%")

# 分析哪些晶格贡献了偏差
print(f"\nd=3残差分布:")
print(f"  残差<-10%: {np.sum(residuals[mask_d3] < -10)}")
print(f"  -10%≤残差<-5%: {np.sum((residuals[mask_d3] >= -10) & (residuals[mask_d3] < -5))}")
print(f"  -5%≤残差<0%: {np.sum((residuals[mask_d3] >= -5) & (residuals[mask_d3] < 0))}")
print(f"  0%≤残差<5%: {np.sum((residuals[mask_d3] >= 0) & (residuals[mask_d3] < 5))}")
print(f"  5%≤残差<10%: {np.sum((residuals[mask_d3] >= 5) & (residuals[mask_d3] < 10))}")
print(f"  残差≥10%: {np.sum(residuals[mask_d3] >= 10)}")

# 检查是否与z或g相关
print(f"\nd=3残差与z、g的相关性:")
print(f"  与z的相关系数: {np.corrcoef(z[mask_d3], residuals[mask_d3])[0,1]:.3f}")
print(f"  与g的相关系数: {np.corrcoef(g[mask_d3], residuals[mask_d3])[0,1]:.3f}")
print(f"  与p_c的相关系数: {np.corrcoef(pc[mask_d3], residuals[mask_d3])[0,1]:.3f}")

# ============================================================
# 2. 2D sq系列分析
# ============================================================
print("\n" + "=" * 80)
print("2. 2D sq系列分析 (Malarz 2023)")
print("=" * 80)

mask_sq = (d == 2) & (pd.Series(names).str.contains('sq-', na=False))
idx_sq = np.where(mask_sq)[0]
idx_sq_sorted = idx_sq[np.argsort(errors[idx_sq])[::-1]]

print(f"\n2D sq系列: {len(idx_sq)}个晶格")
print(f"MAPE: {np.mean(errors[mask_sq]):.2f}%")
print(f"残差均值: {np.mean(residuals[mask_sq]):+.2f}%")

print(f"\n误差最大的10个sq晶格:")
print(f"{'#':>2} {'晶格':<20} {'z':>4} {'g':>2} {'p_c实测':>9} {'p_c预测':>9} {'残差%':>7}")
print("-" * 70)
for rank, idx in enumerate(idx_sq_sorted[:10], 1):
    print(f"{rank:>2} {names[idx]:<20} {int(z[idx]):>4} {int(g[idx]):>2} "
          f"{pc[idx]:>9.5f} {pred[idx]:>9.5f} {residuals[idx]:>+7.2f}")

# 检查sq系列的模式
print(f"\nsq系列残差模式:")
print(f"  g=3的sq: n={np.sum((mask_sq) & (g==3))}, 残差均值={np.mean(residuals[mask_sq & (g==3)]):+.2f}%")
print(f"  g=4的sq: n={np.sum((mask_sq) & (g==4))}, 残差均值={np.mean(residuals[mask_sq & (g==4)]):+.2f}%")

# ============================================================
# 3. 全局残差模式
# ============================================================
print("\n" + "=" * 80)
print("3. 全局残差模式")
print("=" * 80)

# 按(d, g)组合
print(f"\n按(d, g)组合的残差:")
print(f"{'(d,g)':<10} {'n':>4} {'残差均值':>10} {'残差std':>10} {'MAPE':>8}")
print("-" * 50)

for d_val in sorted(set(d.astype(int))):
    for g_val in sorted(set(g.astype(int))):
        mask = (d.astype(int) == d_val) & (g.astype(int) == g_val)
        if np.sum(mask) >= 3:  # 至少3个样本
            print(f"({d_val},{g_val}){'':<6} {np.sum(mask):>4} "
                  f"{np.mean(residuals[mask]):>+10.2f} {np.std(residuals[mask]):>10.2f} "
                  f"{np.mean(errors[mask]):>8.2f}")

# ============================================================
# 4. 缺失变量检查
# ============================================================
print("\n" + "=" * 80)
print("4. 缺失变量检查")
print("=" * 80)

# 检查是否d×g交互项有解释力
dg_interaction = d * g
corr_dg = np.corrcoef(dg_interaction, residuals)[0, 1]
print(f"\n残差与d×g的相关系数: {corr_dg:.3f}")

# 检查z/d比率
z_over_d = z / d
corr_zd = np.corrcoef(z_over_d, residuals)[0, 1]
print(f"残差与z/d的相关系数: {corr_zd:.3f}")

# 检查1/z
corr_1z = np.corrcoef(1/z, residuals)[0, 1]
print(f"残差与1/z的相关系数: {corr_1z:.3f}")

# 检查log(z)
corr_logz = np.corrcoef(np.log(z), residuals)[0, 1]
print(f"残差与log(z)的相关系数: {corr_logz:.3f}")

# ============================================================
# 5. 数据质量检查
# ============================================================
print("\n" + "=" * 80)
print("5. 数据质量检查")
print("=" * 80)

# 检查同一晶格在不同来源的数据是否一致
print(f"\n检查重复晶格:")
unique_lattices = {}
for i in range(len(df)):
    key = (names[i], int(z[i]), int(d[i]), int(g[i]))
    if key in unique_lattices:
        pc_prev = unique_lattices[key]
        diff = abs(pc[i] - pc_prev)
        if diff > 0.001:
            print(f"  ⚠ 不一致: {names[i]} (z={int(z[i])}, d={int(d[i])}, g={int(g[i])})")
            print(f"     p_c={pc_prev:.6f} vs {pc[i]:.6f} (差异{diff:.6f})")
    else:
        unique_lattices[key] = pc[i]

print(f"✓ 共{len(unique_lattices)}个唯一晶格配置")

# 检查是否有异常值
print(f"\n异常值检测 (|残差| > 3σ):")
threshold = 3 * np.std(residuals)
outliers = np.abs(residuals) > threshold
if outliers.any():
    for idx in np.where(outliers)[0]:
        print(f"  {names[idx]}: 残差={residuals[idx]:+.2f}% (>{threshold:.2f}%)")
else:
    print(f"  ✓ 无异常值 (阈值={threshold:.2f}%)")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
