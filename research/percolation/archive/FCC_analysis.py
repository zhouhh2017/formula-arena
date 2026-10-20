from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
FCC 异常分析
====================
公式对 FCC (z=12, d=3, g=3) 给出 72.76% 误差！
这揭示了公式的根本缺陷。
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')

z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values

print("=" * 80)
print("[FCC 异常深度分析]")
print("=" * 80)

# 经验公式
a_emp, alpha_emp, gamma_emp, z0_emp = 12.3346, 0.3368, 0.8477, 9.6171

def model_emp(z, d, g):
    correction = (a_emp / g**gamma_emp) * np.exp(-alpha_emp * d) * (1 - np.exp(-z / z0_emp))
    return (1 + correction) / (z - 1)

# ============================================================
# 分析 FCC 异常
# ============================================================
print("\n1. FCC 在各公式分量中的值:")
z_fcc, d_fcc, g_fcc = 12, 3, 3
pc_fcc = 0.1190

girth_factor = a_emp / g_fcc**gamma_emp
dim_factor = np.exp(-alpha_emp * d_fcc)
sat_factor = 1 - np.exp(-z_fcc / z0_emp)

pred = model_emp(z_fcc, d_fcc, g_fcc)
R_pred = pred * (z_fcc - 1)
Delta_R_pred = R_pred - 1

R_exact = pc_fcc * (z_fcc - 1)
Delta_R_exact = R_exact - 1

print(f"  z={z_fcc}, d={d_fcc}, g={g_fcc}")
print(f"  girth factor: a/g^gamma = {girth_factor:.4f}")
print(f"  dim factor: exp(-alpha*d) = {dim_factor:.4f}")
print(f"  sat factor: 1-exp(-z/z0) = {sat_factor:.4f}")
print(f"  correction = {girth_factor * dim_factor * sat_factor:.4f}")
print(f"  pred = {pred:.4f}, exact = {pc_fcc:.4f}, error = {abs(pred-pc_fcc)/pc_fcc*100:.2f}%")
print(f"  Delta_R pred = {Delta_R_pred:.4f}, exact = {Delta_R_exact:.4f}")

# 分解问题
print(f"\n2. 各因子的贡献分析:")
print(f"  需要的 correction = {Delta_R_exact:.4f}")
print(f"  实际的 correction = {Delta_R_pred:.4f}")
print(f"  比率 = {Delta_R_pred/Delta_R_exact:.2f}x (overestimate)")

# 哪个因子导致了偏差？
# 如果固定其他因子，需要 a/g^gamma = ?
needed_girth = Delta_R_exact / (dim_factor * sat_factor)
print(f"\n3. 需要的 girth 因子:")
print(f"  实际: a/g^gamma = {girth_factor:.4f}")
print(f"  需要: {needed_girth:.4f}")
print(f"  比率: {girth_factor/needed_girth:.2f}x")

# ============================================================
# 比较所有高 z 晶格
# ============================================================
print(f"\n4. 所有 z >= 10 的晶格:")
print(f"{'Name':<25} {'z':>4} {'d':>2} {'g':>2} {'pc':>8} {'pred':>8} {'error%':>8}")
print("-" * 70)

mask = z >= 10
for i in np.where(mask)[0]:
    pred_i = model_emp(z[i], d[i], g[i])
    err = abs(pred_i - pc[i]) / pc[i] * 100
    print(f"{names[i]:<25} {z[i]:>4.0f} {d[i]:>2.0f} {g[i]:>2.0f} {pc[i]:>8.4f} {pred_i:>8.4f} {err:>8.2f}%")

# ============================================================
# 分析误差与 z 的关系
# ============================================================
print(f"\n5. 误差 vs z 的系统性:")
pred_all = model_emp(z, d, g)
err_all = (pred_all - pc) / pc * 100

for z_range in [(3, 5), (5, 8), (8, 12), (12, 20), (20, 100)]:
    mask = (z >= z_range[0]) & (z < z_range[1])
    if np.sum(mask) > 0:
        mean_err = np.mean(err_all[mask])
        print(f"  z in [{z_range[0]}, {z_range[1]}): N={np.sum(mask)}, mean error = {mean_err:+.2f}%")

# ============================================================
# 关键问题：为什么 FCC 误差这么大？
# ============================================================
print(f"\n6. FCC 的特殊性分析")
print(f"""
FCC (face-centered cubic) 的特殊性:

- z = 12 (非常高)
- d = 3
- g = 3 (每个键都在三角形中)
- p_c = 0.1190 (非常低)

FCC 是"最密排"的三维格：
- 每个格点被 12 个近邻包围
- 近邻之间形成大量三角形
- 结构极其"紧凑"

问题：对于高 z + 高"紧密度"的格，
我们的公式高估了 p_c。

原因可能是：
1. 饱和因子 1-exp(-z/z0) 增长太快
2. 没有考虑"紧密度"的独立效应
3. 高 z 时，环路重叠导致修正减小
""")

# ============================================================
# 计算"公共近邻"数
# ============================================================
print(f"\n7. 公共近邻分析")
print("""
对于相邻格点 u, v：
- b = 公共近邻数
- g=3 时 b > 0
- g>=4 时 b = 0

对于 FCC:
- 每个键在多个三角形中
- b 可能很大

实际上，对于 FCC：
- 近邻形成一个 cuboctahedron
- 12 个近邻之间的连接数 = 24 (边)
- 所以 b = 24/12 * 2 = 4? (每个近邻有4个公共近邻)

对于三角格 (z=6, d=2, g=3):
- 近邻形成一个六边形
- 相邻近邻之间共享一个近邻
- b = 2

b 越大 → 环路越多 → p_c 应该越低?

等等！这与我们的公式方向相反！
我们的公式中 g=3 → 修正大 → p_c 升高
但 b 大 → 更多连通性 → p_c 应该降低？

让我重新思考...
""")

# ============================================================
# 重新检查：修正的方向
# ============================================================
print(f"\n8. 修正方向的分析")

# Bethe 极限
pc_bethe_fcc = 1 / (z_fcc - 1)
print(f"  Bethe 近似 FCC: p_c = 1/11 = {pc_bethe_fcc:.4f}")
print(f"  实际 FCC: p_c = {pc_fcc:.4f}")
print(f"  比率: p_c/Bethe = {pc_fcc/pc_bethe_fcc:.4f}")

# 对比其他格
print(f"\n  不同格的 p_c / p_c^Bethe:")
print(f"  {'Name':<25} {'z':>4} {'g':>2} {'p_c':>8} {'p_c^Bethe':>10} {'Ratio':>8}")
print(f"  " + "-" * 65)

for i in [0, 1, 2, len(df)-1]:  # 选几个代表性的
    pc_bethe = 1 / (z[i] - 1)
    ratio = pc[i] / pc_bethe
    print(f"  {names[i]:<25} {z[i]:>4.0f} {g[i]:>2.0f} {pc[i]:>8.4f} {pc_bethe:>10.4f} {ratio:>8.4f}")

# FCC
pc_bethe_fcc = 1 / 11
print(f"  {'FCC':<25} {'12':>4} {'3':>2} {0.1190:>8.4f} {pc_bethe_fcc:>10.4f} {0.1190/pc_bethe_fcc:>8.4f}")

# ============================================================
# 关键发现
# ============================================================
print(f"""
9. 关键发现！

Bethe 近似：
  FCC: p_c^Bethe = 1/11 = 0.0909
  实际: p_c = 0.1190
  Ratio = 1.31

  三角: p_c^Bethe = 1/5 = 0.2000
  实际: p_c = 0.5000
  Ratio = 2.50

所以 FCC 实际上比三角更接近 Bethe 极限！

这意味着：
- 虽然 FCC 有大量三角形
- 但高配位数 z=12 使得局部结构更像树
- 环路修正相对较小

我们的公式的问题：
- girth factor 给出大的修正（因为 g=3）
- sat factor 也给出大的修正（因为 z=12）
- 两者结合导致过度修正

正确的物理应该是：
- 高 z 时，虽然环路多，但每个环路的相对贡献小
- 饱和函数应该更快饱和
- 或者 girth 因子应该被 z 调制

10. 可能的改进
----------------------------------

方案 A: girth-z 交叉项
  girth_factor = a / g^(gamma * f(z))
  其中 f(z) 使得高 z 时 girth 效应减弱

方案 B: 公共近邻数 b 作为独立参数
  需要计算每个格的 b 值

方案 C: 修改饱和函数
  使用不同的 z 依赖

方案 D: 接受 FCC 作为异常值
  在论文中诚实报告局限性
""")

# ============================================================
# 检查数据集中的 FCC
# ============================================================
print(f"\n11. 数据集中 FCC 的具体信息:")
fcc_mask = pd.Series(names).str.contains('fcc|FCC|fcc', case=False, na=False)
for i in np.where(fcc_mask)[0]:
    print(f"  {names[i]}: z={z[i]:.0f}, d={d[i]:.0f}, g={g[i]:.0f}, pc={pc[i]:.4f}")

# 如果没有找到 FCC，检查 z=12, d=3 的格
mask_z12_d3 = (z == 12) & (d == 3)
print(f"\n  z=12, d=3 的格:")
for i in np.where(mask_z12_d3)[0]:
    pred_i = model_emp(z[i], d[i], g[i])
    err = abs(pred_i - pc[i]) / pc[i] * 100
    print(f"    {names[i]}: pc={pc[i]:.4f}, pred={pred_i:.4f}, err={err:.2f}%")

# 也检查 z=12 的其他格
mask_z12 = z == 12
print(f"\n  所有 z=12 的格:")
for i in np.where(mask_z12)[0]:
    pred_i = model_emp(z[i], d[i], g[i])
    err = abs(pred_i - pc[i]) / pc[i] * 100
    print(f"    {names[i]}: d={d[i]:.0f}, g={g[i]:.0f}, pc={pc[i]:.4f}, pred={pred_i:.4f}, err={err:.2f}%")

# ============================================================
# 总结
# ============================================================
print(f"\n" + "=" * 80)
print("总结: FCC 异常的意义")
print("=" * 80)
print(f"""
FCC 的 72.76% 误差揭示了公式的根本问题：

1. 公式假设 g, d, z 的效应是独立可分离的
2. 但实际上它们有交叉效应
3. 对于高 z + g=3 的密排格，公式严重失效

这对 PRL 的意义：

- 如果公式对某些格误差 >70%，它不是"普适"的
- 这限制了论文的影响力和价值
- PRE 审稿人可能会质疑公式的适用范围

建议：
1. 在论文中诚实讨论 FCC 异常
2. 将公式适用范围限定为 z < 10 或特定格类
3. 或者寻找能解释 FCC 的新模型

这可能是改进的方向！
""")
