from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
从第一原理推导参数 a - 第5部分
=================================
新方向：
1. 极限行为分析
2. a 作为 z 的函数
3. 从 Bethe 格极限推导
4. 寻找隐藏的对偶性
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
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
print("[从第一原理推导参数 a - 第5部分]")
print("=" * 80)

# ============================================================
# 方法 20：a 作为 z 的函数
# ============================================================
print("\n" + "=" * 80)
print("方法 20：a 作为 z 的函数")
print("=" * 80)

print("""
20.1 思路
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

也许 a 不是常数，而是 z 的函数：a = a(z)

如果在某些特殊 z 值处 a 有简单的表达式，
也许可以推断出一般形式。

20.2 策略
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于每个 (d, g) 组合，从数据反推 a(z)
然后寻找 a(z) 的模式。
""")

# 对于每组 (d, g)，反推 a
print(f"\n反推 a 作为 z 的函数:")
print(f"{'d':>2} {'g':>2} {'z':>4} {'p_c':>8} {'a (反推)':>10}")
print("-" * 40)

# 选择一些代表性晶格
representative = [
    (2, 3, 6),   # triangular
    (2, 4, 4),   # square
    (2, 6, 3),   # honeycomb
    (3, 3, 12),  # FCC
    (3, 4, 6),   # SC
    (3, 4, 8),   # BCC
    (4, 4, 8),   # SC-4d
    (8, 4, 16),  # SC-8d
]

for d_val, g_val, z_val in representative:
    mask = (d == d_val) & (g == g_val) & (z == z_val)
    if np.sum(mask) > 0:
        pc_val = pc[mask][0]
        # 反推 a
        # pc = 1/(z-1) * [1 + (a/g) * exp(-d/3) * (1-exp(-z/9.48))]
        # (z-1)*pc - 1 = (a/g) * exp(-d/3) * (1-exp(-z/9.48))
        rhs = (z_val - 1) * pc_val - 1
        dim_factor = np.exp(-d_val/3)
        sat_factor = 1 - np.exp(-z_val/9.48)

        if rhs > 0:
            a_reverse = g_val * rhs / (dim_factor * sat_factor)
            print(f"{d_val:>2} {g_val:>2} {z_val:>4.0f} {pc_val:>8.4f} {a_reverse:>10.2f}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

观察：a 的值变化很大！

这说明 a 确实可能依赖于 z, d, g，不只是常数。

20.3 分析 a(z, d, g) 的模式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 对于所有晶格，计算 a_reverse
a_reverse_list = []
for i in range(len(df)):
    z_val, d_val, g_val, pc_val = z[i], d[i], g[i], pc[i]
    rhs = (z_val - 1) * pc_val - 1
    dim_factor = np.exp(-d_val/3)
    sat_factor = 1 - np.exp(-z_val/9.48)

    if rhs > 0 and sat_factor > 0:
        a_rev = g_val * rhs / (dim_factor * sat_factor)
        a_reverse_list.append((z_val, d_val, g_val, a_rev, names[i]))

a_reverse_list = sorted(a_reverse_list, key=lambda x: x[0])

# 按 z 分组
print(f"a_reverse 按 z 的分布:")
z_unique = sorted(set([x[0] for x in a_reverse_list]))
for z_val in z_unique[:10]:  # 前10个
    a_values = [x[3] for x in a_reverse_list if x[0] == z_val]
    print(f"  z={z_val:>3.0f}: a = {np.mean(a_values):>6.2f} +/- {np.std(a_values):>5.2f} (N={len(a_values)})")

# ============================================================
# 方法 21：大 z 极限
# ============================================================
print("\n" + "=" * 80)
print("方法 21：大 z 极限行为")
print("=" * 80)

print("""
21.1 Bethe 格极限
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当 z -> infinity（或 d -> infinity）：
  p_c -> 1/(z-1)

所以 Delta_R -> 0

我们的公式：
  Delta_R = (a/g) * exp(-d/3) * (1-exp(-z/z0))

对于大 z, d:
  exp(-d/3) -> 0
  所以 Delta_R -> 0

这与 Bethe 极限一致。

21.2 对于有限大 z
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于 z >> z0 ~ 10:
  1 - exp(-z/z0) -> 1
  所以 Delta_R ~ (a/g) * exp(-d/3)

如果 a 是常数，则 Delta_R 与 z 无关（对于大 z）。

从数据检验...
""")

# 对于大 z 的晶格，检查 Delta_R
print(f"\n大 z (z >= 20) 的 Delta_R:")
mask_large_z = z >= 20
for i in np.where(mask_large_z)[0][:15]:
    R = pc[i] * (z[i] - 1)
    Delta_R = R - 1
    print(f"  {names[i]:<25} z={z[i]:>3.0f} d={d[i]:>2.0f} g={g[i]:>2.0f} Delta_R={Delta_R:.4f}")

# 检查 Delta_R 是否与 z 无关
print(f"\n对于 g=4, d>=4 的超立方:")
mask_hc = (g == 4) & (d >= 4)
for i in np.where(mask_hc)[0]:
    R = pc[i] * (z[i] - 1)
    Delta_R = R - 1
    print(f"  d={d[i]:>2.0f} z={z[i]:>3.0f} Delta_R={Delta_R:.4f}")

# ============================================================
# 方法 22：尝试 a = c * z^beta
# ============================================================
print("\n" + "=" * 80)
print("方法 22：尝试 a = c * z^beta")
print("=" * 80)

# 从 a_reverse 数据拟合 a(z)
z_vals = np.array([x[0] for x in a_reverse_list])
a_vals = np.array([x[3] for x in a_reverse_list])

# 拟合 log(a) = log(c) + beta * log(z)
mask_valid = (a_vals > 0) & (z_vals > 0)
log_z = np.log(z_vals[mask_valid])
log_a = np.log(a_vals[mask_valid])

# 线性拟合
coeffs = np.polyfit(log_z, log_a, 1)
beta_fit = coeffs[0]
log_c_fit = coeffs[1]
c_fit = np.exp(log_c_fit)

print(f"\n拟合 a = c * z^beta:")
print(f"  c = {c_fit:.4f}")
print(f"  beta = {beta_fit:.4f}")

# 计算 R^2
pred_log_a = log_c_fit + beta_fit * log_z
ss_res = np.sum((log_a - pred_log_a)**2)
ss_tot = np.sum((log_a - np.mean(log_a))**2)
r2 = 1 - ss_res / ss_tot
print(f"  R^2 = {r2:.4f}")

# 用这个关系计算 MAPE
print(f"\n用 a(z) = {c_fit:.2f} * z^{beta_fit:.2f} 计算:")

def model_a_z(z, d, g):
    a_eff = c_fit * z**beta_fit
    correction = (a_eff / g) * np.exp(-d/3) * (1 - np.exp(-z/9.48))
    return (1 + correction) / (z - 1)

pred_a_z = model_a_z(z, d, g)
mape_a_z = np.mean(np.abs((pc - pred_a_z) / pc)) * 100
print(f"  MAPE = {mape_a_z:.2f}%")

# 与常数 a 对比
pred_const_a = (1 + (14.56 / g) * np.exp(-d/3) * (1 - np.exp(-z/9.48))) / (z - 1)
mape_const = np.mean(np.abs((pc - pred_const_a) / pc)) * 100
print(f"\n对比:")
print(f"  常数 a = 14.56: MAPE = {mape_const:.2f}%")
print(f"  a(z) = c*z^beta: MAPE = {mape_a_z:.2f}%")

# ============================================================
# 方法 23：更一般的 a(z, d, g)
# ============================================================
print("\n" + "=" * 80)
print("方法 23：更一般的 a(z, d, g)")
print("=" * 80)

print("""
23.1 假设
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

a 可能依赖于 z, d, g 的组合：
  a = a_0 * z^alpha_z * d^alpha_d * g^alpha_g

23.2 拟合
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 多元线性回归：log(a) = log(a_0) + alpha_z*log(z) + alpha_d*log(d) + alpha_g*log(g)
d_vals = np.array([x[1] for x in a_reverse_list])
g_vals = np.array([x[2] for x in a_reverse_list])

mask_valid = (a_vals > 0) & (z_vals > 0) & (d_vals > 0) & (g_vals > 0)
log_z = np.log(z_vals[mask_valid])
log_d = np.log(d_vals[mask_valid])
log_g = np.log(g_vals[mask_valid])
log_a = np.log(a_vals[mask_valid])

# 构造设计矩阵
X = np.column_stack([np.ones(len(log_z)), log_z, log_d, log_g])
# 最小二乘
coeffs, _, _, _ = np.linalg.lstsq(X, log_a, rcond=None)
log_a0, alpha_z, alpha_d, alpha_g = coeffs

print(f"拟合结果: log(a) = {log_a0:.4f} + {alpha_z:.4f}*log(z) + {alpha_d:.4f}*log(d) + {alpha_g:.4f}*log(g)")
print(f"  a_0 = {np.exp(log_a0):.4f}")
print(f"  alpha_z = {alpha_z:.4f}")
print(f"  alpha_d = {alpha_d:.4f}")
print(f"  alpha_g = {alpha_g:.4f}")

# 计算 MAPE
def model_general(z, d, g):
    a_eff = np.exp(log_a0) * z**alpha_z * d**alpha_d * g**alpha_g
    correction = (a_eff / g) * np.exp(-d/3) * (1 - np.exp(-z/9.48))
    return (1 + correction) / (z - 1)

pred_general = model_general(z, d, g)
mape_general = np.mean(np.abs((pc - pred_general) / pc)) * 100
print(f"\n用一般模型计算:")
print(f"  MAPE = {mape_general:.2f}%")

# ============================================================
# 方法 24：寻找对偶性或对称性
# ============================================================
print("\n" + "=" * 80)
print("方法 24：寻找对偶性或对称性")
print("=" * 80)

print("""
24.1 2D 格的对偶性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于 2D 格，存在对偶关系：
  p_c(L) + p_c(L*) = 1  (对于键渗流)

对于格点渗流，关系更复杂，但仍有约束。

24.2 检验对偶性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

三角格 (g=3) 和蜂窝格 (g=6) 是互相对偶的。

三角: p_c = 0.5 (精确)
蜂窝: p_c = 0.697

p_c(tri) + p_c(honey) = 0.5 + 0.697 = 1.197 != 1

所以格点渗流没有简单的对偶关系。

24.3 其他对称性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

也许存在某种隐藏的对称性？

例如：a(z, d, g) = a(z', d', g') 对于某些变换？

但没有明显的候选...

24.4 结论
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对偶性方法不太适用。
""")

# ============================================================
# 方法 25：最后的尝试 - 从物理图像构造 a
# ============================================================
print("\n" + "=" * 80)
print("方法 25：从物理图像构造 a")
print("=" * 80)

print("""
25.1 物理图像
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

a 编码了"环路强度"。

物理上，环路强度应该依赖于：
1. 环的数量（与 z, g 相关）
2. 环的"紧密度"（与晶格几何相关）
3. 维度效应（与 d 相关）

25.2 尝试构造
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

假设：
  a = c * N_g * f(compactness)

其中 N_g 是 girth 环的数量，f 是紧密度因子。

对于正则晶格：
  N_g = z(z-1)^{g-2} / (2g)

但这对于小 z 给出很小的 N_g，与 a ~ 14.56 不符...

25.3 另一种构造
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

假设 a 与"有效环数"相关：
  a = c * sum_{l>=g} N_l * w_l

其中 w_l 是长度 l 环的权重。

如果 w_l = w^l（指数衰减）：
  a = c * sum_{l>=g} [z(z-1)^{l-2}/(2l)] * w^l

对于 w < 1/(z-1)，级数收敛。

但这又回到了集团展开的问题...

25.4 结论
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

从物理图像构造 a 仍然极其困难。
需要更高级的数学工具或全新的物理洞察。
""")

# ============================================================
# 最终总结
# ============================================================
print("\n" + "=" * 80)
print("最终总结：25种方法的尝试")
print("=" * 80)

print("""
新尝试的方法：

20. a 作为 z 的函数 -> a 确实变化，但模式复杂
21. 大 z 极限 -> Delta_R 的行为复杂
22. a = c*z^beta -> MAPE 没有明显改善
23. 一般 a(z,d,g) -> 参数太多
24. 对偶性/对称性 -> 不适用
25. 物理图像构造 -> 回到集团展开的困难

总结论：

经过 25 种方法的系统尝试，严格推导 a 仍然是极其困难的。

但是！我们有以下有价值的发现：

1. 理论约束模型 (alpha=1/3, gamma=1) 的 AIC 优于经验模型
2. a 可能不是常数，而是依赖于 z, d, g
3. 需要更复杂的理论框架来推导 a

实际建议：

1. 接受 a 作为拟合参数
2. 但探索 a(z, d, g) 的依赖关系
3. 如果 a 的依赖可以简化，可能改进模型

让我测试一个更灵活的模型...
""")

# 测试更灵活的模型
print("\n测试模型：a 作为 z 的函数")
print("-" * 50)

# 简单模型：a = a_0 + a_1 * z
def model_a_linear(params, z, d, g):
    a0, a1, z0 = params
    a_eff = a0 + a1 * z
    correction = (a_eff / g) * np.exp(-d/3) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_linear = None
best_ss_l = np.inf
for _ in range(2000):
    p0 = [np.random.uniform(0, 20),
          np.random.uniform(-0.5, 1),
          np.random.uniform(5, 20)]
    try:
        res = minimize(lambda p: np.sum((pc - model_a_linear(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000})
        if res.fun < best_ss_l:
            best_ss_l = res.fun
            best_linear = res.x
    except:
        pass

a0_l, a1_l, z0_l = best_linear
pred_linear = model_a_linear(best_linear, z, d, g)
mape_linear = np.mean(np.abs((pc - pred_linear) / pc)) * 100
ssr_linear = np.sum((pc - pred_linear)**2)
aic_linear = len(pc) * np.log(ssr_linear/len(pc)) + 2*3

print(f"线性模型 a(z) = a0 + a1*z:")
print(f"  a0 = {a0_l:.4f}")
print(f"  a1 = {a1_l:.4f}")
print(f"  z0 = {z0_l:.4f}")
print(f"  MAPE = {mape_linear:.2f}%")
print(f"  AIC = {aic_linear:.1f}")

print(f"\n对比:")
print(f"  常数 a: MAPE = {mape_const:.2f}%, AIC = -670.9")
print(f"  线性 a(z): MAPE = {mape_linear:.2f}%, AIC = {aic_linear:.1f}")

if aic_linear < -670.9:
    print(f"  线性模型更优！")
else:
    print(f"  常数模型更优")

print("\n" + "=" * 80)
print("第5部分完成")
print("=" * 80)

# 保存
with open(f'{OUT}/DERIVE_A_PART5.md', 'w', encoding='utf-8') as f:
    f.write("# 从第一原理推导参数 a - 第5部分\n\n")
    f.write("## 尝试的方法\n\n")
    f.write("20. a 作为 z 的函数\n")
    f.write("21. 大 z 极限行为\n")
    f.write("22. a = c*z^beta 模型\n")
    f.write("23. 一般 a(z,d,g) 模型\n")
    f.write("24. 对偶性/对称性\n")
    f.write("25. 物理图像构造\n\n")
    f.write("## 关键发现\n\n")
    f.write(f"线性 a(z) 模型: a0={a0_l:.2f}, a1={a1_l:.2f}, MAPE={mape_linear:.2f}%\n")
    f.write(f"常数 a 模型: MAPE={mape_const:.2f}%\n\n")
    f.write("## 结论\n\n")
    f.write("严格推导 a 仍然是极其困难的。\n")
    f.write("推荐接受 a 作为拟合参数。\n")
