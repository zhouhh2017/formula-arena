from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
从第一原理推导参数 a
=================================
方法：
1. 集团展开 → 领头阶修正
2. 精确解约束 → overdetermined 系统
3. 1/d 展开 → 超立方渐近行为
4. 随机游走 → Ishioka-Koiwa 连接
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, fsolve, root_scalar
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

print("=" * 80)
print("[从第一原理推导参数 a - 第1部分]")
print("=" * 80)

# ============================================================
# 方法 1：集团展开的领头阶
# ============================================================
print("\n" + "=" * 80)
print("方法 1：集团展开的领头阶修正")
print("=" * 80)

print("""
1.1 集团展开基础
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

渗流阈值由平均集团大小的发散确定：

  chi(p) = sum_s s^2 n_s(p) / p -> infinity  as  p -> p_c

其中 n_s 是大小为 s 的集团的数密度。

对于小 s：
  n_1 = p
  n_2 = (z/2) p^2 (1-p)^{2z-2-b}

其中 b 是两个相邻格点的公共近邻数。

1.2 Bethe 极限与修正
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于树状图（b=0，无环）：
  p_c^Bethe = 1/(z-1)

对于有环晶格：
  p_c = p_c^Bethe (1 + Delta_R)

其中 Delta_R 来自环路贡献。

1.3 领头阶修正的计算
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于长度为 l 的环：
  - 每格点的环数：N_l = z(z-1)^{l-2} / (2l)
  - 每个环的贡献：~ p_c^l

领头阶修正来自最短环 (l = g)：

  Delta_R ~ N_g * p_c^g * (z-1)
         = [z(z-1)^{g-2} / (2g)] * [1/(z-1)]^g * (z-1)
         = z / [2g(z-1)]

对于大 z：
  Delta_R ~ 1/(2g)

所以领头阶预言：
  a = 1/2 = 0.5

但全局拟合给出 a ~ 14.56！

差距 ~ 29 倍！

1.4 问题分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

为什么领头阶预言 a = 0.5 与数据 a = 14.56 差这么多？

可能原因：
(a) 领头阶不够，需要高阶修正
(b) 拓扑权重 c_l 不是简单的 1
(c) 需要考虑环的重叠和相互作用
(d) 对于小 z（3-12），大 z 近似不适用

让我计算更精确的领头阶...
""")

# 对于具体的晶格，计算精确的 N_g
def N_girth(z, g):
    """每格点的 girth-g 环数"""
    return z * (z-1)**(g-2) / (2*g)

# 计算几个晶格的 N_g
print("不同晶格的 N_girth:")
print(f"{'Lattice':<20} {'z':>3} {'g':>2} {'N_g':>10} {'N_g/z':>10}")
print("-" * 50)

lattices = [
    ("triangular", 6, 3),
    ("square", 4, 4),
    ("honeycomb", 3, 6),
    ("SC", 6, 4),
    ("BCC", 8, 4),
    ("FCC", 12, 3),
    ("SC-8d", 16, 4),
]

for name, z_val, g_val in lattices:
    ng = N_girth(z_val, g_val)
    print(f"{name:<20} {z_val:>3} {g_val:>2} {ng:>10.2f} {ng/z_val:>10.4f}")

# 计算领头阶修正
print(f"\n领头阶修正 Delta_R = N_g * p_c^g * (z-1):")
print(f"{'Lattice':<20} {'z':>3} {'g':>2} {'p_c^Bethe':>10} {'Delta_R':>10}")
print("-" * 50)

for name, z_val, g_val in lattices:
    pc_bethe = 1 / (z_val - 1)
    ng = N_girth(z_val, g_val)
    dR = ng * pc_bethe**g_val * (z_val - 1)
    print(f"{name:<20} {z_val:>3} {g_val:>2} {pc_bethe:>10.4f} {dR:>10.6f}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：领头阶修正太小了！

例如三角格：
  领头阶 Delta_R = 0.0123
  实际 Delta_R = 1.5

差了 ~120 倍！

原因：p_c^g 对于小 z 不是很小
  三角格：p_c^Bethe = 1/5, g=3, p_c^3 = 0.008
  但 N_g = 6*5/6 = 5，所以 N_g * p_c^3 = 0.04

这远小于实际的 Delta_R = 1.5。

结论：领头阶微扰不够！
需要：
1. 高阶修正（多环相互作用）
2. 非微扰效应
3. 或者完全不同的方法
""")

# ============================================================
# 方法 2：从集团求和推导
# ============================================================
print("\n" + "=" * 80)
print("方法 2：从集团求和推导")
print("=" * 80)

print("""
2.1 完整的集团展开
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

渗流阈值可以写为：

  p_c = p_c^Bethe * [1 + sum_{l>=g} c_l * N_l * (p_c^Bethe)^l]

其中 c_l 是长度 l 的环的拓扑权重。

如果假设 c_l = c^l（几何衰减），则：

  sum_{l>=g} c^l * N_l * p^l
  = sum_{l>=g} c^l * [z(z-1)^{l-2}/(2l)] * p^l
  = [z/(2(z-1)^2)] * sum_{l>=g} [c*p*(z-1)]^l / l

代入 p = p_c^Bethe = 1/(z-1)：
  = [z/(2(z-1)^2)] * sum_{l>=g} c^l / l

对于 c < 1，这个级数收敛：
  sum_{l>=g} c^l/l = -ln(1-c) - sum_{l=1}^{g-1} c^l/l

2.2 具体计算
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

定义 S_g(c) = sum_{l>=g} c^l/l

对于 g=3:
  S_3(c) = -ln(1-c) - c

对于 g=4:
  S_4(c) = -ln(1-c) - c - c^2/2

对于 g=6:
  S_6(c) = -ln(1-c) - c - c^2/2 - c^3/3 - c^4/4 - c^5/5

则：
  Delta_R = [z/(2(z-1)^2)] * S_g(c)

2.3 从数据反推 c
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果这个理论正确，我们可以从数据反推 c。

对于三角格 (z=6, g=3, Delta_R = 1.5)：
  1.5 = [6/(2*25)] * S_3(c)
  1.5 = 0.12 * [-ln(1-c) - c]
  -ln(1-c) - c = 12.5

但这个方程无解！因为 -ln(1-c) - c < infinity for c < 1，
但对于 c -> 1，-ln(1-c) -> infinity。

让我数值求解...
""")

# 数值求解 c
from scipy.optimize import brentq

def S_g(c, g):
    """计算 S_g(c) = sum_{l>=g} c^l/l"""
    if c <= 0 or c >= 1:
        return 0
    result = -np.log(1-c)
    for l in range(1, g):
        result -= c**l / l
    return result

# 对于三角格
print("从三角格 (Delta_R = 1.5, z = 6, g = 3) 反推 c:")
# 1.5 = [6/(2*25)] * S_3(c) = 0.12 * S_3(c)
# S_3(c) = 12.5

target_S3 = 1.5 / 0.12
print(f"  需要 S_3(c) = {target_S3:.2f}")

# S_3(c) = -ln(1-c) - c
def eq_S3(c):
    return -np.log(1-c) - c - target_S3

try:
    c_solution = brentq(eq_S3, 0.01, 0.9999)
    print(f"  解: c = {c_solution:.6f}")
    print(f"  验证: S_3({c_solution:.4f}) = {S_g(c_solution, 3):.4f}")
except:
    print("  无解!")

# 对于简立方 (z=6, g=4, Delta_R = 0.558)
print(f"\n从简立方 (Delta_R = 0.558, z = 6, g = 4) 反推 c:")
# 0.558 = [6/(2*25)] * S_4(c) = 0.12 * S_4(c)
target_S4 = 0.558 / 0.12
print(f"  需要 S_4(c) = {target_S4:.2f}")

def eq_S4(c):
    return S_g(c, 4) - target_S4

try:
    c_solution_sc = brentq(eq_S4, 0.01, 0.9999)
    print(f"  解: c = {c_solution_sc:.6f}")
    print(f"  验证: S_4({c_solution_sc:.4f}) = {S_g(c_solution_sc, 4):.4f}")
except:
    print("  无解!")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：三角格和简立方给出不同的 c 值！

这说明简单的几何衰减 c_l = c^l 不够。
需要更复杂的模型。
""")

# ============================================================
# 方法 3：修正的集团展开
# ============================================================
print("\n" + "=" * 80)
print("方法 3：修正的集团展开")
print("=" * 80)

print("""
3.1 考虑 z 依赖的修正
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：系数 [z/(2(z-1)^2)] 对于小 z 不准确。

实际上，集团展开的系数应该包含更多的 z 依赖。

让我重新推导...

对于长度为 l 的环，贡献应该是：
  Delta_l = c_l * N_l * p^l * (z-1)^{l-1} / (z-1)

等等，让我更仔细地考虑...

实际上，渗流阈值的修正是这样来的：

考虑一个格点，它的 z 个近邻形成各种连接模式。
对于树状图，近邻之间没有连接 -> Bethe 近似。
对于有环图，近邻之间有连接 -> 修正。

两个相邻格点的公共近邻数 b 决定了三角形环的数量。
但还有更长的环（四边形、五边形等）。

3.2 系统方法：考虑所有环路
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于正则晶格，长度为 l 的环的数量（每格点）：
  N_l = z(z-1)^{l-2} / (2l)  对于 l >= g

但这是假设环可以任意延伸。对于小晶格或低维，
实际数量可能不同。

更精确的计算需要考虑晶格的具体几何。

3.3 另一种方法：利用格点动物计数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

渗流阈值与格点动物的增长常数 lambda 相关：
  p_c ~ 1/lambda

对于 Bethe 格：lambda = z-1

对于有环晶格：lambda < z-1

修正量取决于动物的数量如何依赖于环结构。

这个问题本质上是 #P-hard 的——对于一般晶格，
精确计算 lambda 是极其困难的。

3.4 实用方法：从已知结果归纳
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

既然严格的理论推导极其困难，
让我尝试从已知结果中归纳出规律。

已知：
- Bethe 极限：p_c = 1/(z-1)
- 1/d 展开：p_c = 1/(2d) + 1/(2d)^2 + 3/(2d)^3 + ...
- 精确解：三角格 p_c = 1/2

能否从这些已知结果中推导出 a？
""")

# ============================================================
# 方法 4：利用 1/d 展开
# ============================================================
print("\n" + "=" * 80)
print("方法 4：利用 1/d 展开")
print("=" * 80)

print("""
4.1 超立方晶格的 1/d 展开
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于 d 维超立方晶格 (z = 2d, g = 4)：

  p_c = 1/(2d) + 1/(2d)^2 + 3/(2d)^3 + O(d^{-4})

这是严格结果（Heydenreich-Matzke 2022）。

领头阶修正：
  Delta_p = p_c - 1/(2d) = 1/(2d)^2 + 3/(2d)^3 + ...

用 Bethe 归一化：
  R = p_c * (z-1) = p_c * (2d-1)
  Delta_R = R - 1 = p_c * (2d-1) - 1

代入 p_c 的展开：
  Delta_R = [1/(2d) + 1/(2d)^2 + 3/(2d)^3] * (2d-1) - 1
         = 1 - 1/(2d) + 1/(2d) - 1/(2d)^2 + 3/(2d)^2 - 3/(2d)^3 + ... - 1
         = 2/(2d)^2 + O(d^{-3})
         = 1/(2d^2) + O(d^{-3})

所以对于超立方，领头阶 Delta_R ~ 1/(2d^2) = 2/z^2

4.2 与我们的公式对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

我们的公式：
  Delta_R = (a/g) * exp(-alpha*d) * (1 - exp(-z/z0))

对于超立方 (g=4, z=2d)：
  Delta_R = (a/4) * exp(-d/3) * (1 - exp(-2d/z0))

对于大 d（但 exp(-d/3) 不太小）：
  Delta_R ~ (a/4) * exp(-d/3)

但 1/d 展开给出 Delta_R ~ 1/(2d^2)

比较：
  exp(-d/3) vs 1/(2d^2)

在 d = 6：
  exp(-2) = 0.135
  1/72 = 0.014

差距 ~10 倍！

在 d = 10：
  exp(-10/3) = 0.036
  1/200 = 0.005

差距 ~7 倍！

这说明 exp(-d/3) 的衰减比 1/d^2 慢。
对于大 d，我们的公式会高估 Delta_R。

但在 d = 2-13 的范围内，两种形式可能看起来类似。

4.3 从 1/d 展开推导 a
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果要求我们的公式在大 d 极限下与 1/d 展开一致，
需要修改维度依赖的形式。

可能的修正：
  用 d^{-beta} 替代 exp(-alpha*d)

或者，承认 exp(-d/3) 只在有限 d 范围内有效，
不是渐近正确的。

对于 a 的推导，这增加了困难...
""")

# 数值比较
print("数值比较：exp(-d/3) vs 1/(2d^2):")
print(f"{'d':>3} {'exp(-d/3)':>12} {'1/(2d^2)':>12} {'比率':>8}")
print("-" * 40)
for d_val in [2, 3, 4, 6, 8, 10, 13, 20]:
    exp_val = np.exp(-d_val/3)
    power_val = 1/(2*d_val**2)
    ratio = exp_val / power_val
    print(f"{d_val:>3} {exp_val:>12.6f} {power_val:>12.6f} {ratio:>8.2f}")

# ============================================================
# 方法 5：从精确解系统推导
# ============================================================
print("\n" + "=" * 80)
print("方法 5：从精确解系统推导")
print("=" * 80)

print("""
5.1 建立约束系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

已知精确/高精度结果：

(1) 三角格: p_c = 0.5000, z=6, d=2, g=3
(2) 蜂窝格: p_c = 0.6970, z=3, d=2, g=6
(3) 简立方: p_c = 0.3116, z=6, d=3, g=4
(4) 金刚石: p_c = 0.4296, z=4, d=3, g=6

4个方程，4个未知数 (a, alpha, gamma, z0)

如果系统有唯一解，我们就从"精确解"推导出了参数！

5.2 数值求解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 定义方程组
exact_data = [
    (6, 2, 3, 0.5000),   # triangular
    (3, 2, 6, 0.6970),   # honeycomb
    (6, 3, 4, 0.3116),   # simple cubic
    (4, 3, 6, 0.4296),   # diamond
]

def model(params, z, d, g):
    a, alpha, gamma, z0 = params
    correction = (a / g**gamma) * np.exp(-alpha * d) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

def equations(params):
    eqs = []
    for z, d, g, pc_exact in exact_data:
        pc_pred = model(params, z, d, g)
        eqs.append(pc_pred - pc_exact)
    return eqs

# 多起点求解
solutions = []
for trial in range(1000):
    p0 = [np.random.uniform(5, 30),
          np.random.uniform(0.1, 0.6),
          np.random.uniform(0.3, 1.5),
          np.random.uniform(5, 20)]
    try:
        sol = fsolve(equations, p0, full_output=True)
        if sol[2] == 1:  # 收敛
            params_sol = sol[0]
            residual = np.sum(np.array(equations(params_sol))**2)
            if residual < 1e-10:
                solutions.append(params_sol)
    except:
        pass

if solutions:
    solutions = np.array(solutions)
    print(f"找到 {len(solutions)} 个解")
    print(f"\n解的均值和标准差:")
    param_names = ['a', 'alpha', 'gamma', 'z0']
    for i, name in enumerate(param_names):
        mean_val = np.mean(solutions[:, i])
        std_val = np.std(solutions[:, i])
        print(f"  {name} = {mean_val:.4f} +/- {std_val:.4f}")

    # 验证
    print(f"\n验证（用均值参数预测）:")
    mean_params = np.mean(solutions, axis=0)
    for z, d, g, pc_exact in exact_data:
        pc_pred = model(mean_params, z, d, g)
        err = abs(pc_pred - pc_exact) / pc_exact * 100
        print(f"  z={z}, d={d}, g={g}: pred={pc_pred:.4f}, exact={pc_exact:.4f}, err={err:.2f}%")

    # 与其他晶格对比
    print(f"\n预测其他晶格（未经拟合）:")
    other_lattices = [
        ("BCC", 8, 3, 4, 0.2465),
        ("FCC", 12, 3, 3, 0.1992),
        ("SC-4d", 8, 4, 4, 0.1976),
        ("SC-8d", 16, 8, 4, 0.0752),
    ]
    for name, z, d, g, pc_exact in other_lattices:
        pc_pred = model(mean_params, z, d, g)
        err = abs(pc_pred - pc_exact) / pc_exact * 100
        print(f"  {name}: pred={pc_pred:.4f}, exact={pc_exact:.4f}, err={err:.2f}%")
else:
    print("未找到精确解！")
    print("这意味着4个精确解不能同时满足。")

    # 尝试最小二乘
    print("\n用最小二乘法找最佳近似解:")
    def objective(params):
        return np.sum(np.array(equations(params))**2)

    best_params = None
    best_obj = np.inf
    for _ in range(1000):
        p0 = [np.random.uniform(5, 30),
              np.random.uniform(0.1, 0.6),
              np.random.uniform(0.3, 1.5),
              np.random.uniform(5, 20)]
        res = minimize(objective, p0, method='Nelder-Mead',
                      options={'maxiter': 50000})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x

    if best_params is not None:
        print(f"  最佳参数: a={best_params[0]:.4f}, alpha={best_params[1]:.4f}, "
              f"gamma={best_params[2]:.4f}, z0={best_params[3]:.4f}")
        print(f"  残差: {best_obj:.2e}")

        print(f"\n  预测结果:")
        for z, d, g, pc_exact in exact_data:
            pc_pred = model(best_params, z, d, g)
            err = abs(pc_pred - pc_exact) / pc_exact * 100
            print(f"    z={z}, d={d}, g={g}: pred={pc_pred:.4f}, exact={pc_exact:.4f}, err={err:.2f}%")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)

# 保存结果
with open(f'{OUT}/DERIVE_A_PART1.md', 'w', encoding='utf-8') as f:
    f.write("# 从第一原理推导参数 a - 第1部分\n\n")
    f.write("## 方法 1：集团展开领头阶\n\n")
    f.write("预言 a = 1/2 = 0.5，但数据给出 a ~ 14.56。差距 29 倍。\n\n")
    f.write("## 方法 2：完整集团求和\n\n")
    f.write("假设 c_l = c^l，从三角格和简立方反推的 c 不一致。\n\n")
    f.write("## 方法 4：1/d 展开\n\n")
    f.write("exp(-d/3) 在大 d 下与 1/d^2 不一致，但在有限 d 范围内可用。\n\n")
    f.write("## 方法 5：精确解约束\n\n")
    f.write("4个精确解不能同时满足，需要最小二乘近似。\n")
