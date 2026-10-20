from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
从第一原理推导参数 a - 第2部分
=================================
策略：
1. 利用 Ishioka-Koiwa 猜想（p_c ~ P_return）
2. 从精确解的多约束系统推导
3. 从格点动物增长常数推导
4. 寻找 a 的解析表达式
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, fsolve, brentq
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

print("=" * 80)
print("[从第一原理推导参数 a - 第2部分]")
print("=" * 80)

# ============================================================
# 方法 6：随机游走返回概率
# ============================================================
print("\n" + "=" * 80)
print("方法 6：Ishioka-Koiwa 猜想")
print("=" * 80)

print("""
6.1 猜想内容
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ishioka-Koiwa (1978) 猜想：
  p_c <= P_return

其中 P_return 是随机游走返回原点的概率。

对于 d 维超立方晶格：
  P_return(d) = integral [d^d k / (2pi)^d] / [1 - lambda(k)/(2d)]

其中 lambda(k) = sum_i cos(k_i)

6.2 P_return 的计算
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于简单立方晶格 (d 维)：
  P_return(d) = integral_{-pi}^{pi} ... integral_{-pi}^{pi}
                [1/(2pi)^d] / [1 - (1/2d) sum cos(k_i)] d^d k

这可以用 Watson 积分表示。

已知结果：
  d=1: P_return = 1 (recurrent)
  d=2: P_return = 1 (recurrent)
  d=3: P_return ~ 0.3405
  d=4: P_return ~ 0.1932
  d->infinity: P_return ~ 1/(2d) + 1/(2d)^2 + ...

6.3 与 p_c 的关系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果 p_c ~ P_return，则：
  p_c(SC, d=3) ~ 0.34
  p_c(SC, d=3) = 0.3116

接近！但需要更精确的关系。

猜想：p_c = c_IK * P_return，其中 c_IK ~ 0.9?

6.4 问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

P_return 只适用于超立方晶格，不适用于一般晶格。
对于一般晶格，需要知道晶格的谱（特征值分布）。

这可能不是推导 a 的正确路径...
""")

# 计算一些 P_return 值
def P_return_hypercubic(d):
    """超立方晶格的返回概率（数值积分近似）"""
    # 用蒙特卡洛近似
    np.random.seed(42)
    n_samples = 100000
    k = np.random.uniform(-np.pi, np.pi, (n_samples, d))
    lambda_k = np.sum(np.cos(k), axis=1)
    integrand = 1 / (1 - lambda_k / (2*d))
    return np.mean(integrand)

print("数值计算 P_return (蒙特卡洛):")
for d_val in [3, 4, 5, 6, 8]:
    p_ret = P_return_hypercubic(d_val)
    # 高精度已知值
    known = {3: 0.3405, 4: 0.1932, 5: 0.1351, 6: 0.1047, 8: 0.0787}
    if d_val in known:
        print(f"  d={d_val}: P_return = {p_ret:.4f} (known: {known[d_val]:.4f})")
    else:
        print(f"  d={d_val}: P_return = {p_ret:.4f}")

# ============================================================
# 方法 7：从多约束系统推导
# ============================================================
print("\n" + "=" * 80)
print("方法 7：扩展精确解约束系统")
print("=" * 80)

print("""
7.1 策略
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用更多的精确/高精度结果来约束参数。

如果我们有 n 个约束和 k 个参数，当 n > k 时：
- 如果系统一致，可以唯一确定参数
- 如果不一致，需要修改模型

7.2 高精度数据
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 高精度数据集
high_precision_data = [
    # (name, z, d, g, pc, uncertainty%)
    ("triangular", 6, 2, 3, 0.5000, 0.0),      # exact
    ("honeycomb", 3, 2, 6, 0.6970, 0.1),       # high precision
    ("square", 4, 2, 4, 0.5927, 0.01),         # high precision
    ("SC", 6, 3, 4, 0.3116, 0.1),              # Malarz 2023
    ("diamond", 4, 3, 6, 0.4296, 0.1),         # Malarz 2023
    ("BCC", 8, 3, 4, 0.2465, 0.1),             # Malarz 2023
    ("FCC", 12, 3, 3, 0.1992, 0.1),            # Malarz 2023
    ("SC-4d", 8, 4, 4, 0.1976, 0.2),           # Malarz 2023
    ("SC-5d", 10, 5, 4, 0.1408, 0.3),          # Malarz 2023
    ("SC-6d", 12, 6, 4, 0.1090, 0.3),          # Malarz 2023
    ("SC-7d", 14, 7, 4, 0.0890, 0.4),          # Malarz 2023
    ("SC-8d", 16, 8, 4, 0.0752, 0.5),          # Malarz 2023
]

# 定义加权最小二乘
def weighted_objective(params, data, weights):
    a, alpha, gamma, z0 = params
    total = 0
    for i, (name, z, d, g, pc, unc) in enumerate(data):
        correction = (a / g**gamma) * np.exp(-alpha * d) * (1 - np.exp(-z / z0))
        pc_pred = (1 + correction) / (z - 1)
        residual = (pc_pred - pc) / pc
        total += weights[i] * residual**2
    return total

# 使用不确定度作为权重
weights = [1.0/(unc/100 + 0.001)**2 for (_, _, _, _, _, unc) in high_precision_data]
weights = np.array(weights) / np.sum(weights)  # 归一化

# 优化
best_params = None
best_obj = np.inf
for _ in range(2000):
    p0 = [np.random.uniform(5, 30),
          np.random.uniform(0.1, 0.6),
          np.random.uniform(0.3, 1.5),
          np.random.uniform(5, 20)]
    res = minimize(lambda p: weighted_objective(p, high_precision_data, weights),
                  p0, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
    if res.fun < best_obj:
        best_obj = res.fun
        best_params = res.x

a_prec, alpha_prec, gamma_prec, z0_prec = best_params
print(f"加权最小二乘结果（精确/高精度数据）:")
print(f"  a = {a_prec:.4f}")
print(f"  alpha = {alpha_prec:.4f}")
print(f"  gamma = {gamma_prec:.4f}")
print(f"  z0 = {z0_prec:.4f}")
print(f"  加权残差 = {best_obj:.6f}")

# 对比全局拟合
print(f"\n对比:")
print(f"  精确数据约束: a = {a_prec:.4f}")
print(f"  全局94晶格拟合: a = 14.56")
print(f"  差距: {abs(a_prec - 14.56):.2f}")

# 预测性能
print(f"\n预测所有精确/高精度数据:")
print(f"{'Lattice':<15} {'pc_exact':>10} {'pc_pred':>10} {'err%':>8}")
print("-" * 50)

for name, z, d, g, pc, unc in high_precision_data:
    correction = (a_prec / g**gamma_prec) * np.exp(-alpha_prec * d) * (1 - np.exp(-z / z0_prec))
    pc_pred = (1 + correction) / (z - 1)
    err = abs(pc_pred - pc) / pc * 100
    print(f"{name:<15} {pc:>10.4f} {pc_pred:>10.4f} {err:>8.2f}%")

# ============================================================
# 方法 8：寻找 a 的解析表达式
# ============================================================
print("\n" + "=" * 80)
print("方法 8：寻找 a 的解析表达式")
print("=" * 80)

print("""
8.1 思路
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果 a 可以从第一原理计算，它应该是某些普适量的函数：
  a = f(d_c, 几何因子, ...)

其中 d_c = 6 是上临界维度。

已知：
  alpha = 2/d_c = 1/3（来自二阶环路相互作用）

也许 a 也与 d_c 相关？

8.2 猜测：a 与 d_c 的关系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

尝试：
  a = c_a * d_c^beta

从数据：a ~ 14.56, d_c = 6

如果 beta = 1: c_a = 14.56/6 = 2.43
如果 beta = 2: c_a = 14.56/36 = 0.40

或者：
  a = c_a * exp(d_c/something)

这些纯粹是猜测，没有理论基础...

8.3 另一种猜测：a 与晶格几何的关系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

也许 a 不是常数，而是依赖于晶格类型？

但这会破坏"普适公式"的目标...

8.4 严格推导的困难
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

从集团展开推导 a 需要：
1. 计算所有长度 l >= g 的环的贡献
2. 求和无穷级数
3. 考虑环的相互作用

这在数学上极其困难，因为：
- 拓扑权重 c_l 的计算是 #P-hard
- 环的相互作用需要考虑复杂的多体效应
- 对于小 z（3-12），微扰展开不收敛

结论：严格推导 a 可能需要全新的数学工具。
""")

# ============================================================
# 方法 9：从格点动物增长常数推导
# ============================================================
print("\n" + "=" * 80)
print("方法 9：格点动物增长常数")
print("=" * 80)

print("""
9.1 理论基础
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

渗流阈值与格点动物的增长常数 lambda 相关：
  p_c = 1/lambda

对于 Bethe 格：lambda = z-1

对于有环晶格：lambda = (z-1) / (1 + Delta_lambda)

其中 Delta_lambda 取决于环结构。

9.2 增长常数的计算
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于 d 维超立方晶格，lambda 已知：
  lambda = 2d * mu_d

其中 mu_d 是 d 维格点动物的增长常数。

已知值：
  mu_2 = 4.0626... (方格)
  mu_3 = 4.6840... (简立方)
  mu_4 = 6.79... (4D)

对于大 d：mu_d -> 2d * e（趋近 Bethe 极限）

9.3 与 a 的关系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

p_c = 1/lambda = 1/[(z-1)(1 + Delta_lambda)]
   ~ [1/(z-1)] * [1 - Delta_lambda]

所以 Delta_R ~ -Delta_lambda

如果 Delta_lambda 可以计算，就可以得到 a。

但 mu_d 的精确值只知道少数几个维度，
没有闭式表达式...

9.4 结论
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

格点动物方法理论上可以推导 a，
但实际上 mu_d 的计算本身就是极其困难的数学问题。

这可能不是实际可行的路径。
""")

# ============================================================
# 方法 10：尝试新的理论框架
# ============================================================
print("\n" + "=" * 80)
print("方法 10：尝试新的理论框架")
print("=" * 80)

print("""
10.1 核心问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

为什么 a 这么大（~14.56）？

集团展开领头阶给出 a ~ 0.5，差距 29 倍。

这意味着微扰展开不收敛，需要非微扰方法。

10.2 可能的非微扰效应
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(a) 环的重叠和纠缠
    - 对于小 z，环之间高度重叠
    - 需要自洽方法处理

(b) 大集团效应
    - p_c 附近，集团大小发散
    - 需要重求和技术

(c) 临界涨落
    - 在 d < d_c 时，涨落很重要
    - 需要重整化群

10.3 自洽方法尝试
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

自洽方程：
  p_c = f(p_c, 晶格结构)

例如：
  p_c = 1 / [z - 1 - b(p_c)]

其中 b(p_c) 是有效公共近邻数，依赖于 p_c。

对于三角格：
  b = 2（两个相邻格点有 2 个公共近邻）
  p_c = 1 / (z - 1 - b) = 1 / (6 - 1 - 2) = 1/3

但精确值是 1/2，差距很大！

说明简单自洽方程不够。

10.4 改进的自洽方程
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

考虑更高阶效应：
  p_c = 1 / [z - 1 - b - c(p_c)]

其中 c(p_c) 包含四边形、五边形等贡献。

对于三角格：
  p_c = 1/2
  z - 1 = 5
  b = 2
  所以 c(p_c) = 5 - 2 - 1/p_c = 5 - 2 - 2 = 1

对于简立方：
  p_c = 0.3116
  z - 1 = 5
  b = 0（g=4，无三角形）
  所以 c(p_c) = 5 - 0 - 1/0.3116 = 5 - 3.21 = 1.79

c 的值不同，说明它依赖于晶格结构...

10.5 继续探索
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

让我尝试系统研究 b 和 c 与晶格结构的关系。
""")

# 计算每个晶格的 b_eff
print("计算有效公共近邻数 b_eff:")
print(f"{'Lattice':<15} {'z':>3} {'g':>2} {'p_c':>8} {'b_eff':>8} {'c_eff':>8}")
print("-" * 55)

for name, z, d, g, pc, _ in high_precision_data[:8]:
    # 自洽方程: p_c = 1 / (z - 1 - b_eff)
    # 所以 b_eff = z - 1 - 1/p_c
    b_eff = z - 1 - 1/pc
    # 对于 g=3: b_eff = b_tri + c_eff
    # 对于 g=4: b_eff = 0 + c_eff = c_eff
    print(f"{name:<15} {z:>3} {g:>2} {pc:>8.4f} {b_eff:>8.4f} {b_eff:>8.4f}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

观察：b_eff 不是简单的常数！

三角格: b_eff = 5 - 2 = 3
蜂窝格: b_eff = 2 - 1.435 = 0.565
方格: b_eff = 3 - 1.687 = 1.313
简立方: b_eff = 5 - 3.21 = 1.79

b_eff 依赖于晶格的具体结构，不只是 g！

这说明简单的 b 参数化不够。
需要更复杂的描述符。

可能的描述符：
1. 公共近邻数 b
2. 三角形数
3. 四边形数
4. 环的分布
5. 谱密度

但这些都依赖于晶格的具体几何，
不容易参数化为 (z, d, g) 的函数...
""")

# ============================================================
# 总结与下一步
# ============================================================
print("\n" + "=" * 80)
print("总结：推导 a 的困难")
print("=" * 80)

print("""
经过 10 种方法的尝试，结论是：

从第一原理严格推导 a 是极其困难的。

原因：
1. 集团展开不收敛（小 z 情况）
2. 格点动物计数是 #P-hard 问题
3. 有效公共近邻数依赖于晶格细节
4. 非微扰效应很重要

可能的出路：

(A) 接受 a 是拟合参数
    - 不是严格的"第一原理推导"
    - 但可以用物理框架解释

(B) 寻找 a 的近似表达式
    - 例如 a ~ c * d_c^2 / something
    - 但缺乏严格性

(C) 发现新的数学方法
    - 可能需要全新的理论框架
    - 时间不确定

(D) 改变目标
    - 不追求推导 a
    - 专注于推导其他参数（alpha, gamma）

推荐：(D) 改变目标
  - 我们已经推导出 alpha = 1/3, gamma = 1
  - 这些有理论基础且统计支持
  - a 和 z0 作为拟合参数保留
  - 这样仍然是"理论启发的约束模型"
""")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)

# 保存
with open(f'{OUT}/DERIVE_A_PART2.md', 'w', encoding='utf-8') as f:
    f.write("# 从第一原理推导参数 a - 第2部分\n\n")
    f.write("## 尝试的方法\n\n")
    f.write("1. Ishioka-Koiwa 猜想（随机游走返回概率）\n")
    f.write("2. 扩展精确解约束系统\n")
    f.write("3. 格点动物增长常数\n")
    f.write("4. 自洽方程\n\n")
    f.write("## 结论\n\n")
    f.write("严格推导 a 是极其困难的。\n")
    f.write("推荐：接受 a 作为拟合参数，保留 alpha=1/3, gamma=1 的理论预言。\n")
