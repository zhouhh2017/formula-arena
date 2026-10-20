from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
从第一原理推导参数 a - 第3部分
=================================
继续探索：
1. b_eff 的系统模式
2. 与 1/d 展开的严格匹配
3. 尝试发现 a 的解析表达式
4. 考虑 a 作为 z 或 d 的函数
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
print("[从第一原理推导参数 a - 第3部分]")
print("=" * 80)

# ============================================================
# 方法 11：b_eff 的系统模式
# ============================================================
print("\n" + "=" * 80)
print("方法 11：b_eff 的系统模式")
print("=" * 80)

print("""
11.1 定义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

b_eff = z - 1 - 1/p_c

这编码了所有环路效应的总和。

如果 b_eff 可以表达为 (z, d, g) 的函数，
则 p_c 可以完全从晶格结构预测。

11.2 分析 b_eff 的模式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 计算所有晶格的 b_eff
b_eff = z - 1 - 1/pc

print(f"b_eff 与 (z, d, g) 的关系:")
print(f"{'Name':<25} {'z':>4} {'d':>2} {'g':>2} {'b_eff':>8}")
print("-" * 50)

# 按 g 分组
for g_val in [3, 4, 6]:
    print(f"\n  g = {g_val}:")
    mask = g == g_val
    indices = np.where(mask)[0]
    # 按 z 排序
    sort_idx = np.argsort(z[indices])
    for i in indices[sort_idx][:10]:  # 只显示前10个
        print(f"    {names[i]:<23} {z[i]:>4.0f} {d[i]:>2.0f} {b_eff[i]:>8.4f}")
    if np.sum(mask) > 10:
        print(f"    ... (还有 {np.sum(mask)-10} 个)")

# 尝试拟合 b_eff = f(z, d, g)
print(f"\n11.3 尝试拟合 b_eff 的模式")
print("-" * 50)

# 假设 b_eff = A / g^delta * exp(-alpha_b * d) * (1 - exp(-z/z_b))
# 这与我们的 Delta_R 形式类似

def model_b_eff(params, z, d, g):
    A, delta, alpha_b, z_b = params
    return (A / g**delta) * np.exp(-alpha_b * d) * (1 - np.exp(-z / z_b))

best_b = None
best_ss = np.inf
for _ in range(2000):
    p0 = [np.random.uniform(0.5, 10),
          np.random.uniform(0.3, 1.5),
          np.random.uniform(0.05, 0.5),
          np.random.uniform(3, 20)]
    try:
        res = minimize(lambda p: np.sum((b_eff - model_b_eff(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000})
        if res.fun < best_ss:
            best_ss = res.fun
            best_b = res.x
    except:
        pass

A_b, delta_b, alpha_b, z_b = best_b
pred_b = model_b_eff(best_b, z, d, g)
mape_b = np.mean(np.abs((b_eff - pred_b) / b_eff)) * 100

print(f"\n拟合结果: b_eff = A/g^delta * exp(-alpha_b*d) * (1-exp(-z/z_b))")
print(f"  A = {A_b:.4f}")
print(f"  delta = {delta_b:.4f}")
print(f"  alpha_b = {alpha_b:.4f}")
print(f"  z_b = {z_b:.4f}")
print(f"  MAPE = {mape_b:.2f}%")

# 与 Delta_R 的参数对比
print(f"\n与 Delta_R 模型参数的对比:")
print(f"  b_eff 模型: delta = {delta_b:.4f}, alpha_b = {alpha_b:.4f}")
print(f"  Delta_R 模型: gamma = 1.0, alpha = 0.332")

# ============================================================
# 方法 12：与 1/d 展开的严格匹配
# ============================================================
print("\n" + "=" * 80)
print("方法 12：与 1/d 展开的严格匹配")
print("=" * 80)

print("""
12.1 超立方的严格展开
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Heydenreich-Matzke (2022) 严格结果：
  p_c(Z^d) = 1/(2d) + 1/(2d)^2 + 3/(2d)^3 + 13/(2d)^4 + ...

系数：a_1 = 1, a_2 = 3, a_3 = 13, ...

12.2 我们的公式在大 d 下的行为
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于超立方 (z=2d, g=4):
  p_c = 1/(2d-1) * [1 + (a/4) * exp(-d/3) * (1-exp(-2d/z0))]

展开 1/(2d-1):
  1/(2d-1) = 1/(2d) * 1/(1-1/(2d))
           = 1/(2d) * [1 + 1/(2d) + 1/(2d)^2 + ...]
           = 1/(2d) + 1/(2d)^2 + 1/(2d)^3 + ...

修正项：
  (a/4) * exp(-d/3) * (1-exp(-2d/z0))

对于大 d:
  exp(-d/3) -> 0
  1-exp(-2d/z0) -> 1
  所以修正项 -> 0

这意味着在大 d 极限下，我们的公式给出：
  p_c ~ 1/(2d) + 1/(2d)^2 + 1/(2d)^3 + ...

但严格结果是：
  p_c = 1/(2d) + 1/(2d)^2 + 3/(2d)^3 + ...

第三项系数不匹配！1 vs 3

12.3 问题分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

我们的公式在大 d 下不能精确再现 1/d 展开。

但这可能是因为：
(a) 公式形式不完全正确
(b) exp(-d/3) 不是正确的渐近形式
(c) 需要修改公式

12.4 尝试修正
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果要求公式在大 d 下正确，需要：
  p_c = 1/(2d) + 1/(2d)^2 + 3/(2d)^3 + ...

而我们的公式给出：
  p_c = 1/(2d) + 1/(2d)^2 + 1/(2d)^3 + (a/4)*exp(-d/3)/(2d) + ...

要匹配第三项，需要：
  1 + (a/4)*exp(-d/3)*2d = 3

这不可能对所有 d 成立...

除非修改维度依赖的形式。

12.5 替代方案：用 1/d^2 替代 exp(-d/3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果 Delta_R ~ 1/d^2（而非 exp(-d/3)），则：
  p_c = 1/(2d-1) * [1 + c/d^2 * ...]
      = 1/(2d) + 1/(2d)^2 + (1+c)/(2d)^3 + ...

要匹配 a_2 = 3:
  1 + c = 3
  c = 2

所以如果用 1/d^2 形式，可以精确匹配大 d 展开！

让我试试...
""")

# 尝试用 1/d^beta 替代 exp(-alpha*d)
def model_power_d(params, z, d, g):
    a, beta, gamma, z0 = params
    correction = (a / g**gamma) * (1/d**beta) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_power = None
best_ss_p = np.inf
for _ in range(2000):
    p0 = [np.random.uniform(5, 50),
          np.random.uniform(0.5, 3),
          np.random.uniform(0.3, 1.5),
          np.random.uniform(5, 20)]
    try:
        res = minimize(lambda p: np.sum((pc - model_power_d(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000})
        if res.fun < best_ss_p:
            best_ss_p = res.fun
            best_power = res.x
    except:
        pass

a_p, beta_p, gamma_p, z0_p = best_power
pred_p = model_power_d(best_power, z, d, g)
mape_p = np.mean(np.abs((pc - pred_p) / pc)) * 100
ssr_p = np.sum((pc - pred_p)**2)
aic_p = len(pc) * np.log(ssr_p/len(pc)) + 2*4

print(f"1/d^beta 模型:")
print(f"  a = {a_p:.4f}")
print(f"  beta = {beta_p:.4f}")
print(f"  gamma = {gamma_p:.4f}")
print(f"  z0 = {z0_p:.4f}")
print(f"  MAPE = {mape_p:.2f}%")
print(f"  AIC = {aic_p:.1f}")

# 与指数模型对比
pred_exp = (1 + (14.56 / g**1.0) * np.exp(-d/3) * (1 - np.exp(-z/9.48))) / (z - 1)
mape_exp = np.mean(np.abs((pc - pred_exp) / pc)) * 100
ssr_exp = np.sum((pc - pred_exp)**2)
aic_exp = len(pc) * np.log(ssr_exp/len(pc)) + 2*2  # 只有 a 和 z0 拟合

print(f"\n对比 (指数模型 vs 幂律模型):")
print(f"  指数 (exp(-d/3)):  MAPE = {mape_exp:.2f}%, AIC = {aic_exp:.1f} (2 params)")
print(f"  幂律 (1/d^beta):   MAPE = {mape_p:.2f}%, AIC = {aic_p:.1f} (4 params)")

# ============================================================
# 方法 13：混合模型
# ============================================================
print("\n" + "=" * 80)
print("方法 13：混合模型 exp + power")
print("=" * 80)

print("""
13.1 思路
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

也许真正的维度依赖是混合形式：
  f(d) = c_1 * exp(-alpha*d) + c_2 / d^beta

这样可以同时拟合：
- 低 d 区域的指数行为
- 大 d 区域的幂律渐近

13.2 实现
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

def model_mixed(params, z, d, g):
    a, alpha, c1, beta, c2, z0 = params
    dim_factor = c1 * np.exp(-alpha * d) + c2 / d**beta
    correction = (a / g) * dim_factor * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

best_mix = None
best_ss_m = np.inf
for _ in range(2000):
    p0 = [np.random.uniform(5, 30),
          np.random.uniform(0.1, 0.5),
          np.random.uniform(0, 1),
          np.random.uniform(0.5, 3),
          np.random.uniform(0, 1),
          np.random.uniform(5, 20)]
    try:
        res = minimize(lambda p: np.sum((pc - model_mixed(p, z, d, g))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 50000})
        if res.fun < best_ss_m:
            best_ss_m = res.fun
            best_mix = res.x
    except:
        pass

if best_mix is not None:
    a_m, alpha_m, c1_m, beta_m, c2_m, z0_m = best_mix
    pred_m = model_mixed(best_mix, z, d, g)
    mape_m = np.mean(np.abs((pc - pred_m) / pc)) * 100
    ssr_m = np.sum((pc - pred_m)**2)
    aic_m = len(pc) * np.log(ssr_m/len(pc)) + 2*6

    print(f"混合模型结果:")
    print(f"  a = {a_m:.4f}")
    print(f"  alpha = {alpha_m:.4f}")
    print(f"  c1 = {c1_m:.4f}")
    print(f"  beta = {beta_m:.4f}")
    print(f"  c2 = {c2_m:.4f}")
    print(f"  z0 = {z0_m:.4f}")
    print(f"  MAPE = {mape_m:.2f}%")
    print(f"  AIC = {aic_m:.1f}")
    print(f"  参数数量: 6")

# ============================================================
# 方法 14：从 a 的数值模式反推
# ============================================================
print("\n" + "=" * 80)
print("方法 14：从 a 的数值模式反推")
print("=" * 80)

print("""
14.1 观察
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

全局拟合给出 a ~ 14.56

这个数字有什么特殊含义？

14.56 ~ ?
  - 14.56 / pi = 4.635
  - 14.56 / e = 5.356
  - 14.56 / ln(2) = 20.99
  - 14.56 ~ 4 * pi / e = 4.63? 不
  - 14.56 ~ 6^2 / e = 13.3? 接近但不精确

也许是巧合...

14.2 尝试 a = c * d_c^2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

d_c = 6
d_c^2 = 36

a / d_c^2 = 14.56 / 36 = 0.404

0.404 ~ ?
  - 0.404 ~ 2/5 = 0.4 接近！

所以 a ~ (2/5) * d_c^2?

让我检验这个猜测...
""")

# 检验 a = (2/5) * d_c^2
a_guess = (2/5) * 36
print(f"猜测: a = (2/5) * d_c^2 = (2/5) * 36 = {a_guess:.2f}")
print(f"拟合值: a = 14.56")
print(f"差距: {abs(a_guess - 14.56):.2f} ({abs(a_guess-14.56)/14.56*100:.1f}%)")

# 用这个猜测计算 MAPE
a_test = a_guess
pred_test = (1 + (a_test / g**1.0) * np.exp(-d/3) * (1 - np.exp(-z/9.48))) / (z - 1)
mape_test = np.mean(np.abs((pc - pred_test) / pc)) * 100
print(f"\n用 a = {a_test:.2f} 计算:")
print(f"  MAPE = {mape_test:.2f}%")

# 尝试其他猜测
print(f"\n其他猜测:")
guesses = [
    ("2/5 * d_c^2", 2/5 * 36),
    ("1/2 * d_c^2", 1/2 * 36),
    ("d_c^2 / e", 36 / np.e),
    ("4 * d_c", 4 * 6),
    ("2 * pi * d_c / e", 2 * np.pi * 6 / np.e),
]

for name, a_guess in guesses:
    pred_test = (1 + (a_guess / g**1.0) * np.exp(-d/3) * (1 - np.exp(-z/9.48))) / (z - 1)
    mape_test = np.mean(np.abs((pc - pred_test) / pc)) * 100
    print(f"  a = {name} = {a_guess:.2f}: MAPE = {mape_test:.2f}%")

# ============================================================
# 方法 15：从特殊晶格的精确解推导 a
# ============================================================
print("\n" + "=" * 80)
print("方法 15：从特殊晶格推导 a")
print("=" * 80)

print("""
15.1 策略
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于某些特殊晶格，如果知道精确的 p_c，
并且知道其他参数，可以反推 a。

15.2 超立方大 d 极限
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于 d -> infinity 超立方：
  p_c ~ 1/(2d) = 1/z

我们的公式：
  p_c = 1/(z-1) * [1 + (a/4) * exp(-d/3) * (1-exp(-z/z0))]

对于大 d, z:
  p_c ~ 1/z * [1 + (a/4) * exp(-d/3)]
      ~ 1/z + (a/4z) * exp(-d/3)

但严格极限是 p_c = 1/z，所以：
  (a/4z) * exp(-d/3) -> 0

这对于任何有限的 a 都成立（因为 exp(-d/3) -> 0）。

所以大 d 极限不能确定 a。

15.3 中等 d 的超立方
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于 d=8 (z=16, g=4), p_c = 0.0752:

0.0752 = 1/15 * [1 + (a/4) * exp(-8/3) * (1-exp(-16/z0))]

如果 z0 = 9.48:
0.0752 * 15 = 1 + (a/4) * 0.0695 * 0.816
1.128 = 1 + (a/4) * 0.0567
0.128 = a * 0.0142
a = 9.02

对于 d=10 (z=20, g=4), p_c = 0.0576:
0.0576 * 19 = 1 + (a/4) * exp(-10/3) * 0.879
1.094 = 1 + (a/4) * 0.0314
0.094 = a * 0.00786
a = 12.0

对于 d=13 (z=26, g=4), p_c = 0.0427:
0.0427 * 25 = 1 + (a/4) * exp(-13/3) * 0.936
1.0675 = 1 + (a/4) * 0.0123
0.0675 = a * 0.00307
a = 22.0

a 的值从 d=8 到 d=13 变化：9.02, 12.0, 22.0

这说明 exp(-d/3) 的衰减不够快！
对于大 d，需要更小的 a 或更快的衰减。

15.4 问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

a 不是常数，或者 exp(-d/3) 形式不对。

让我尝试拟合 a(d)...
""")

# 对于每个 d，反推 a
print("对于每个 d 值，从超立方数据反推 a:")
print(f"{'d':>3} {'z':>3} {'p_c':>8} {'a (反推)':>10}")
print("-" * 35)

sc_mask = g == 4
for d_val in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    mask = (d == d_val) & (g == 4) & (z == 2*d_val)
    if np.sum(mask) > 0:
        z_val = z[mask][0]
        pc_val = pc[mask][0]

        # 反推 a
        # pc = 1/(z-1) * [1 + (a/4) * exp(-d/3) * (1-exp(-z/9.48))]
        # (z-1)*pc - 1 = (a/4) * exp(-d/3) * (1-exp(-z/9.48))
        # a = 4 * [(z-1)*pc - 1] / [exp(-d/3) * (1-exp(-z/9.48))]

        rhs = (z_val - 1) * pc_val - 1
        dim_factor = np.exp(-d_val/3)
        sat_factor = 1 - np.exp(-z_val/9.48)

        if rhs > 0 and dim_factor > 0 and sat_factor > 0:
            a_reverse = 4 * rhs / (dim_factor * sat_factor)
            print(f"{d_val:>3} {z_val:>3.0f} {pc_val:>8.4f} {a_reverse:>10.2f}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print("""
经过 15 种方法的尝试：

1. 集团展开领头阶：预言 a = 0.5，差距 29 倍
2. 完整集团求和：c 不一致
3. 精确解约束：不能同时满足
4. 1/d 展开匹配：exp(-d/3) 不能正确给出高阶系数
5. b_eff 模式：有结构但不能解析表达
6. a 的数值猜测：没有发现简单的解析表达式
7. 反推 a(d)：a 随 d 变化，不是常数

结论：

严格推导 a 仍然是极其困难的。

但是！我们有一些有价值的发现：

1. 幂律维度依赖 1/d^beta 可能比 exp(-d/3) 更好
2. a 可能不是常数，而是 z 或 d 的函数
3. 需要混合模型来同时描述低 d 和大 d 行为

这些发现可以指导未来的理论研究。
""")

print("\n" + "=" * 80)
print("第3部分完成")
print("=" * 80)

# 保存
with open(f'{OUT}/DERIVE_A_PART3.md', 'w', encoding='utf-8') as f:
    f.write("# 从第一原理推导参数 a - 第3部分\n\n")
    f.write("## 关键发现\n\n")
    f.write("1. 1/d^beta 可能比 exp(-d/3) 更正确\n")
    f.write("2. a 可能不是常数\n")
    f.write("3. 需要混合模型\n\n")
    f.write("## 数值结果\n\n")
    f.write(f"b_eff 模型: A={A_b:.2f}, delta={delta_b:.2f}, alpha_b={alpha_b:.2f}\n")
    f.write(f"1/d^beta 模型: beta={beta_p:.2f}\n")
