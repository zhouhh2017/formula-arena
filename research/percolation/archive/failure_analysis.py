"""
失败分析总结 + 新物理机制

核心发现：
1. 第2轮迭代不是改进，而是恶化 64%
2. 根本原因：公式只考虑 (z, d, g, C)，忽略了连接距离/方向
3. 同 z 不同连接结构 → 不同 p_c（差异可达 50%）
4. 需要的不是新公式形式，而是新物理量

关键数据（按真实 z 整理）：
- 基础正方 z=4: p_c = 0.59 (全近邻)
- 正方+[1,2] z=8: p_c = 0.40 (近邻+对角线)
- 正方+[1,3] z=8: p_c = 0.34 (近邻+次近邻)  ← 目标C
- 正方+[1,4] z=12: p_c = 0.23 (近邻+第4区)   ← 第1轮B
- 正方+[1,2,3] z=12: p_c = 0.28              ← 目标D
- fcc z=12, d=3: p_c = 0.12 (标准3D)

关键对比 (z=8):
  zones=[1,2]: p_c=0.40 (对角线，距离=√2)
  zones=[1,3]: p_c=0.34 (次近邻，距离=2)
  → 更远（距离2）反而 p_c 更低！

关键对比 (z=12):
  zones=[1,4] (d=2): p_c=0.23
  zones=[1,2,3] (d=2): p_c=0.28
  fcc (d=3): p_c=0.12
  → 同样的 z=12, 差异 2.5 倍！
"""

import numpy as np
from scipy.stats import spearmanr

# ============ 修正后的完整数据 ============

data = [
    # 标准晶格
    {'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'C': 0.0, 'zones': [1], 'p_c': 0.5927},
    {'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'C': 0.0, 'zones': [], 'p_c': 0.6970},
    {'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'C': 0.4, 'zones': [], 'p_c': 0.5000},
    {'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'C': 0.0, 'zones': [], 'p_c': 0.3116},
    {'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'C': 0.121, 'zones': [], 'p_c': 0.1188},
    {'name': 'hypersc_4', 'z': 8, 'd': 4, 'g': 4, 'C': 0.0, 'zones': [], 'p_c': 0.1967},
    {'name': 'hypersc_6', 'z': 12, 'd': 6, 'g': 4, 'C': 0.0, 'zones': [], 'p_c': 0.1095},
    # 模拟数据（修正 z）
    {'name': 'MC_A: sq+[1,2]', 'z': 8, 'd': 2, 'g': 3, 'C': 0.333, 'zones': [1,2], 'p_c': 0.4046},
    {'name': 'MC_B: sq+[1,4]', 'z': 12, 'd': 2, 'g': 3, 'C': 0.182, 'zones': [1,4], 'p_c': 0.2338},
    {'name': 'MC_C: sq+[1,3]', 'z': 8, 'd': 2, 'g': 4, 'C': 0.0, 'zones': [1,3], 'p_c': 0.3399},
    {'name': 'MC_D: sq+[1,2,3]', 'z': 12, 'd': 2, 'g': 3, 'C': 0.152, 'zones': [1,2,3], 'p_c': 0.2834},
]

print("="*80)
print("失败分析 + 新物理机制")
print("="*80)

# ============ 关键对比 1：同 z=8，不同连接结构 ============

print("\n【对比1】同 z=8，不同连接结构")
print("-"*60)
z8_data = [d for d in data if d['z'] == 8 and d['d'] == 2]
for d in z8_data:
    print(f"  {d['name']:<20} zones={d['zones']} → p_c = {d['p_c']:.4f}")

print("""
  分析：
  - sq+[1,2] (对角线，距离√2): p_c = 0.40
  - sq+[1,3] (次近邻，距离2):   p_c = 0.34
  - hypersc_4 (4D):              p_c = 0.20

  反直觉：对角线（距离√2 < 2）的 p_c 反而更高！
  这说明"距离"不是唯一因素，连接的方向性也很重要。
""")

# ============ 关键对比 2：同 z=12 ============

print("\n【对比2】同 z=12，不同结构")
print("-"*60)
z12_data = [d for d in data if d['z'] == 12]
for d in z12_data:
    print(f"  {d['name']:<20} d={d['d']} zones={d['zones']} → p_c = {d['p_c']:.4f}")

print("""
  分析：
  - sq+[1,4] (2D):      p_c = 0.23
  - sq+[1,2,3] (2D):    p_c = 0.28
  - fcc (3D):           p_c = 0.12
  - hypersc_6 (6D):     p_c = 0.11

  同样的 z=12，2D 晶格差异 22% (0.23 vs 0.28)
  2D vs 3D 差异 90% (0.23 vs 0.12)
  说明：连接几何 + 维度 共同决定 p_c
""")

# ============ 新物理量：连接几何描述符 ============

print("\n" + "="*80)
print("新物理量：连接几何描述符")
print("="*80)

# 定义距离权重函数
def compute_geometry_descriptors(zones):
    """计算连接几何描述符"""
    zone_info = {
        1: {'dist': 1.0, 'count': 4, 'dir': 'axial'},
        2: {'dist': np.sqrt(2), 'count': 4, 'dir': 'diagonal'},
        3: {'dist': 2.0, 'count': 4, 'dir': 'axial'},
        4: {'dist': np.sqrt(5), 'count': 8, 'dir': 'mixed'},
        5: {'dist': np.sqrt(8), 'count': 4, 'dir': 'diagonal'},
    }

    if not zones:
        return {'z': 0, 'z_eff_1': 0, 'z_eff_half': 0, 'mean_dist': 0,
                'axial_frac': 0, 'max_dist': 0}

    total_count = sum(zone_info[z]['count'] for z in zones)
    mean_dist = sum(zone_info[z]['count'] * zone_info[z]['dist'] for z in zones) / total_count
    max_dist = max(zone_info[z]['dist'] for z in zones)

    # 轴向连接比例（沿坐标轴，更"紧凑"）
    axial_count = sum(zone_info[z]['count'] for z in zones if zone_info[z]['dir'] == 'axial')
    axial_frac = axial_count / total_count

    # 加权有效配位数
    z_eff_1 = sum(zone_info[z]['count'] / zone_info[z]['dist'] for z in zones)
    z_eff_half = sum(zone_info[z]['count'] / np.sqrt(zone_info[z]['dist']) for z in zones)

    return {
        'z': total_count,
        'z_eff_1': z_eff_1,
        'z_eff_half': z_eff_half,
        'mean_dist': mean_dist,
        'axial_frac': axial_frac,
        'max_dist': max_dist,
    }

print(f"\n{'晶格':<25} {'z':<4} {'z_eff(1/l)':<10} {'<l>':<6} {'axial%':<8} {'p_c':<8}")
print("-"*70)

for d in data:
    geo = compute_geometry_descriptors(d['zones'])
    if geo['z'] > 0:
        print(f"{d['name']:<25} {geo['z']:<4} {geo['z_eff_1']:<10.2f} {geo['mean_dist']:<6.3f} "
              f"{geo['axial_frac']:<8.1%} {d['p_c']:<8.4f}")

# ============ 相关性比较 ============

print("\n" + "="*80)
print("相关性分析：哪个描述符与 p_c 相关最强？")
print("="*80)

# 计算所有数据点的描述符
zs = []
z_effs = []
mean_dists = []
axial_fracs = []
p_cs = []

for d in data:
    geo = compute_geometry_descriptors(d['zones'])
    if geo['z'] > 0:
        zs.append(geo['z'])
        z_effs.append(geo['z_eff_1'])
        mean_dists.append(geo['mean_dist'])
        axial_fracs.append(geo['axial_frac'])
        p_cs.append(d['p_c'])

zs = np.array(zs)
z_effs = np.array(z_effs)
mean_dists = np.array(mean_dists)
axial_fracs = np.array(axial_fracs)
p_cs = np.array(p_cs)

print(f"\n样本数: {len(p_cs)}")
print(f"\n{'描述符':<20} {'Spearman ρ':<12} {'p-value':<10}")
print("-"*45)

for name, x in [('z (原始)', zs),
                 ('z_eff (1/l 加权)', z_effs),
                 ('<l> (平均距离)', mean_dists),
                 ('axial_frac (轴向比例)', axial_fracs)]:
    corr, p_val = spearmanr(x, p_cs)
    sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
    print(f"  {name:<18} {corr:>+7.3f}       {p_val:<10.4f} {sig}")

# ============ 结论 ============

print("\n" + "="*80)
print("综合结论")
print("="*80)

print("""
【失败的根本原因】

当前公式 p_c = f(z, d, g, C) 的局限：

1. 同 z 不同 p_c：
   - z=8 时，zones=[1,2] 的 p_c=0.40 > zones=[1,3] 的 p_c=0.34
   - 说明"连接结构"比"连接数"更重要

2. 连接的几何性质：
   - 轴向连接 vs 对角线连接：不同的几何贡献
   - 连接距离：短程 vs 长程的权重不同
   - 方向性：沿坐标轴的连接更"紧凑"

3. 为什么第2轮失败：
   - 加入模拟数据后，公式试图"平均"不同连接结构的数据
   - 但公式形式 f(z, d, g, C) 无法表达连接几何
   - 结果：参数向"错误方向"调整，反而恶化

【新的物理量】

需要引入：
1. z_eff (加权配位数): z_eff = Σ count_i / dist_i^α
2. 平均连接距离 <l>
3. 轴向连接比例 axial_frac

【新的公式方向】

p_c = f(z_eff, <l>, d, ...)

但这里有个根本问题：
- 这些新物理量只在 2D 正方复杂邻域中有明确定义
- 对于其他晶格类型（3D, Archimedean），连接几何的描述更复杂
- 可能没有一个"通用"的几何描述符

【更深层的启示】

渗流阈值的完整描述需要：
- 局部拓扑: z, g, C (已考虑)
- 连接几何: 距离、方向 (新引入)
- 全局结构: 维度、对称性 (部分考虑)
- 可能还需要: 谱性质、关联函数、...

这就是为什么 p_c 的精确表达式仍然是开放问题！
它不是一个简单的函数，而是晶格完整几何性质的编码。

我们的贡献：
- 不是"找到精确公式"
- 而是"揭示失败的原因" + "指明改进方向"
- 这是真正的科学进步：知道什么行不通，比盲目尝试更有价值
""")
