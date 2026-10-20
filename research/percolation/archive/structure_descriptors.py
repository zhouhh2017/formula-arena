from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
晶格结构描述符计算
===================
  计算所有晶格的:
  - n_3: 每个原子的三角形面数 (triangles per site)
  - n_4: 每个原子的四边形面数
  - n_tri_per_bond: 每条键参与的三角形数

  核心假说: FCC/HCP的异常源于短程环密度不同
"""

import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

print("=" * 75)
print("【晶格结构描述符计算】")
print("=" * 75)

# ============================================================
# 1. 构建各晶格的邻接关系并计算 n_3
# ============================================================

def count_triangles_per_site(neighbors_of):
    """
    计算每个原子的三角形面数。
    neighbors_of: dict {site: set of neighbor sites}
    返回: 每个site参与的三角形数 (取平均)
    """
    n_tri_total = 0
    for site, nbrs in neighbors_of.items():
        for j, k in combinations(nbrs, 2):
            if k in neighbors_of.get(j, set()) or j in neighbors_of.get(k, set()):
                n_tri_total += 1
    # 每个三角形被3个顶点各数一次
    n_sites = len(neighbors_of)
    return 3 * n_tri_total / n_sites if n_sites > 0 else 0

def count_4cycles_per_site(neighbors_of):
    """计算每个原子参与的四边形数。"""
    n_4_total = 0
    for site, nbrs in neighbors_of.items():
        for j, k in combinations(nbrs, 2):
            # j-k之间没有直接键（否则是三角形）
            if k not in neighbors_of.get(j, set()) and j not in neighbors_of.get(k, set()):
                # 检查j和k是否有公共邻居（除了site之外）
                common = set(neighbors_of.get(j, set())) & set(neighbors_of.get(k, set())) - {site}
                n_4_total += len(common)
    # 每个四边形被4个顶点各数一次，每个顶点数的次数需要校正
    n_sites = len(neighbors_of)
    return 4 * n_4_total / n_sites if n_sites > 0 else 0

# ============================================================
# 2. 经典晶格的解析计算
# ============================================================

print(f"\n【经典晶格 — 解析计算】")
print(f"\n{'晶格':<15} {'z':>3} {'d':>2} {'g':>2} {'n_3':>5} {'n_4':>5} {'p_c':>8}")
print("-" * 50)

classic_data = []

# --- 2D ---

# Honeycomb (z=3): 六边形网格，无三角形
# 每个原子的3个邻居之间无键
classic_data.append({'name': 'honeycomb', 'z': 3, 'd': 2, 'g': 6, 'n_3': 0, 'n_4': 0, 'pc': 0.697040})

# Square (z=4): 正方形网格，无三角形
# 每个原子的4个邻居形成正方形，但无对角键
classic_data.append({'name': 'square', 'z': 4, 'd': 2, 'g': 4, 'n_3': 0, 'n_4': 4, 'pc': 0.592746})

# Triangular (z=6): 三角网格
# 每个原子被6个三角形包围
# 每个邻居与2个其他邻居有键 → n_3 = 6×2/2 = 6
classic_data.append({'name': 'triangular', 'z': 6, 'd': 2, 'g': 3, 'n_3': 6, 'n_4': 0, 'pc': 0.500000})

# Kagome (z=4): 角 sharing 三角形
# 每个原子在2个三角形中
# 4个邻居中，2对之间有键
classic_data.append({'name': 'kagome', 'z': 4, 'd': 2, 'g': 3, 'n_3': 2, 'n_4': 0, 'pc': 0.652703})

# --- 3D ---

# SC (z=6, g=4): 简单立方
# 邻居在±x,±y,±z方向，无三角形
# 每个原子在多个正方形中：3个平面×每个平面1个正方形 = 3个
# 但每个邻居与其他邻居的正方形关系：
# (1,0,0)与(0,1,0)有公共邻居(1,1,0)？不，(1,1,0)不是SC的近邻
# SC的n_4: 每个原子在3个正方形中(每个坐标平面1个)
# 不对，应该是每对相邻轴方向形成1个正方形
# n_4 = 选择2个轴方向 × 每个方向1个 = C(3,2) = 3... 不对
# 每个原子参与的4-环数：
# 从原点出发，沿+x, +y, -(x), -(y) 形成一个4-环
# 这样的环有 C(3,2) × 4 = 12？
# 实际上: 从原点看，n_4 = 每对轴方向形成1个正方形面 = 3个（xy, xz, yz平面）
# 但每个原子被更多正方形包围
# 在SC中，每个原子是8个立方体的角 → 8×3 = 24个正方形面共享这个角
# 每个面被4个原子共享 → 每个原子"拥有" 24/4 = 6个正方形
# 不对，让我用更系统的方法

# SC中: 邻居 = {(±1,0,0), (0,±1,0), (0,0,±1)}
# 4-环需要: i→j→k→l→i 其中每步都是近邻
# 从原点(0,0,0)参与的4-环：
# (0,0,0)→(1,0,0)→(1,1,0)→(0,1,0)→(0,0,0) ✓ (但(1,1,0)不是原点邻居)
# 等等，我需要的是包含原点的4-环
# 原点→邻居1→邻居2→邻居3→原点
# 邻居1和邻居3都必须是原点的邻居
# 邻居1和邻居2之间必须有键
# 邻居2和邻居3之间必须有键

# 对于SC:
# (1,0,0)和(0,1,0)之间的距离 = √2 → 不是近邻 → 不能形成4-环经过这两步
# 所以SC中，原点的两个邻居(1,0,0)和(0,1,0)之间有公共近邻(1,1,0)?
# (1,1,0)到(1,0,0)距离=1 ✓, (1,1,0)到(0,1,0)距离=1 ✓
# 但(1,1,0)到原点距离=√2 → 不是原点的近邻
# 所以4-环: 原点→(1,0,0)→(1,1,0)→(0,1,0)→原点
# 长度: 1+1+1+1 = 4步，但(1,1,0)不是原点的近邻 → 最后一步距离=1 ✓

# 好，所以SC中有4-环。让我计数。
# 原点参与的4-环：选择两个轴方向(如x,y)，然后有两种走法
# (+x → +x+y → +y) 或 (+x → +x-y → -y) 等
# 对于每对轴(x,y), (x,z), (y,z)，有4个4-环
# n_4 = 3 × 4 = 12?

# 让我用程序方法计算
sc_nbrs = {}
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if abs(dx)+abs(dy)+abs(dz) <= 1:
                site = (dx, dy, dz)
                nbrs = set()
                for ddx, ddy, ddz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nbrs.add((dx+ddx, dy+ddy, dz+ddz))
                sc_nbrs[site] = nbrs

sc_n3 = count_triangles_per_site(sc_nbrs)
sc_n4 = count_4cycles_per_site(sc_nbrs)
# 修正：count_4cycles可能有误，让我手动验证
# 对于SC中心原子(0,0,0)，邻居有6个
# 找4-环: (0,0,0)→n1→m→n2→(0,0,0)
# n1, n2是(0,0,0)的邻居, m是n1和n2的共同邻居
# 对于n1=(1,0,0), n2=(0,1,0): 共同邻居(除原点外)包含(1,1,0)
# 对每对正交轴方向，有1个公共邻居 → 4-环
# 正交对: (±x,±y), (±x,±z), (±y,±z) → 4×3 = 12对
# 但每对只有1个公共邻居(在正方向)
# 所以从原点看: 12个4-环... 但每个4-环被2个"起始"邻居对共享？

# 让我直接用小图计算
# 取3x3x3的SC格子
sc_sites = {}
L = 3
for x in range(-L, L+1):
    for y in range(-L, L+1):
        for z in range(-L, L+1):
            site = (x, y, z)
            nbrs = set()
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nbrs.add((x+dx, y+dy, z+dz))
            sc_sites[site] = nbrs

sc_n3_exact = count_triangles_per_site(sc_sites)
# 4-cycles: 更可靠的方法
sc_n4_count = 0
center = (0, 0, 0)
nbrs_c = sc_sites[center]
for j in nbrs_c:
    for k in nbrs_c:
        if j >= k: continue  # 避免重复
        # 找j和k的共同邻居（除center外）
        common = (sc_sites[j] & sc_sites[k]) - {center}
        sc_n4_count += len(common)
# 每个4-环从center看被数了C(4,2)=6次？不对
# 4-环: center→j→m→k→center
# 被数了 (j,k) 和 (k,j) 各一次 → 除以2
# 不对，j<k已经去重了
# 每对(j,k)的每个common neighbor m形成一个4-环
# 所以sc_n4_count/2 (因为j<k)就是4-环数...
# 不对，每个4-环中center有两个"对角"邻居j,k
# 所以sc_n4_count就是从center出发的4-环数

# 对于SC:
# 正交邻居对 (1,0,0)-(0,1,0): 共同邻居(非原点) = {(1,1,0)} → 1
# 正交邻居对 (1,0,0)-(0,-1,0): 共同邻居 = {(1,-1,0)} → 1
# ... 每对正交方向有2个4-环 (正正和正负等)
# 正交对总数 = 3个轴对 × 4个符号组合 / 2 = ...

# 让我直接用数字
n4_center = 0
nbrs_list = list(nbrs_c)
for i in range(len(nbrs_list)):
    for j in range(i+1, len(nbrs_list)):
        common = (sc_sites[nbrs_list[i]] & sc_sites[nbrs_list[j]]) - {center}
        n4_center += len(common)

classic_data[-1]['n_3'] = 0  # honeycomb
classic_data[-2]['n_4'] = 4  # square, overwritten

# SC
classic_data.append({'name': 'sc', 'z': 6, 'd': 3, 'g': 4, 'n_3': 0, 'n_4': n4_center, 'pc': 0.311600})

# BCC (z=8, g=4)
# BCC: 角原子+体心原子
# 最近邻: 每个角原子有8个体心近邻
# 体心近邻之间的距离: 对于(1/2,1/2,1/2)和(-1/2,1/2,1/2), 距离=1
# 角原子之间的距离: √3 > 1 → 不是近邻
# 所以BCC的最近邻之间: 8个体心近邻构成正方体
# 正方体有12条边 → 每个近邻有4个其他近邻作为邻居
# n_3 = 0 (正方体无三角形面)
# n_4: 正方体有6个面 → n_4 = 6
# 但需要考虑BCC中原子的完整环境
# 在BCC中，体心原子也是角原子的近邻 → 角原子也在6个正方体面中

bcc_nbrs = {}
L = 2
for x in range(-L, L+1):
    for y in range(-L, L+1):
        for z in range(-L, L+1):
            for bx in [0, 0.5]:
                for by in [0, 0.5]:
                    for bz in [0, 0.5]:
                        if bx == by == bz == 0.5 or bx == by == bz == 0:
                            site = (x+bx, y+by, z+bz)
                            nbrs = set()
                            for dx in [-0.5, 0.5]:
                                for dy in [-0.5, 0.5]:
                                    for dz in [-0.5, 0.5]:
                                        if abs(dx)+abs(dy)+abs(dz) == 1.5:
                                            nbr = (x+bx+dx, y+by+dy, z+bz+dz)
                                            if nbr in [(xx+bbx, yy+bby, zz+bbz)
                                                       for xx in range(-L, L+1)
                                                       for yy in range(-L, L+1)
                                                       for zz in range(-L, L+1)
                                                       for bbx in [0, 0.5]
                                                       for bby in [0, 0.5]
                                                       for bbz in [0, 0.5]
                                                       if (bbx == bby == bbz == 0.5 or bbx == bby == bbz == 0)]:
                                                nbrs.add(nbr)
                            bcc_nbrs[site] = nbrs

# 直接用简单方法计算BCC
# BCC近邻向量: (±1/2, ±1/2, ±1/2) → 8个
# 两个近邻之间的距离: 如(1/2,1/2,1/2)和(-1/2,1/2,1/2) → 距离=1
# 但BCC的最近邻距离 = √3/2 ≈ 0.866
# 所以近邻之间距离=1 > √3/2 → 不是近邻！
# 所以BCC中，最近邻之间没有键 → n_3=0, n_4=0?

# 等等，BCC中次近邻距离 = 1 (沿轴方向)
# 最近邻距离 = √3/2 ≈ 0.866
# 所以如果只考虑最近邻，BCC的近邻之间没有键
# 但是4-环: 原点→n1→?→n2→原点
# 需要找一个点m，同时与n1和n2是近邻
# n1=(1/2,1/2,1/2), n2=(-1/2,1/2,1/2)
# m需要同时是n1和n2的近邻(距离√3/2)
# m-n1距离 = √3/2, m-n2距离 = √3/2
# 从n1出发，近邻是n1+(±1/2,±1/2,±1/2)
# = (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
# 从n2出发，近邻是n2+(±1/2,±1/2,±1/2)
# = (-1,0,0), (-1,0,1), (-1,1,0), (-1,1,1), (0,0,0), (0,0,1), (0,1,0), (0,1,1)
# 共同近邻(不含原点): (0,0,1), (0,1,0), (0,1,1)
# 所以n1和n2有3个共同近邻 → 3个4-环

# 让我用更系统的方法
# BCC近邻向量
bcc_nn = [(sx*0.5, sy*0.5, sz*0.5) for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]]

# 计算中心原子的n_4
bcc_n4 = 0
for i in range(len(bcc_nn)):
    for j in range(i+1, len(bcc_nn)):
        n1 = bcc_nn[i]
        n2 = bcc_nn[j]
        # 找n1的近邻中也是n2的近邻的点（除原点外）
        n1_nbrs = set(tuple(n1[k]+d for k,d in enumerate(v)) for v in bcc_nn)
        n2_nbrs = set(tuple(n2[k]+d for k,d in enumerate(v)) for v in bcc_nn)
        common = n1_nbrs & n2_nbrs - {(0,0,0)}
        bcc_n4 += len(common)

classic_data.append({'name': 'bcc', 'z': 8, 'd': 3, 'g': 4, 'n_3': 0, 'n_4': bcc_n4, 'pc': 0.245800})

# FCC (z=12, g=3)
# FCC近邻向量: (±1/2, ±1/2, 0) 的排列 → 12个
fcc_nn = []
for perm in [(0,1,2), (0,2,1), (1,0,2)]:
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            v = [0, 0, 0]
            v[perm[1]] = s1 * 0.5
            v[perm[2]] = s2 * 0.5
            fcc_nn.append(tuple(v))

# 计算FCC的n_3
fcc_n3 = 0
for i in range(len(fcc_nn)):
    for j in range(i+1, len(fcc_nn)):
        for k in range(j+1, len(fcc_nn)):
            # 检查n1-n2, n2-n3, n1-n3之间是否都有键
            d12 = sum((fcc_nn[i][m]-fcc_nn[j][m])**2 for m in range(3))
            d23 = sum((fcc_nn[j][m]-fcc_nn[k][m])**2 for m in range(3))
            d13 = sum((fcc_nn[i][m]-fcc_nn[k][m])**2 for m in range(3))
            nn_dist_sq = 0.5  # (0.5)^2 + (0.5)^2 = 0.5
            if abs(d12-nn_dist_sq) < 0.01 and abs(d23-nn_dist_sq) < 0.01 and abs(d13-nn_dist_sq) < 0.01:
                fcc_n3 += 1
# fcc_n3 = 每个三角形被数了1次（从中心原子看）

# FCC的n_4
fcc_n4 = 0
for i in range(len(fcc_nn)):
    for j in range(i+1, len(fcc_nn)):
        n1 = fcc_nn[i]
        n2 = fcc_nn[j]
        # 找n1和n2的共同近邻(除原点外)
        n1_nbrs = set(tuple(n1[m]+v[m] for m in range(3)) for v in fcc_nn)
        n2_nbrs = set(tuple(n2[m]+v[m] for m in range(3)) for v in fcc_nn)
        common = (n1_nbrs & n2_nbrs) - {(0,0,0)}
        # 只计算距离=最近邻距离的共同点
        real_common = [c for c in common if abs(sum(c[m]**2 for m in range(3))-0.75)<0.1
                      or abs(sum((c[m]-n1[m])**2 for m in range(3))-0.5)<0.1]
        fcc_n4 += len([c for c in common if abs(sum((c[m]-n1[m])**2 for m in range(3))-0.5)<0.01])

classic_data.append({'name': 'fcc', 'z': 12, 'd': 3, 'g': 3, 'n_3': fcc_n3, 'n_4': fcc_n4, 'pc': 0.118800})

# Diamond (z=4, g=6)
# 金刚石中每个原子有4个近邻（四面体排列）
# 最近邻之间没有键（距离>最近邻距离）
# 最短环是6元环
# n_3 = 0, n_4 = 0
classic_data.append({'name': 'diamond', 'z': 4, 'd': 3, 'g': 6, 'n_3': 0, 'n_4': 0, 'pc': 0.431900})

# HCP (z=12, d=3, g=3)
# HCP近邻: 6个同层 + 3个上层 + 3个下层 = 12
# 同层: 六角排列 (距离a)
# 上层: 3个 (距离a)
# 下层: 3个 (距离a)
# n_3: 同层的三角形(6个, 与triangular一样) + 层间三角形
a = 1.0
c = np.sqrt(8/3) * a  # 理想HCP的c/a比

# HCP近邻向量（相对于中心原子）
hcp_nn = []
# 同层6个
for i in range(6):
    angle = i * np.pi / 3
    hcp_nn.append((a*np.cos(angle), a*np.sin(angle), 0))
# 上层3个
for i in range(3):
    angle = i * 2*np.pi/3 + np.pi/6
    hcp_nn.append((a*np.cos(angle), a*np.sin(angle), c/2))
# 下层3个
for i in range(3):
    angle = i * 2*np.pi/3 + np.pi/6
    hcp_nn.append((a*np.cos(angle), a*np.sin(angle), -c/2))

# 计算HCP的n_3
hcp_n3 = 0
nn_dist_sq = a**2
for i in range(len(hcp_nn)):
    for j in range(i+1, len(hcp_nn)):
        for k in range(j+1, len(hcp_nn)):
            d12 = sum((hcp_nn[i][m]-hcp_nn[j][m])**2 for m in range(3))
            d23 = sum((hcp_nn[j][m]-hcp_nn[k][m])**2 for m in range(3))
            d13 = sum((hcp_nn[i][m]-hcp_nn[k][m])**2 for m in range(3))
            if abs(d12-nn_dist_sq)<0.01 and abs(d23-nn_dist_sq)<0.01 and abs(d13-nn_dist_sq)<0.01:
                hcp_n3 += 1

# HCP的n_4
hcp_n4 = 0
for i in range(len(hcp_nn)):
    for j in range(i+1, len(hcp_nn)):
        n1 = hcp_nn[i]
        n2 = hcp_nn[j]
        d12 = sum((n1[m]-n2[m])**2 for m in range(3))
        if abs(d12-nn_dist_sq) > 0.01:  # 只考虑有键的近邻对
            continue
        n1_nbrs = set(tuple(n1[m]+v[m] for m in range(3)) for v in hcp_nn)
        n2_nbrs = set(tuple(n2[m]+v[m] for m in range(3)) for v in hcp_nn)
        common = n1_nbrs & n2_nbrs
        # 排除原点
        common -= {(0,0,0)}
        # 只计算确实是近邻的
        valid = [c for c in common if abs(sum(c[m]**2 for m in range(3))-nn_dist_sq)<0.1]
        hcp_n4 += len(valid)

classic_data.append({'name': 'hcp', 'z': 12, 'd': 3, 'g': 3, 'n_3': hcp_n3, 'n_4': hcp_n4, 'pc': 0.195716})

# Hypercubic lattices (d=4,...,9): z=2d, g=4, 无三角形
for d_val in range(4, 10):
    classic_data.append({
        'name': f'hypersc_{d_val}', 'z': 2*d_val, 'd': d_val, 'g': 4,
        'n_3': 0, 'n_4': 0, 'pc': 0  # pc will be filled from data
    })

# Print results
print(f"\n经典晶格结构描述符:")
print(f"{'晶格':<15} {'z':>3} {'d':>2} {'g':>2} {'n_3':>5} {'n_4':>5}")
print("-" * 40)
for cd in classic_data:
    if cd['pc'] > 0:
        print(f"{cd['name']:<15} {cd['z']:>3} {cd['d']:>2} {cd['g']:>2} {cd['n_3']:>5} {cd['n_4']:>5}")

# ============================================================
# 3. 关键对比: FCC vs HCP
# ============================================================
print(f"\n{'='*75}")
print("【关键对比: FCC vs HCP】")
print(f"{'='*75}")

fcc_entry = [c for c in classic_data if c['name'] == 'fcc'][0]
hcp_entry = [c for c in classic_data if c['name'] == 'hcp'][0]

print(f"\nFCC: z={fcc_entry['z']}, d={fcc_entry['d']}, g={fcc_entry['g']}")
print(f"  n_3 = {fcc_entry['n_3']}")
print(f"  n_4 = {fcc_entry['n_4']}")
print(f"  p_c = {fcc_entry['pc']}")

print(f"\nHCP: z={hcp_entry['z']}, d={hcp_entry['d']}, g={hcp_entry['g']}")
print(f"  n_3 = {hcp_entry['n_3']}")
print(f"  n_4 = {hcp_entry['n_4']}")
print(f"  p_c = {hcp_entry['pc']}")

print(f"\nn_3 差异: FCC={fcc_entry['n_3']}, HCP={hcp_entry['n_3']}")
print(f"n_4 差异: FCC={fcc_entry['n_4']}, HCP={hcp_entry['n_4']}")
print(f"p_c 差异: {(hcp_entry['pc']-fcc_entry['pc'])/fcc_entry['pc']*100:.1f}%")

# ============================================================
# 4. Malarz 2D 晶格的 n_3 计算
# ============================================================
print(f"\n{'='*75}")
print("【Malarz 2D 晶格的 n_3 计算】")
print(f"{'='*75}")

# Malarz晶格的zone定义
zone_offsets = {
    1: [(1,0),(-1,0),(0,1),(0,-1)],
    2: [(1,1),(1,-1),(-1,1),(-1,-1)],
    3: [(2,0),(-2,0),(0,2),(0,-2)],
    4: [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)],
    5: [(2,2),(2,-2),(-2,2),(-2,-2)],
    6: [(3,0),(-3,0),(0,3),(0,-3)],
}

def parse_malarz_name(name):
    """解析 sq-X,Y,Z 格式的晶格名"""
    zones_str = name.replace('sq-', '')
    zones = [int(z) for z in zones_str.split(',')]
    return zones

def build_malarz_lattice(zones, L=10):
    """构建Malarz晶格的邻接关系"""
    # 生成邻居向量
    nn_vecs = set()
    for z in zones:
        for offset in zone_offsets[z]:
            nn_vecs.add(offset)
    nn_vecs = list(nn_vecs)

    # 构建小规模格子用于计算n_3
    # 使用L×L的格子
    sites = {}
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            site = (x, y)
            nbrs = set()
            for dx, dy in nn_vecs:
                nbrs.add((x+dx, y+dy))
            sites[site] = nbrs

    return sites, nn_vecs

def compute_malarz_n3(zones):
    """计算Malarz晶格的n_3"""
    # 需要足够大的格子以避免边界效应
    L = 5
    sites, nn_vecs = build_malarz_lattice(zones, L)

    # 只计算中心区域的原子（避免边界效应）
    center_sites = [(x,y) for x in range(-L+3, L-2) for y in range(-L+3, L-2)]

    n3_total = 0
    for site in center_sites:
        nbrs = sites[site]
        for j, k in combinations(nbrs, 2):
            # 检查j-k之间是否有键
            diff = (k[0]-j[0], k[1]-j[1])
            if diff in nn_vecs:
                n3_total += 1

    n3_per_site = n3_total / len(center_sites)
    return n3_per_site

# 计算所有Malarz晶格的n_3
malarz_raw = [
    ('sq-1,2,3,4,5,6', 28, 0.14326), ('sq-2,3,4,5,6', 24, 0.14575),
    ('sq-1,3,4,5,6', 24, 0.14801), ('sq-1,2,4,5,6', 24, 0.15223),
    ('sq-1,2,3,5,6', 20, 0.16661), ('sq-1,2,3,4,6', 24, 0.16134),
    ('sq-3,4,5,6', 20, 0.15221), ('sq-2,4,5,6', 20, 0.15844),
    ('sq-2,3,5,6', 16, 0.17601), ('sq-2,3,4,6', 20, 0.16529),
    ('sq-1,4,5,6', 20, 0.15815), ('sq-1,3,5,6', 16, 0.18007),
    ('sq-1,3,4,6', 20, 0.16675), ('sq-1,2,5,6', 16, 0.18216),
    ('sq-1,2,4,6', 20, 0.17409), ('sq-1,2,3,6', 16, 0.20134),
    ('sq-4,5,6', 16, 0.16819), ('sq-3,5,6', 12, 0.19867),
    ('sq-3,4,6', 16, 0.17288), ('sq-2,5,6', 12, 0.20036),
    ('sq-2,4,6', 16, 0.18454), ('sq-2,3,6', 12, 0.21503),
    ('sq-1,5,6', 12, 0.19936), ('sq-1,4,6', 16, 0.18143),
    ('sq-1,3,6', 12, 0.22577), ('sq-1,2,6', 12, 0.23076),
    ('sq-5,6', 8, 0.24422), ('sq-4,6', 12, 0.19799),
    ('sq-3,6', 8, 0.25673), ('sq-2,6', 8, 0.26600),
    ('sq-1,6', 8, 0.27309),
]

# 确定每个Malarz晶格的girth
def compute_girth(nn_vecs):
    """计算BFS girth"""
    from collections import deque
    # BFS from origin
    visited = {tuple([0,0]): 0}
    queue = deque([(tuple([0,0]), 0)])
    parent = {tuple([0,0]): None}

    while queue:
        node, dist = queue.popleft()
        for dx, dy in nn_vecs:
            nbr = (node[0]+dx, node[1]+dy)
            if nbr in visited:
                if parent.get(node) != nbr and parent.get(nbr) != node:
                    # Found a cycle
                    cycle_len = dist + visited[nbr] + 1
                    return cycle_len  # This is approximate
            else:
                visited[nbr] = dist + 1
                parent[nbr] = node
                queue.append((nbr, dist+1))
    return None

# 实际计算更简单的girth
def compute_girth_exact(nn_vecs):
    """精确计算girth"""
    # BFS from origin, look for shortest cycle
    max_depth = 20
    # Adjacency as set of offset vectors
    offset_set = set(nn_vecs)

    # Check all cycles of length 3, 4, 5, ...
    # Length 3: a+b+c=0 where a,b,c ∈ nn_vecs
    for a in nn_vecs:
        for b in nn_vecs:
            if a == b: continue
            c = (-a[0]-b[0], -a[1]-b[1])
            if c in offset_set and c != a and c != b:
                return 3

    # Length 4: a+b+c+d=0
    for a in nn_vecs:
        for b in nn_vecs:
            for c in nn_vecs:
                d = (-a[0]-b[0]-c[0], -a[1]-b[1]-c[1])
                if d in offset_set:
                    # Check it's a proper cycle (no sub-cycles)
                    if a != -b and b != -c and c != -d:
                        if (a[0]+b[0], a[1]+b[1]) != (0,0) and \
                           (b[0]+c[0], b[1]+c[1]) != (0,0):
                            return 4
    return 6  # default

print(f"\n计算Malarz晶格的 n_3 和 girth...")
print(f"{'晶格':<20} {'z':>3} {'g':>2} {'n_3':>6} {'p_c':>8}")
print("-" * 50)

malarz_data = []
for name, z_val, pc_val in malarz_raw:
    zones = parse_malarz_name(name)
    nn_vecs = []
    for zone in zones:
        nn_vecs.extend(zone_offsets[zone])
    nn_vecs_set = list(set(nn_vecs))

    g = compute_girth_exact(nn_vecs_set)
    n3 = compute_malarz_n3(zones)

    malarz_data.append({
        'name': name, 'z': z_val, 'd': 2, 'g': g,
        'n_3': n3, 'n_4': 0, 'pc': pc_val
    })

    print(f"{name:<20} {z_val:>3} {g:>2} {n3:>6.1f} {pc_val:>8.5f}")

# ============================================================
# 5. 合并所有数据并保存
# ============================================================
print(f"\n{'='*75}")
print("【合并数据】")
print(f"{'='*75}")

# 保存
all_struct = classic_data + malarz_data
df_struct = pd.DataFrame(all_struct)
df_struct = df_struct[df_struct['pc'] > 0]  # 去掉没有pc的行
df_struct.to_csv(f'{OUT}/lattice_structure_descriptors.csv', index=False)

print(f"\n保存: lattice_structure_descriptors.csv")
print(f"总晶格数: {len(df_struct)}")

# ============================================================
# 6. 分析 n_3 与 p_c 残差的关系
# ============================================================
print(f"\n{'='*75}")
print("【n_3 与 p_c 残差的关系】")
print(f"{'='*75}")

z = df_struct['z'].values.astype(float)
d = df_struct['d'].values.astype(float)
g = df_struct['g'].values.astype(float)
n3 = df_struct['n_3'].values.astype(float)
pc = df_struct['pc'].values

# 当前公式的残差
a_opt = 1.695
pred = 1/(z-1) * (1 + a_opt * np.power(z, 0.6) / (np.power(d, 1.5) * np.power(g, 0.5)))
residuals = pc - pred
rel_errors = residuals / pc * 100

# n_3 与残差的相关性
from scipy.stats import pearsonr
n3_per_z = n3 / z  # 归一化
r_n3, p_n3 = pearsonr(n3_per_z, rel_errors)
r_n3_abs, p_n3_abs = pearsonr(n3, np.abs(rel_errors))

print(f"\n当前公式残差与 n_3 的关系:")
print(f"  corr(n_3/z, residual%) = {r_n3:.4f} (p={p_n3:.2e})")
print(f"  corr(n_3, |residual|%) = {r_n3_abs:.4f} (p={p_n3_abs:.2e})")

# 引入 n_3 修正
print(f"\n尝试引入 n_3 修正:")

# 方案1: p_c = 1/(z-1) · (1 + a·z^α/(d^β·g^γ) - b·n_3/z)
from scipy.optimize import minimize

def model_n3_v1(params, z, d, g, n3):
    a, alpha, beta, gamma, b = params
    return 1/(z-1) * (1 + a * np.power(z, alpha) / (np.power(d, beta) * np.power(g, gamma)) - b * n3/z)

def obj_n3_v1(params):
    pred = model_n3_v1(params, z, d, g, n3)
    return np.sum((pc - pred)**2)

best_ss = np.inf
best_p = None
for _ in range(50):
    init = [2.0, 0.5, 1.5, 0.5, 0.01]
    init_r = [x * np.random.uniform(0.5, 2.0) for x in init]
    res = minimize(obj_n3_v1, init_r, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_ss:
        best_ss = res.fun
        best_p = res.x

if best_p is not None:
    pred_v1 = model_n3_v1(best_p, z, d, g, n3)
    R2_v1 = 1 - np.sum((pc-pred_v1)**2)/np.sum((pc-np.mean(pc))**2)
    MAPE_v1 = np.mean(np.abs((pc-pred_v1)/pc))*100
    print(f"\n  方案1: Bethe·(1+a·z^α/(d^β·g^γ) - b·n_3/z)")
    print(f"    a={best_p[0]:.3f}, α={best_p[1]:.3f}, β={best_p[2]:.3f}, γ={best_p[3]:.3f}, b={best_p[4]:.4f}")
    print(f"    R²={R2_v1:.4f}, MAPE={MAPE_v1:.2f}%")

# 方案2: p_c = 1/(z-1) · (1 + a·z^α/(d^β·g^γ) · (1 - b·n_3/z^δ))
def model_n3_v2(params, z, d, g, n3):
    a, alpha, beta, gamma, b, delta = params
    correction = 1 + a * np.power(z, alpha) / (np.power(d, beta) * np.power(g, gamma))
    reduction = 1 - b * n3 / np.power(z, delta)
    return 1/(z-1) * correction * reduction

def obj_n3_v2(params):
    pred = model_n3_v2(params, z, d, g, n3)
    return np.sum((pc - pred)**2)

best_ss2 = np.inf
best_p2 = None
for _ in range(50):
    init2 = [2.0, 0.5, 1.5, 0.5, 0.1, 0.5]
    init_r2 = [x * np.random.uniform(0.5, 2.0) for x in init2]
    res2 = minimize(obj_n3_v2, init_r2, method='Nelder-Mead',
                   options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res2.fun < best_ss2:
        best_ss2 = res2.fun
        best_p2 = res2.x

if best_p2 is not None:
    pred_v2 = model_n3_v2(best_p2, z, d, g, n3)
    R2_v2 = 1 - np.sum((pc-pred_v2)**2)/np.sum((pc-np.mean(pc))**2)
    MAPE_v2 = np.mean(np.abs((pc-pred_v2)/pc))*100
    print(f"\n  方案2: Bethe·(1+a·z^α/(d^β·g^γ))·(1 - b·n_3/z^δ)")
    print(f"    a={best_p2[0]:.3f}, α={best_p2[1]:.3f}, β={best_p2[2]:.3f}, γ={best_p2[3]:.3f}, b={best_p2[4]:.4f}, δ={best_p2[5]:.3f}")
    print(f"    R²={R2_v2:.4f}, MAPE={MAPE_v2:.2f}%")

# 方案3: n_3 直接作为独立修正项
def model_n3_v3(params, z, d, g, n3):
    a, b, c, w = params
    base = a * np.power(z, b) / (np.power(d, c))
    return 1/(z-1) * (1 + base / np.power(g, 0.5)) * (1 - w * n3 / (z * np.sqrt(g)))

def obj_n3_v3(params):
    pred = model_n3_v3(params, z, d, g, n3)
    return np.sum((pc - pred)**2)

best_ss3 = np.inf
best_p3 = None
for _ in range(50):
    init3 = [2.0, 0.5, 1.5, 0.01]
    init_r3 = [x * np.random.uniform(0.5, 2.0) for x in init3]
    res3 = minimize(obj_n3_v3, init_r3, method='Nelder-Mead',
                   options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res3.fun < best_ss3:
        best_ss3 = res3.fun
        best_p3 = res3.x

if best_p3 is not None:
    pred_v3 = model_n3_v3(best_p3, z, d, g, n3)
    R2_v3 = 1 - np.sum((pc-pred_v3)**2)/np.sum((pc-np.mean(pc))**2)
    MAPE_v3 = np.mean(np.abs((pc-pred_v3)/pc))*100
    print(f"\n  方案3: 1/(z-1)·(1+a·z^b/(d^c·√g))·(1-w·n_3/(z·√g))")
    print(f"    a={best_p3[0]:.3f}, b={best_p3[1]:.3f}, c={best_p3[2]:.3f}, w={best_p3[3]:.4f}")
    print(f"    R²={R2_v3:.4f}, MAPE={MAPE_v3:.2f}%")

# 对比原始公式
pred_orig = 1/(z-1) * (1 + 1.695 * np.power(z, 0.6) / (np.power(d, 1.5) * np.power(g, 0.5)))
R2_orig = 1 - np.sum((pc-pred_orig)**2)/np.sum((pc-np.mean(pc))**2)
MAPE_orig = np.mean(np.abs((pc-pred_orig)/pc))*100

print(f"\n对比:")
print(f"  原始 (无n_3): R²={R2_orig:.4f}, MAPE={MAPE_orig:.2f}%")

# FCC 和 HCP 的特殊分析
print(f"\n{'='*75}")
print("【FCC 和 HCP 的 n_3 分析】")
print(f"{'='*75}")

fcc_mask = df_struct['name'].values == 'fcc'
hcp_mask = df_struct['name'].values == 'hcp'

if np.any(fcc_mask):
    print(f"\n  FCC: n_3={n3[fcc_mask][0]}, z={z[fcc_mask][0]}")
    print(f"    n_3/z = {n3[fcc_mask][0]/z[fcc_mask][0]:.2f}")
    print(f"    p_c = {pc[fcc_mask][0]:.4f}")
    print(f"    公式预测 = {pred_orig[fcc_mask][0]:.4f}")
    print(f"    误差 = {(pc[fcc_mask][0]-pred_orig[fcc_mask][0])/pc[fcc_mask][0]*100:.1f}%")
    if best_p is not None:
        print(f"    v1预测 = {pred_v1[fcc_mask][0]:.4f}, 误差 = {(pc[fcc_mask][0]-pred_v1[fcc_mask][0])/pc[fcc_mask][0]*100:.1f}%")
    if best_p2 is not None:
        print(f"    v2预测 = {pred_v2[fcc_mask][0]:.4f}, 误差 = {(pc[fcc_mask][0]-pred_v2[fcc_mask][0])/pc[fcc_mask][0]*100:.1f}%")

if np.any(hcp_mask):
    print(f"\n  HCP: n_3={n3[hcp_mask][0]}, z={z[hcp_mask][0]}")
    print(f"    n_3/z = {n3[hcp_mask][0]/z[hcp_mask][0]:.2f}")
    print(f"    p_c = {pc[hcp_mask][0]:.4f}")
    print(f"    公式预测 = {pred_orig[hcp_mask][0]:.4f}")
    print(f"    误差 = {(pc[hcp_mask][0]-pred_orig[hcp_mask][0])/pc[hcp_mask][0]*100:.1f}%")

print(f"\n{'='*75}")
print("【计算完成】")
print(f"{'='*75}")
