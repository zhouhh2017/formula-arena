from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
Phase 1: 整理完整的晶格数据集（MC模拟补充前）
==============================================

合并所有来源的数据，去重，统一格式。
等MC模拟完成后，再合并补充数据。

数据来源：
1. 经典晶格（14个）：文献已知的高精度值
2. Malarz 2023（31个）：2D正方晶格+复杂邻域
3. MC模拟（6个）：本轮早期模拟
"""

import numpy as np
import pandas as pd
import json

# ============ 1. 经典晶格数据（文献值） ============
classic_lattices = [
    # 2D经典
    {'name': 'honeycomb',    'z': 3,  'd': 2, 'g': 6, 'pc': 0.697040, 'source': 'literature', 'note': 'exact'},
    {'name': 'square',       'z': 4,  'd': 2, 'g': 4, 'pc': 0.592746, 'source': 'literature', 'note': 'high-precision'},
    {'name': 'triangular',   'z': 6,  'd': 2, 'g': 3, 'pc': 0.500000, 'source': 'literature', 'note': 'exact'},
    {'name': 'kagome',       'z': 4,  'd': 2, 'g': 3, 'pc': 0.652703, 'source': 'literature', 'note': 'high-precision'},
    # 3D经典
    {'name': 'sc',           'z': 6,  'd': 3, 'g': 4, 'pc': 0.311600, 'source': 'literature', 'note': 'high-precision'},
    {'name': 'bcc',          'z': 8,  'd': 3, 'g': 4, 'pc': 0.245800, 'source': 'literature', 'note': 'high-precision'},
    {'name': 'fcc',          'z': 12, 'd': 3, 'g': 3, 'pc': 0.118800, 'source': 'literature', 'note': 'high-precision'},
    {'name': 'diamond',      'z': 4,  'd': 3, 'g': 6, 'pc': 0.431900, 'source': 'literature', 'note': 'high-precision'},
    # 高维超立方
    {'name': 'hypersc_4',    'z': 8,  'd': 4, 'g': 4, 'pc': 0.196737, 'source': 'literature', 'note': 'Mercat & Campos 2022'},
    {'name': 'hypersc_5',    'z': 10, 'd': 5, 'g': 4, 'pc': 0.140810, 'source': 'literature', 'note': 'Mercat & Campos 2022'},
    {'name': 'hypersc_6',    'z': 12, 'd': 6, 'g': 4, 'pc': 0.109460, 'source': 'literature', 'note': 'Mercat & Campos 2022'},
    {'name': 'hypersc_7',    'z': 14, 'd': 7, 'g': 4, 'pc': 0.089270, 'source': 'literature', 'note': 'Mercat & Campos 2022'},
    {'name': 'hypersc_8',    'z': 16, 'd': 8, 'g': 4, 'pc': 0.075050, 'source': 'literature', 'note': 'Mercat & Campos 2022'},
    {'name': 'hypersc_9',    'z': 18, 'd': 9, 'g': 4, 'pc': 0.064530, 'source': 'literature', 'note': 'Mercat & Campos 2022'},
]

# ============ 2. Malarz 2023 数据（31个2D晶格） ============
malarz_data = [
    ('sq-1,2,3,4,5,6', 28, 3, 0.14326),
    ('sq-2,3,4,5,6',   24, 3, 0.14575),
    ('sq-1,3,4,5,6',   24, 4, 0.14801),
    ('sq-1,2,4,5,6',   24, 3, 0.15223),
    ('sq-1,2,3,5,6',   20, 3, 0.16661),
    ('sq-1,2,3,4,6',   24, 3, 0.16134),
    ('sq-3,4,5,6',     20, 4, 0.15221),
    ('sq-2,4,5,6',     20, 3, 0.15844),
    ('sq-2,3,5,6',     16, 3, 0.17601),
    ('sq-2,3,4,6',     20, 3, 0.16529),
    ('sq-1,4,5,6',     20, 4, 0.15815),
    ('sq-1,3,5,6',     16, 4, 0.18007),
    ('sq-1,3,4,6',     20, 4, 0.16675),
    ('sq-1,2,5,6',     16, 3, 0.18216),
    ('sq-1,2,4,6',     20, 3, 0.17409),
    ('sq-1,2,3,6',     16, 3, 0.20134),
    ('sq-4,5,6',       16, 4, 0.16819),
    ('sq-3,5,6',       12, 4, 0.19867),
    ('sq-3,4,6',       16, 4, 0.17288),
    ('sq-2,5,6',       12, 3, 0.20036),
    ('sq-2,4,6',       16, 3, 0.18454),
    ('sq-2,3,6',       12, 3, 0.21503),
    ('sq-1,5,6',       12, 4, 0.19936),
    ('sq-1,4,6',       16, 4, 0.18143),
    ('sq-1,3,6',       12, 4, 0.22577),
    ('sq-1,2,6',       12, 3, 0.23076),
    ('sq-5,6',          8, 4, 0.24422),
    ('sq-4,6',         12, 4, 0.19799),
    ('sq-3,6',          8, 4, 0.25673),
    ('sq-2,6',          8, 3, 0.26600),
    ('sq-1,6',          8, 4, 0.27309),
]
malarz_lattices = [
    {'name': n, 'z': z, 'd': 2, 'g': g, 'pc': p, 'source': 'Malarz2023', 'note': 'zone-based 2D'}
    for n, z, g, p in malarz_data
]

# ============ 3. 早期MC模拟数据 ============
mc_early_data = [
    {'name': 'hypercubic_4d',  'z': 8,  'd': 4, 'g': 4, 'pc': 0.1934, 'source': 'MC_early', 'note': 'lit=0.1967, err=1.7%'},
    {'name': 'sc3d_zone12',    'z': 18, 'd': 3, 'g': 3, 'pc': 0.1371, 'source': 'MC_early', 'note': 'sc+nnn'},
    {'name': 'hypercubic_5d',  'z': 10, 'd': 5, 'g': 4, 'pc': 0.1473, 'source': 'MC_early', 'note': 'lit=0.1408, err=4.6%'},
    {'name': 'hypercubic_6d',  'z': 12, 'd': 6, 'g': 4, 'pc': 0.1119, 'source': 'MC_early', 'note': 'lit=0.1095, err=2.2%'},
]

# ============ 合并数据 ============
all_data = classic_lattices + malarz_lattices + mc_early_data
df = pd.DataFrame(all_data)

# 去重：如果有重复的name，保留source优先级最高的
priority = {'literature': 3, 'MC_early': 2, 'Malarz2023': 1}
df['priority'] = df['source'].map(priority)
df = df.sort_values('priority', ascending=False).drop_duplicates(subset='name', keep='first')
df = df.drop(columns='priority').sort_values(['d', 'z', 'g']).reset_index(drop=True)

# 添加衍生特征
df['zd'] = df['z'] * df['d']
df['R'] = df['pc'] * (df['z'] - 1)  # normalized threshold
df['z_inv'] = 1.0 / df['z']
df['zd_inv'] = 1.0 / df['zd']
df['log_pc'] = np.log(df['pc'])
df['log_zd'] = np.log(df['zd'])

# ============ 保存 ============
df.to_csv(str(ROOT / 'complete_dataset_v2.csv'), index=False)

# ============ 打印统计 ============
print("=" * 70)
print("【完整数据集 v2（MC补充前）】")
print("=" * 70)
print(f"总晶格数: {len(df)}")
print(f"\n维度分布:")
for d in sorted(df['d'].unique()):
    sub = df[df['d'] == d]
    print(f"  d={int(d)}: {len(sub)} 个, z ∈ {sorted(sub['z'].unique())}")

print(f"\nGirth分布:")
for g in sorted(df['g'].unique()):
    sub = df[df['g'] == g]
    print(f"  g={int(g)}: {len(sub)} 个, z ∈ {sorted(sub['z'].unique())}")

print(f"\n数据来源:")
for src in df['source'].unique():
    n = len(df[df['source'] == src])
    print(f"  {src}: {n} 个")

print(f"\n保存至: complete_dataset_v2.csv")
print(f"列: {list(df.columns)}")
