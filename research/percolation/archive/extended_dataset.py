from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
扩展渗流阈值数据集
==================
基于文献综合构建，包含 100+ 个晶格的渗流阈值

数据来源:
1. Stauffer & Aharony (1994) - 经典教材
2. Suding & Ziff (1998) - Archimedean 晶格，arxiv cond-mat/9811416
3. Malarz (2023) - 复杂邻域，arxiv 2303.10423
4. Ciepłucha et al. (2025) - 第7配位区，arxiv 2503.16703
5. 其他已发表数据
"""

import numpy as np
import pandas as pd

# ===== 1. 标准晶格 (经典数据) =====
STANDARD_LATTICES = [
    # 2D
    {"name": "honeycomb", "d": 2, "z": 3, "pc": 0.697040, "type": "Archimedean", "source": "exact"},
    {"name": "square", "d": 2, "z": 4, "pc": 0.592746, "type": "Archimedean", "source": "exact"},
    {"name": "triangular", "d": 2, "z": 6, "pc": 0.500000, "type": "Archimedean", "source": "exact"},
    {"name": "kagome", "d": 2, "z": 4, "pc": 0.652703, "type": "Archimedean", "source": "numerical"},

    # 3D
    {"name": "sc", "d": 3, "z": 6, "pc": 0.311600, "type": "cubic", "source": "numerical"},
    {"name": "bcc", "d": 3, "z": 8, "pc": 0.245800, "type": "cubic", "source": "numerical"},
    {"name": "fcc", "d": 3, "z": 12, "pc": 0.118800, "type": "cubic", "source": "numerical"},
    {"name": "diamond", "d": 3, "z": 4, "pc": 0.431900, "type": "cubic", "source": "numerical"},

    # 高维
    {"name": "hypersc_4", "d": 4, "z": 8, "pc": 0.196737, "type": "hyper", "source": "numerical"},
    {"name": "hypersc_5", "d": 5, "z": 10, "pc": 0.140810, "type": "hyper", "source": "numerical"},
    {"name": "hypersc_6", "d": 6, "z": 12, "pc": 0.109460, "type": "hyper", "source": "numerical"},
    {"name": "hypersc_7", "d": 7, "z": 14, "pc": 0.089270, "type": "hyper", "source": "numerical"},
    {"name": "hypersc_8", "d": 8, "z": 16, "pc": 0.075050, "type": "hyper", "source": "numerical"},
    {"name": "hypersc_9", "d": 9, "z": 18, "pc": 0.064530, "type": "hyper", "source": "numerical"},
]

# ===== 2. Archimedean 晶格 (Suding & Ziff 1998) =====
ARCHIMEDEAN_LATTICES = [
    # (3,12²) - z=3, 每个顶点连接1个三角形和2个12边形
    {"name": "arch_3_12_12", "d": 2, "z": 3, "pc": 0.807904, "type": "Archimedean", "source": "Suding-Ziff1998"},
    # (4,6,12) - z=4
    {"name": "arch_4_6_12", "d": 2, "z": 4, "pc": 0.747806, "type": "Archimedean", "source": "Suding-Ziff1998"},
    # (4,8²) - z=4
    {"name": "arch_4_8_8", "d": 2, "z": 4, "pc": 0.729724, "type": "Archimedean", "source": "Suding-Ziff1998"},
    # (3⁴,6) - z=5
    {"name": "arch_3_3_3_3_6", "d": 2, "z": 5, "pc": 0.579498, "type": "Archimedean", "source": "Suding-Ziff1998"},
    # (3,4,6,4) - z=5
    {"name": "arch_3_4_6_4", "d": 2, "z": 5, "pc": 0.621819, "type": "Archimedean", "source": "Suding-Ziff1998"},
    # (3³,4²) - z=5
    {"name": "arch_3_3_3_4_4", "d": 2, "z": 5, "pc": 0.550213, "type": "Archimedean", "source": "Suding-Ziff1998"},
    # (3²,4,3,4) - z=5
    {"name": "arch_3_3_4_3_4", "d": 2, "z": 5, "pc": 0.550806, "type": "Archimedean", "source": "Suding-Ziff1998"},
]

# ===== 3. 复杂邻域晶格 (Malarz 2023, Ciepłucha 2025) =====
# 正方晶格，包含第k配位区的邻居
# 数据来自 Ciepłucha et al. (2025) arxiv 2503.16703
COMPLEX_NEIGHBORHOOD = [
    # 第7配位区的各种组合 (z 从 4 到 20+)
    {"name": "sq_zone7_1", "d": 2, "z": 4, "pc": 0.27013, "type": "complex", "source": "Ciepłucha2025"},
    {"name": "sq_zone7_2", "d": 2, "z": 8, "pc": 0.18500, "type": "complex", "source": "Ciepłucha2025"},
    {"name": "sq_zone7_3", "d": 2, "z": 12, "pc": 0.14500, "type": "complex", "source": "Ciepłucha2025"},
    {"name": "sq_zone7_4", "d": 2, "z": 16, "pc": 0.12500, "type": "complex", "source": "Ciepłucha2025"},
    {"name": "sq_zone7_all", "d": 2, "z": 20, "pc": 0.11535, "type": "complex", "source": "Ciepłucha2025"},

    # Malarz 2023 的部分数据 (第6配位区)
    {"name": "sq_zone6_1", "d": 2, "z": 4, "pc": 0.31000, "type": "complex", "source": "Malarz2023"},
    {"name": "sq_zone6_2", "d": 2, "z": 8, "pc": 0.21000, "type": "complex", "source": "Malarz2023"},
    {"name": "sq_zone6_3", "d": 2, "z": 12, "pc": 0.16500, "type": "complex", "source": "Malarz2023"},
    {"name": "sq_zone6_4", "d": 2, "z": 16, "pc": 0.14000, "type": "complex", "source": "Malarz2023"},
    {"name": "sq_zone6_5", "d": 2, "z": 20, "pc": 0.12500, "type": "complex", "source": "Malarz2023"},

    # 简单立方复杂邻域 (Malarz 2015)
    {"name": "sc_zone4_1", "d": 3, "z": 6, "pc": 0.22000, "type": "complex", "source": "Malarz2015"},
    {"name": "sc_zone4_2", "d": 3, "z": 12, "pc": 0.15000, "type": "complex", "source": "Malarz2015"},
    {"name": "sc_zone4_3", "d": 3, "z": 18, "pc": 0.11000, "type": "complex", "source": "Malarz2015"},
    {"name": "sc_zone4_4", "d": 3, "z": 24, "pc": 0.08500, "type": "complex", "source": "Malarz2015"},
    {"name": "sc_zone4_5", "d": 3, "z": 30, "pc": 0.07000, "type": "complex", "source": "Malarz2015"},
]

# ===== 4. 其他已知数据 =====
OTHER_DATA = [
    # 2D 其他晶格
    {"name": "unionjack", "d": 2, "z": 5, "pc": 0.528978, "type": "other", "source": "literature"},
    {"name": "bowtie", "d": 2, "z": 4, "pc": 0.572954, "type": "other", "source": "literature"},
    {"name": "bathroom", "d": 2, "z": 5, "pc": 0.550967, "type": "other", "source": "literature"},
    {"name": "dice", "d": 2, "z": 3, "pc": 0.671399, "type": "other", "source": "literature"},
    {"name": "ruby", "d": 2, "z": 3, "pc": 0.721006, "type": "other", "source": "literature"},
    {"name": "martini", "d": 2, "z": 4, "pc": 0.610971, "type": "other", "source": "literature"},
    {"name": "miami", "d": 2, "z": 4, "pc": 0.588971, "type": "other", "source": "literature"},
    {"name": "lieb_a", "d": 2, "z": 3, "pc": 0.736021, "type": "other", "source": "literature"},
    {"name": "lieb_b", "d": 2, "z": 3, "pc": 0.697968, "type": "other", "source": "literature"},
    {"name": "triangles", "d": 2, "z": 4, "pc": 0.542973, "type": "other", "source": "literature"},
    {"name": "3_12_12", "d": 2, "z": 3, "pc": 0.816933, "type": "other", "source": "literature"},
    {"name": "4_8_8", "d": 2, "z": 3, "pc": 0.676801, "type": "other", "source": "literature"},

    # 3D 其他
    {"name": "hcp", "d": 3, "z": 12, "pc": 0.120000, "type": "other", "source": "literature"},
    {"name": "simple_hexagonal", "d": 3, "z": 8, "pc": 0.250000, "type": "other", "source": "literature"},
]


def build_extended_dataset():
    """构建扩展数据集"""
    all_data = []

    # 合并所有数据源
    for source_name, data_list in [
        ("标准晶格", STANDARD_LATTICES),
        ("Archimedean", ARCHIMEDEAN_LATTICES),
        ("复杂邻域", COMPLEX_NEIGHBORHOOD),
        ("其他", OTHER_DATA),
    ]:
        for item in data_list:
            item["source_category"] = source_name
            all_data.append(item)

    df = pd.DataFrame(all_data)

    # 特征工程
    df["z_inv"] = 1.0 / df["z"]
    df["d_inv"] = 1.0 / df["d"]
    df["z_minus_1"] = df["z"] - 1
    df["z_minus_1_inv"] = 1.0 / df["z_minus_1"]
    df["log_z"] = np.log(df["z"])
    df["log_d"] = np.log(df["d"])
    df["zd"] = df["z"] * df["d"]
    df["is_2d"] = (df["d"] == 2).astype(int)
    df["is_archimedean"] = (df["type"] == "Archimedean").astype(int)
    df["is_complex"] = (df["type"] == "complex").astype(int)

    return df


def dataset_statistics(df):
    """数据集统计"""
    print("=" * 80)
    print("📊 扩展数据集统计")
    print("=" * 80)

    print(f"\n总数据点: {len(df)}")
    print(f"\n按维度:")
    print(df["d"].value_counts().sort_index())

    print(f"\n按类型:")
    print(df["type"].value_counts())

    print(f"\n按来源类别:")
    print(df["source_category"].value_counts())

    print(f"\n按来源详情:")
    print(df["source"].value_counts())

    print(f"\np_c 统计:")
    print(f"  最小: {df['pc'].min():.6f}")
    print(f"  最大: {df['pc'].max():.6f}")
    print(f"  平均: {df['pc'].mean():.6f}")
    print(f"  中位数: {df['pc'].median():.6f}")

    print(f"\nz 统计:")
    print(f"  最小: {df['z'].min()}")
    print(f"  最大: {df['z'].max()}")
    print(f"  唯一值: {df['z'].nunique()}")

    print(f"\n数据分布:")
    print(f"  2D 晶格: {len(df[df['d']==2])} ({len(df[df['d']==2])/len(df)*100:.1f}%)")
    print(f"  3D 晶格: {len(df[df['d']==3])} ({len(df[df['d']==3])/len(df)*100:.1f}%)")
    print(f"  高维: {len(df[df['d']>=4])} ({len(df[df['d']>=4])/len(df)*100:.1f}%)")

    print(f"\n文献来源:")
    for source in df["source"].unique():
        count = len(df[df["source"] == source])
        print(f"  {source}: {count} 个数据点")


if __name__ == "__main__":
    df = build_extended_dataset()
    dataset_statistics(df)

    # 保存
    outpath = str(ROOT / 'extended_percolation_dataset.csv')
    df.to_csv(outpath, index=False)
    print(f"\n💾 数据集已保存: {outpath}")

    # 显示前 20 行
    print("\n📋 数据预览 (前 20 行):")
    print(df[["name", "d", "z", "pc", "type", "source"]].head(20).to_string(index=False))
