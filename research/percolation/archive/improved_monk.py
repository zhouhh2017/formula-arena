from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
改进版引擎 - 迭代到 R²=0.99
==============================
改进策略:
1. 增加更多 2D 晶格数据
2. 引入晶格拓扑描述符
3. 改进模板匹配 (分段函数、修正项)
4. 改进参数优化
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monk_engine.math_engine import (
    parse_expression, evaluate_formula, evaluate_formula_batch,
    compute_rmse, compute_r2, formula_complexity
)
from monk_engine.sr_methods import optimize_formula_params

# ===== 扩展数据集: 更多 2D 晶格 =====
EXTENDED_PERCOLATION_DATA = [
    # 2D 晶格 (扩展)
    {"lattice": "honeycomb", "d": 2, "z": 3, "pc": 0.6970, "type": "bipartite", "dual": "triangular"},
    {"lattice": "square", "d": 2, "z": 4, "pc": 0.592746, "type": "bipartite", "dual": "square"},
    {"lattice": "triangular", "d": 2, "z": 6, "pc": 0.5000, "type": "non_bipartite", "dual": "honeycomb"},
    {"lattice": "kagome", "d": 2, "z": 4, "pc": 0.6527, "type": "non_bipartite", "dual": "triangular"},
    {"lattice": "bowtie", "d": 2, "z": 4, "pc": 0.5730, "type": "non_bipartite", "dual": "unknown"},
    {"lattice": "unionjack", "d": 2, "z": 5, "pc": 0.5290, "type": "bipartite", "dual": "unknown"},
    {"lattice": "3-12", "d": 2, "z": 3, "pc": 0.8170, "type": "non_bipartite", "dual": "unknown"},
    {"lattice": "4-8", "d": 2, "z": 3, "pc": 0.6768, "type": "bipartite", "dual": "4-8"},
    {"lattice": "bathroom", "d": 2, "z": 5, "pc": 0.5510, "type": "non_bipartite", "dual": "unknown"},
    {"lattice": "dice", "d": 2, "z": 3, "pc": 0.6714, "type": "non_bipartite", "dual": "triangular"},
    {"lattice": "ruby", "d": 2, "z": 3, "pc": 0.7210, "type": "non_bipartite", "dual": "unknown"},
    {"lattice": "martini", "d": 2, "z": 4, "pc": 0.6110, "type": "non_bipartite", "dual": "unknown"},
    {"lattice": "miami", "d": 2, "z": 4, "pc": 0.5890, "type": "bipartite", "dual": "unknown"},
    {"lattice": "lieb-a", "d": 2, "z": 3, "pc": 0.7360, "type": "non_bipartite", "dual": "triangular"},
    {"lattice": "lieb-b", "d": 2, "z": 3, "pc": 0.6980, "type": "bipartite", "dual": "unknown"},
    {"lattice": "triangles", "d": 2, "z": 4, "pc": 0.5430, "type": "non_bipartite", "dual": "unknown"},

    # 3D 晶格 (原有)
    {"lattice": "sc", "d": 3, "z": 6, "pc": 0.3116, "type": "bipartite", "dual": "sc"},
    {"lattice": "bcc", "d": 3, "z": 8, "pc": 0.2458, "type": "bipartite", "dual": "fcc"},
    {"lattice": "fcc", "d": 3, "z": 12, "pc": 0.1188, "type": "non_bipartite", "dual": "bcc"},
    {"lattice": "diamond", "d": 3, "z": 4, "pc": 0.4319, "type": "bipartite", "dual": "diamond"},

    # 高维 (原有)
    {"lattice": "hyper_sc_4", "d": 4, "z": 8, "pc": 0.1967, "type": "bipartite", "dual": "self"},
    {"lattice": "hyper_sc_5", "d": 5, "z": 10, "pc": 0.1408, "type": "bipartite", "dual": "self"},
    {"lattice": "hyper_sc_6", "d": 6, "z": 12, "pc": 0.1095, "type": "bipartite", "dual": "self"},
    {"lattice": "hyper_sc_7", "d": 7, "z": 14, "pc": 0.0893, "type": "bipartite", "dual": "self"},
    {"lattice": "hyper_sc_8", "d": 8, "z": 16, "pc": 0.0751, "type": "bipartite", "dual": "self"},
    {"lattice": "hyper_sc_9", "d": 9, "z": 18, "pc": 0.0645, "type": "bipartite", "dual": "self"},
]


def build_extended_features():
    """构建扩展特征集"""
    samples = []
    for entry in EXTENDED_PERCOLATION_DATA:
        z = entry["z"]
        d = entry["d"]
        pc = entry["pc"]
        lattice_type = entry["type"]
        has_dual = 1 if entry["dual"] != "unknown" else 0
        is_bipartite = 1 if lattice_type == "bipartite" else 0

        # 基础特征
        sample = {
            "z": z,
            "d": d,
            "pc": pc,
            "z_inv": 1.0 / z if z > 0 else 0,
            "d_inv": 1.0 / d if d > 0 else 0,
            "z_minus_1": z - 1 if z > 1 else 0.001,
            "zd": z * d,
            "z_minus_1_inv": 1.0 / (z - 1) if z > 1 else 0,

            # 新增特征
            "is_bipartite": is_bipartite,
            "has_dual": has_dual,
            "z_squared": z * z,
            "z_cubed": z * z * z,
            "log_z": math.log(z) if z > 0 else 0,
            "log_d": math.log(d) if d > 0 else 0,
            "z_over_d": z / d if d > 0 else 0,
            "d_over_z": d / z if z > 0 else 0,
            "sqrt_z": math.sqrt(z) if z > 0 else 0,
            "sqrt_d": math.sqrt(d) if d > 0 else 0,

            # 拓扑特征
            "z_times_is_bip": z * is_bipartite,
            "z_inv_times_bip": (1.0 / z if z > 0 else 0) * is_bipartite,
        }

        samples.append(sample)

    return samples


def test_improved_formulas(samples):
    """测试改进的公式模板"""

    # 提取目标
    targets = [s["pc"] for s in samples]
    feature_vars = ["z", "d", "z_inv", "d_inv", "z_minus_1", "zd", "z_minus_1_inv",
                    "is_bipartite", "has_dual", "z_squared", "z_cubed", "log_z", "log_d",
                    "z_over_d", "d_over_z", "sqrt_z", "sqrt_d", "z_times_is_bip", "z_inv_times_bip"]

    # 候选公式 (改进版)
    candidates = [
        # 基础幂律
        ("幂律v1", "a * z^b * d^c"),
        ("幂律v2", "a * z^b * d^c + e"),
        ("幂律v3", "a * (z-1)^b * d^c"),

        # Bethe 修正
        ("修正Bethe v1", "1 / (z - 1) * (1 - a/d)"),
        ("修正Bethe v2", "1 / (z - 1 + a/d)"),
        ("修正Bethe v3", "1 / (z - 1) * (1 - a/z - b/d)"),

        # 二分图修正
        ("二分图修正 v1", "a * z^b * d^c + e * is_bipartite"),
        ("二分图修正 v2", "a * z^b * d^c * (1 + e * is_bipartite)"),
        ("二分图修正 v3", "(a * z^b + e * is_bipartite) * d^c"),

        # 对偶性修正
        ("对偶修正 v1", "a * z^b * d^c * (1 + e * has_dual)"),
        ("对偶修正 v2", "1/(z-1) * (1 - a/d + b*has_dual)"),

        # 混合模型
        ("混合 v1", "a * z^b * d^c + e * z_inv_times_bip"),
        ("混合 v2", "a * z^b * d^c + e * is_bipartite / z"),
        ("混合 v3", "a * z^b * d^c + e * log_z * is_bipartite"),

        # 分段思想
        ("分段 v1", "a * z^b * d^c + e * (z - 4) * is_bipartite"),
        ("分段 v2", "a * z^b * d^c + e * (d - 2) * is_bipartite"),

        # 高维精确
        ("高维精确 v1", "a * z^b * d^c + e / (z * d)"),
        ("高维精确 v2", "a * z^b * d^c + e / z^2"),

        # 复杂模型
        ("复杂 v1", "a * z^b * d^c * (1 + e * is_bipartite / z)"),
        ("复杂 v2", "a * (z - b * is_bipartite)^c * d^e"),
        ("复杂 v3", "a * z^b * d^c + e * (is_bipartite - 0.5) * z_inv"),

        # 原始最佳
        ("原始最佳", "3.493798 * z^(-1.190543) * d^(-0.360915)"),
    ]

    results = []

    for name, formula_template in candidates:
        try:
            # 参数优化
            opt_formula, opt_rmse, params = optimize_formula_params(
                formula_template, samples, feature_vars, "pc", max_iter=300
            )

            # 评估
            preds = evaluate_formula_batch(opt_formula, samples)
            r2 = compute_r2(preds, targets)

            if not math.isnan(r2) and not math.isinf(r2):
                results.append({
                    "name": name,
                    "formula": opt_formula,
                    "template": formula_template,
                    "rmse": opt_rmse,
                    "r2": r2,
                    "params": params,
                })

        except Exception as e:
            pass

    # 按 R² 排序
    results.sort(key=lambda x: x["r2"], reverse=True)

    return results


def print_results(results):
    """打印结果"""
    print("\n" + "=" * 80)
    print("🚀 改进后的公式候选 (按 R² 排序)")
    print("=" * 80)

    print(f"\n{'排名':<4} {'名称':<25} {'R²':<10} {'RMSE':<10} {'公式':<40}")
    print("-" * 100)

    for i, r in enumerate(results[:20], 1):
        formula_short = r["formula"][:37] + "..." if len(r["formula"]) > 40 else r["formula"]
        print(f"{i:<4} {r['name']:<25} {r['r2']:<10.4f} {r['rmse']:<10.4f} {formula_short:<40}")

    if results:
        best = results[0]
        print("\n" + "=" * 80)
        print("🏆 最佳公式:")
        print("=" * 80)
        print(f"名称: {best['name']}")
        print(f"公式: {best['formula']}")
        print(f"R²: {best['r2']:.6f}")
        print(f"RMSE: {best['rmse']:.6f}")
        print(f"参数: {best['params']}")

        # 对比原始
        print("\n📊 对比原始公式:")
        print(f"  原始: p_c ≈ 3.49·z^(-1.19)·d^(-0.36), R²=0.904")
        print(f"  改进: {best['name']}, R²={best['r2']:.4f}")
        print(f"  提升: ΔR² = {best['r2'] - 0.904:.4f} ({(best['r2'] - 0.904) / 0.904 * 100:.1f}%)")


def detailed_analysis(samples, best_formula):
    """详细分析最佳公式"""
    print("\n" + "=" * 80)
    print("📊 详细分析: 最佳公式的预测 vs 真实值")
    print("=" * 80)

    targets = [s["pc"] for s in samples]
    preds = evaluate_formula_batch(best_formula, samples)

    print(f"\n{'晶格':<20} {'d':>2} {'z':>2} {'二部图':>6} {'真实p_c':>8} {'预测':>8} {'误差':>8} {'相对误差':>8}")
    print("-" * 80)

    errors = []
    for i, (sample, pred, target) in enumerate(zip(samples, preds, targets)):
        lattice = EXTENDED_PERCOLATION_DATA[i]["lattice"]
        d = int(sample["d"])
        z = int(sample["z"])
        bip = "是" if sample["is_bipartite"] > 0.5 else "否"
        error = pred - target
        rel_error = abs(error / target) * 100 if target > 0 else 0
        errors.append(abs(error))

        print(f"{lattice:<20} {d:>2} {z:>2} {bip:>6} {target:>8.4f} {pred:>8.4f} {error:>+8.4f} {rel_error:>7.2f}%")

    print("-" * 80)
    print(f"平均绝对误差: {sum(errors)/len(errors):.4f}")
    print(f"最大绝对误差: {max(errors):.4f}")
    print(f"最小绝对误差: {min(errors):.4f}")


if __name__ == '__main__':
    random.seed(42)

    print("🔧 构建扩展数据集...")
    samples = build_extended_features()
    print(f"  数据点: {len(samples)}")
    print(f"  2D 晶格: {sum(1 for s in samples if s['d'] == 2)}")
    print(f"  3D 晶格: {sum(1 for s in samples if s['d'] == 3)}")
    print(f"  高维 (d≥4): {sum(1 for s in samples if s['d'] >= 4)}")

    print("\n🧪 测试改进公式...")
    results = test_improved_formulas(samples)

    print_results(results)

    if results:
        best = results[0]
        detailed_analysis(samples, best["formula"])

        # 保存最佳公式
        print("\n💾 保存结果...")
        with open(str(ROOT / 'improved_formula.txt'), "w") as f:
            f.write(f"最佳公式: {best['formula']}\n")
            f.write(f"R²: {best['r2']:.6f}\n")
            f.write(f"RMSE: {best['rmse']:.6f}\n")
            f.write(f"参数: {best['params']}\n")
        print("  已保存到 improved_formula.txt")
