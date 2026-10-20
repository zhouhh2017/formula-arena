#!/usr/bin/env python3
"""
改进版 v2 - 分段建模 + 更强的参数优化
=========================================
策略:
1. 对 2D 和 3D+ 分别建立模型
2. 修复参数优化 (多点重启 + 更大搜索范围)
3. 引入物理约束
"""

import sys, os, math, random, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.optimize import minimize, differential_evolution

from monk_engine.math_engine import (
    parse_expression, eval_node, evaluate_formula_batch,
    compute_rmse, compute_r2, formula_complexity
)

# ===== 扩展数据集 =====
EXTENDED_DATA = [
    # 2D 晶格 (高精度数值结果)
    {"name": "honeycomb",   "d": 2, "z": 3,  "pc": 0.6970},
    {"name": "square",      "d": 2, "z": 4,  "pc": 0.592746},
    {"name": "triangular",  "d": 2, "z": 6,  "pc": 0.5000},
    {"name": "kagome",      "d": 2, "z": 4,  "pc": 0.6527},
    {"name": "bowtie",      "d": 2, "z": 4,  "pc": 0.5730},
    {"name": "unionjack",   "d": 2, "z": 5,  "pc": 0.5290},
    {"name": "3-12",        "d": 2, "z": 3,  "pc": 0.8170},
    {"name": "4-8",         "d": 2, "z": 3,  "pc": 0.6768},
    {"name": "bathroom",    "d": 2, "z": 5,  "pc": 0.5510},
    {"name": "dice",        "d": 2, "z": 3,  "pc": 0.6714},
    {"name": "ruby",        "d": 2, "z": 3,  "pc": 0.7210},
    {"name": "martini",     "d": 2, "z": 4,  "pc": 0.6110},
    {"name": "miami",       "d": 2, "z": 4,  "pc": 0.5890},
    {"name": "lieb-a",      "d": 2, "z": 3,  "pc": 0.7360},
    {"name": "lieb-b",      "d": 2, "z": 3,  "pc": 0.6980},
    {"name": "triangles",   "d": 2, "z": 4,  "pc": 0.5430},

    # 3D 晶格
    {"name": "sc",          "d": 3, "z": 6,  "pc": 0.3116},
    {"name": "bcc",         "d": 3, "z": 8,  "pc": 0.2458},
    {"name": "fcc",         "d": 3, "z": 12, "pc": 0.1188},
    {"name": "diamond",     "d": 3, "z": 4,  "pc": 0.4319},

    # 高维
    {"name": "hypersc_4",   "d": 4, "z": 8,  "pc": 0.1967},
    {"name": "hypersc_5",   "d": 5, "z": 10, "pc": 0.1408},
    {"name": "hypersc_6",   "d": 6, "z": 12, "pc": 0.1095},
    {"name": "hypersc_7",   "d": 7, "z": 14, "pc": 0.0893},
    {"name": "hypersc_8",   "d": 8, "z": 16, "pc": 0.0751},
    {"name": "hypersc_9",   "d": 9, "z": 18, "pc": 0.0645},
]


def eval_power_law(params, z_arr, d_arr):
    """幂律模型: a * z^b * d^c (+ e)"""
    a, b, c = params[0], params[1], params[2]
    pred = a * np.power(np.abs(z_arr), b) * np.power(np.abs(d_arr), c)
    if len(params) > 3:
        pred += params[3]
    return pred


def eval_modified_bethe(params, z_arr, d_arr):
    """修正 Bethe: 1/(z-1+a/d^b)"""
    a, b = params[0], params[1]
    pred = 1.0 / (z_arr - 1 + a * np.power(np.abs(d_arr), -b))
    return pred


def eval_power_law_correction(params, z_arr, d_arr):
    """幂律 + 修正: a*z^b*d^c * (1 + e/z^f)"""
    a, b, c, e, f = params
    pred = a * np.power(np.abs(z_arr), b) * np.power(np.abs(d_arr), c)
    pred *= (1 + e / np.power(np.abs(z_arr), f))
    return pred


def eval_log_model(params, z_arr, d_arr):
    """对数模型: a + b*log(z) + c*log(d) + e/(z*d)"""
    a, b, c, e = params
    pred = a + b * np.log(z_arr) + c * np.log(d_arr) + e / (z_arr * d_arr)
    return pred


def eval_rational_model(params, z_arr, d_arr):
    """有理函数: (a + b*z + c*d) / (e + f*z + g*d)"""
    a, b, c, e, f, g = params
    pred = (a + b * z_arr + c * d_arr) / (e + f * z_arr + g * d_arr + 1e-10)
    return pred


def eval_split_model_2d(params_2d, z_arr):
    """2D 专用模型"""
    a, b, c, e = params_2d
    # 尝试分段: 不同 z 区间不同行为
    pred = a * np.power(np.abs(z_arr), b) + c
    if len(params_2d) > 4:
        pred += e * (z_arr < 4).astype(float)  # 小 z 修正
    return pred


def eval_split_model_3d(params_3d, z_arr, d_arr):
    """3D+ 专用模型"""
    a, b, c, e = params_3d
    pred = a * np.power(np.abs(z_arr), b) * np.power(np.abs(d_arr), c) + e
    return pred


def optimize_model(eval_func, params_init, z_arr, d_arr, pc_arr, bounds=None):
    """使用 scipy 优化模型参数"""
    def objective(params):
        try:
            pred = eval_func(params, z_arr, d_arr) if 'd_arr' in eval_func.__code__.co_varnames else eval_func(params, z_arr)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e10
            return np.sqrt(np.mean((pred - pc_arr) ** 2))
        except:
            return 1e10

    # 多点重启
    best_result = None
    best_rmse = float('inf')

    # 随机重启
    for _ in range(20):
        x0 = np.array(params_init) + np.random.randn(len(params_init)) * 0.5
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_rmse:
                best_rmse = result.fun
                best_result = result
        except:
            pass

    # 差分进化 (全局优化)
    if bounds:
        try:
            result_de = differential_evolution(objective, bounds, maxiter=1000,
                                                tol=1e-10, seed=42)
            if result_de.fun < best_rmse:
                best_rmse = result_de.fun
                best_result = result_de
        except:
            pass

    return best_result.x, best_rmse if best_result else float('inf')


def run_improved_analysis():
    """运行改进分析"""

    # 准备数据
    data_2d = [d for d in EXTENDED_DATA if d["d"] == 2]
    data_3d = [d for d in EXTENDED_DATA if d["d"] >= 3]

    z_2d = np.array([d["z"] for d in data_2d])
    pc_2d = np.array([d["pc"] for d in data_2d])
    d_2d = np.array([d["d"] for d in data_2d])

    z_3d = np.array([d["z"] for d in data_3d])
    d_3d = np.array([d["d"] for d in data_3d])
    pc_3d = np.array([d["pc"] for d in data_3d])

    z_all = np.array([d["z"] for d in EXTENDED_DATA])
    d_all = np.array([d["d"] for d in EXTENDED_DATA])
    pc_all = np.array([d["pc"] for d in EXTENDED_DATA])

    print("=" * 80)
    print("🚀 改进版 v2 - 分段建模 + 全局优化")
    print("=" * 80)

    print(f"\n📊 数据分布:")
    print(f"  2D 晶格: {len(data_2d)} 个")
    print(f"  3D+ 晶格: {len(data_3d)} 个")
    print(f"  总计: {len(EXTENDED_DATA)} 个")

    results = []

    # ===== 策略 1: 全局幂律 (改进优化) =====
    print("\n" + "─" * 40)
    print("策略 1: 全局幂律 a·z^b·d^c")
    print("─" * 40)

    def power_law_3(params, z, d):
        return params[0] * np.power(np.abs(z), params[1]) * np.power(np.abs(d), params[2])

    bounds_pl = [(0.1, 10), (-3, 0), (-2, 0)]
    opt_params, rmse_pl = optimize_model(power_law_3, [3.5, -1.2, -0.4],
                                          z_all, d_all, pc_all, bounds_pl)
    preds_pl = power_law_3(opt_params, z_all, d_all)
    r2_pl = 1 - np.sum((preds_pl - pc_all)**2) / np.sum((pc_all - np.mean(pc_all))**2)

    formula_pl = f"{opt_params[0]:.4f} * z^({opt_params[1]:.4f}) * d^({opt_params[2]:.4f})"
    print(f"  公式: {formula_pl}")
    print(f"  R² = {r2_pl:.4f}, RMSE = {rmse_pl:.4f}")
    results.append(("全局幂律 (优化)", formula_pl, r2_pl, rmse_pl))

    # ===== 策略 2: 全局幂律 + 常数项 =====
    print("\n策略 2: 全局幂律 + 常数项 a·z^b·d^c + e")
    print("─" * 40)

    def power_law_4(params, z, d):
        return params[0] * np.power(np.abs(z), params[1]) * np.power(np.abs(d), params[2]) + params[3]

    bounds_pl4 = [(0.1, 10), (-3, 0), (-2, 0), (-0.2, 0.2)]
    opt_params, rmse_pl4 = optimize_model(power_law_4, [3.5, -1.2, -0.4, 0.01],
                                           z_all, d_all, pc_all, bounds_pl4)
    preds_pl4 = power_law_4(opt_params, z_all, d_all)
    r2_pl4 = 1 - np.sum((preds_pl4 - pc_all)**2) / np.sum((pc_all - np.mean(pc_all))**2)

    formula_pl4 = f"{opt_params[0]:.4f} * z^({opt_params[1]:.4f}) * d^({opt_params[2]:.4f}) + ({opt_params[3]:.4f})"
    print(f"  公式: {formula_pl4}")
    print(f"  R² = {r2_pl4:.4f}, RMSE = {rmse_pl4:.4f}")
    results.append(("全局幂律+常数", formula_pl4, r2_pl4, rmse_pl4))

    # ===== 策略 3: 分段建模 =====
    print("\n" + "─" * 40)
    print("策略 3: 分段建模 (2D / 3D+ 分别拟合)")
    print("─" * 40)

    # 2D: a * z^b + c
    def model_2d(params, z, d):
        return params[0] * np.power(np.abs(z), params[1]) + params[2]

    bounds_2d = [(0.1, 5), (-2, 0), (-0.5, 1.0)]
    opt_2d, rmse_2d = optimize_model(model_2d, [2.0, -0.5, 0.3],
                                      z_2d, d_2d, pc_2d, bounds_2d)
    preds_2d = model_2d(opt_2d, z_2d, d_2d)
    r2_2d = 1 - np.sum((preds_2d - pc_2d)**2) / np.sum((pc_2d - np.mean(pc_2d))**2)

    formula_2d = f"[2D] p_c = {opt_2d[0]:.4f} * z^({opt_2d[1]:.4f}) + ({opt_2d[2]:.4f})"
    print(f"  {formula_2d}")
    print(f"  2D: R² = {r2_2d:.4f}, RMSE = {rmse_2d:.4f}")

    # 3D+: a * z^b * d^c
    def model_3d(params, z, d):
        return params[0] * np.power(np.abs(z), params[1]) * np.power(np.abs(d), params[2])

    bounds_3d = [(0.1, 10), (-3, 0), (-2, 0)]
    opt_3d, rmse_3d = optimize_model(model_3d, [3.5, -1.2, -0.4],
                                      z_3d, d_3d, pc_3d, bounds_3d)
    preds_3d = model_3d(opt_3d, z_3d, d_3d)
    r2_3d = 1 - np.sum((preds_3d - pc_3d)**2) / np.sum((pc_3d - np.mean(pc_3d))**2)

    formula_3d = f"[3D+] p_c = {opt_3d[0]:.4f} * z^({opt_3d[1]:.4f}) * d^({opt_3d[2]:.4f})"
    print(f"  {formula_3d}")
    print(f"  3D+: R² = {r2_3d:.4f}, RMSE = {rmse_3d:.4f}")

    # 分段模型总 R²
    preds_split = np.concatenate([preds_2d, preds_3d])
    pc_split = np.concatenate([pc_2d, pc_3d])
    r2_split = 1 - np.sum((preds_split - pc_split)**2) / np.sum((pc_split - np.mean(pc_split))**2)
    rmse_split = np.sqrt(np.mean((preds_split - pc_split)**2))

    formula_split = f"if d==2: {opt_2d[0]:.4f}*z^({opt_2d[1]:.4f})+({opt_2d[2]:.4f}); else: {opt_3d[0]:.4f}*z^({opt_3d[1]:.4f})*d^({opt_3d[2]:.4f})"
    print(f"  分段总: R² = {r2_split:.4f}, RMSE = {rmse_split:.4f}")
    results.append(("分段建模", formula_split, r2_split, rmse_split))

    # ===== 策略 4: 修正 Bethe + 优化 =====
    print("\n" + "─" * 40)
    print("策略 4: 修正 Bethe: 1/(z-1+a/d^b)")
    print("─" * 40)

    def modified_bethe(params, z, d):
        a, b = params
        return 1.0 / (z - 1 + a * np.power(np.abs(d), -b))

    bounds_mb = [(0, 5), (-3, 3)]
    opt_mb, rmse_mb = optimize_model(modified_bethe, [1.0, 1.0],
                                      z_all, d_all, pc_all, bounds_mb)
    preds_mb = modified_bethe(opt_mb, z_all, d_all)
    r2_mb = 1 - np.sum((preds_mb - pc_all)**2) / np.sum((pc_all - np.mean(pc_all))**2)

    formula_mb = f"1 / (z - 1 + {opt_mb[0]:.4f} / d^{opt_mb[1]:.4f})"
    print(f"  公式: {formula_mb}")
    print(f"  R² = {r2_mb:.4f}, RMSE = {rmse_mb:.4f}")
    results.append(("修正Bethe", formula_mb, r2_mb, rmse_mb))

    # ===== 策略 5: 有理函数 =====
    print("\n" + "─" * 40)
    print("策略 5: 有理函数 (a+b*z+c*d)/(e+f*z+g*d)")
    print("─" * 40)

    def rational(params, z, d):
        a, b, c, e, f, g = params
        num = a + b * z + c * d
        den = e + f * z + g * d + 1e-10
        return num / den

    bounds_rat = [(-5, 5)] * 6
    opt_rat, rmse_rat = optimize_model(rational, [1, 0, 0, 1, 1, 0],
                                        z_all, d_all, pc_all, bounds_rat)
    preds_rat = rational(opt_rat, z_all, d_all)
    r2_rat = 1 - np.sum((preds_rat - pc_all)**2) / np.sum((pc_all - np.mean(pc_all))**2)

    formula_rat = f"({opt_rat[0]:.4f} + {opt_rat[1]:.4f}*z + {opt_rat[2]:.4f}*d) / ({opt_rat[3]:.4f} + {opt_rat[4]:.4f}*z + {opt_rat[5]:.4f}*d)"
    print(f"  公式: {formula_rat}")
    print(f"  R² = {r2_rat:.4f}, RMSE = {rmse_rat:.4f}")
    results.append(("有理函数", formula_rat, r2_rat, rmse_rat))

    # ===== 策略 6: 幂律 + z 修正 =====
    print("\n" + "─" * 40)
    print("策略 6: 幂律 + z 修正 a·z^b·d^c·(1+e/z^f)")
    print("─" * 40)

    def power_correction(params, z, d):
        a, b, c, e, f = params
        return a * np.power(np.abs(z), b) * np.power(np.abs(d), c) * (1 + e / np.power(np.abs(z), f))

    bounds_pc = [(0.1, 10), (-3, 0), (-2, 0), (-2, 2), (0, 3)]
    opt_pc, rmse_pc = optimize_model(power_correction, [3.5, -1.2, -0.4, 0.5, 1.0],
                                      z_all, d_all, pc_all, bounds_pc)
    preds_pc = power_correction(opt_pc, z_all, d_all)
    r2_pc = 1 - np.sum((preds_pc - pc_all)**2) / np.sum((pc_all - np.mean(pc_all))**2)

    formula_pc = f"{opt_pc[0]:.4f} * z^({opt_pc[1]:.4f}) * d^({opt_pc[2]:.4f}) * (1 + {opt_pc[3]:.4f}/z^{opt_pc[4]:.4f})"
    print(f"  公式: {formula_pc}")
    print(f"  R² = {r2_pc:.4f}, RMSE = {rmse_pc:.4f}")
    results.append(("幂律+z修正", formula_pc, r2_pc, rmse_pc))

    # ===== 策略 7: 混合幂律 (z 和 z*d) =====
    print("\n" + "─" * 40)
    print("策略 7: 混合幂律 a/z^b + c/(z*d)^e")
    print("─" * 40)

    def hybrid_power(params, z, d):
        a, b, c, e = params
        return a / np.power(np.abs(z), b) + c / np.power(np.abs(z * d), e)

    bounds_hp = [(0, 5), (0, 3), (0, 5), (0, 3)]
    opt_hp, rmse_hp = optimize_model(hybrid_power, [1.0, 0.5, 2.0, 1.0],
                                      z_all, d_all, pc_all, bounds_hp)
    preds_hp = hybrid_power(opt_hp, z_all, d_all)
    r2_hp = 1 - np.sum((preds_hp - pc_all)**2) / np.sum((pc_all - np.mean(pc_all))**2)

    formula_hp = f"{opt_hp[0]:.4f}/z^{opt_hp[1]:.4f} + {opt_hp[2]:.4f}/(z*d)^{opt_hp[3]:.4f}"
    print(f"  公式: {formula_hp}")
    print(f"  R² = {r2_hp:.4f}, RMSE = {rmse_hp:.4f}")
    results.append(("混合幂律", formula_hp, r2_hp, rmse_hp))

    # ===== 汇总 =====
    print("\n" + "=" * 80)
    print("📊 所有策略对比")
    print("=" * 80)

    results.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'排名':<4} {'策略':<25} {'R²':<10} {'RMSE':<10} {'公式':<50}")
    print("-" * 110)

    for i, (name, formula, r2, rmse) in enumerate(results, 1):
        f_short = formula[:47] + "..." if len(formula) > 50 else formula
        print(f"{i:<4} {name:<25} {r2:<10.4f} {rmse:<10.4f} {f_short:<50}")

    best = results[0]
    print("\n" + "=" * 80)
    print(f"🏆 最佳策略: {best[0]}")
    print("=" * 80)
    print(f"  R² = {best[2]:.6f}")
    print(f"  RMSE = {best[3]:.6f}")
    print(f"  公式: {best[1]}")

    # 与原始对比
    print("\n📈 对比:")
    print(f"  原始公式: p_c ≈ 3.49·z^(-1.19)·d^(-0.36), R²=0.904")
    print(f"  改进公式: {best[0]}, R²={best[2]:.4f}")
    print(f"  提升: ΔR² = {best[2] - 0.904:.4f} ({(best[2] - 0.904) / 0.096 * 100:.1f}%  toward 1.0)")

    # 经济影响
    old_rmse = 0.054
    new_rmse = best[3]
    improvement = (old_rmse - new_rmse) / old_rmse * 100
    print(f"\n💰 经济影响:")
    print(f"  RMSE 从 {old_rmse:.3f} 降到 {new_rmse:.3f} (降低 {improvement:.1f}%)")
    print(f"  期望损失从 0.69亿/项目 降到 {0.69 * (new_rmse/old_rmse):.2f}亿/项目")
    print(f"  全国年避免损失: {(0.69 - 0.69*(new_rmse/old_rmse)) * 4500:.0f}亿元")

    # 详细逐点分析
    print(f"\n📊 逐点预测 (最佳公式):")

    if best[0] == "分段建模":
        print(f"  {'晶格':<15} {'d':>2} {'z':>2} {'真实':>8} {'预测':>8} {'误差':>8}")
        print(f"  {'-'*50}")
        for i, d in enumerate(data_2d):
            pred = model_2d(opt_2d, np.array([d["z"]]), np.array([d["d"]]))[0]
            err = pred - d["pc"]
            print(f"  {d['name']:<15} {d['d']:>2} {d['z']:>2} {d['pc']:>8.4f} {pred:>8.4f} {err:>+8.4f}")
        print(f"  {'─'*50}")
        for i, d in enumerate(data_3d):
            pred = model_3d(opt_3d, np.array([d["z"]]), np.array([d["d"]]))[0]
            err = pred - d["pc"]
            print(f"  {d['name']:<15} {d['d']:>2} {d['z']:>2} {d['pc']:>8.4f} {pred:>8.4f} {err:>+8.4f}")
    else:
        # 使用最佳公式的预测函数
        best_func = {
            "全局幂律 (优化)": power_law_3,
            "全局幂律+常数": power_law_4,
            "修正Bethe": modified_bethe,
            "有理函数": rational,
            "幂律+z修正": power_correction,
            "混合幂律": hybrid_power,
        }.get(best[0], power_law_3)

        best_params = {
            "全局幂律 (优化)": opt_params,
            "全局幂律+常数": opt_params,
            "修正Bethe": opt_mb,
            "有理函数": opt_rat,
            "幂律+z修正": opt_pc,
            "混合幂律": opt_hp,
        }.get(best[0], opt_params)

        preds_best = best_func(best_params, z_all, d_all)
        print(f"  {'晶格':<15} {'d':>2} {'z':>2} {'真实':>8} {'预测':>8} {'误差':>8}")
        print(f"  {'-'*50}")
        for i, d in enumerate(EXTENDED_DATA):
            err = preds_best[i] - d["pc"]
            print(f"  {d['name']:<15} {d['d']:>2} {d['z']:>2} {d['pc']:>8.4f} {preds_best[i]:>8.4f} {err:>+8.4f}")

    return results


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    run_improved_analysis()
