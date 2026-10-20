from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
在52个真实数据点上运行扫地僧符号回归
====================================
对比: 符号回归 vs 当前幂律公式 vs ML方法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from monk_engine.sr_methods import (
    run_genetic_programming,
    run_sindy,
    run_template_matching,
    run_physics_informed,
)
from large_scale_benchmark import build_full_dataset, test_power_law, test_ml_methods


def run_symbolic_regression(df):
    """在真实数据上运行所有SR方法"""

    # 准备数据
    samples = []
    for _, row in df.iterrows():
        samples.append({
            'z': row['z'],
            'd': row['d'],
            'z_inv': row['z_inv'],
            'd_inv': row['d_inv'],
            'log_z': row['log_z'],
            'log_d': row['log_d'],
            'pc': row['pc'],
        })

    feature_vars = ['z', 'd', 'z_inv', 'd_inv', 'log_z', 'log_d']
    target_var = 'pc'

    # 当前最佳幂律公式
    hypothesis = "a * z^b * d^c"

    print("=" * 80)
    print("🔬 扫地僧符号回归竞技场 (52个真实数据点)")
    print("=" * 80)
    print(f"\n数据规模: {len(samples)} 个晶格")
    print(f"特征变量: {feature_vars}")
    print(f"目标变量: {target_var}")
    print(f"假设公式: {hypothesis}")

    results = {}

    # ============================================================
    # 方法 1: 遗传规划
    # ============================================================
    print("\n" + "=" * 80)
    print("🧬 方法1: 遗传规划 (Genetic Programming)")
    print("=" * 80)

    gp_result = run_genetic_programming(
        samples, feature_vars, target_var, hypothesis,
        max_steps=50, population_size=50
    )

    print(f"\n✓ 公式: {gp_result.formula}")
    print(f"  R² = {gp_result.r2:.6f}")
    print(f"  RMSE = {gp_result.rmse:.6f}")
    print(f"  MAE = {gp_result.mae:.6f}")
    print(f"  复杂度 = {gp_result.complexity}")
    print(f"  时间 = {gp_result.duration_ms}ms")

    results['遗传规划'] = gp_result

    # ============================================================
    # 方法 2: SINDy 稀疏回归
    # ============================================================
    print("\n" + "=" * 80)
    print("🔍 方法2: SINDy 稀疏回归")
    print("=" * 80)

    sindy_result = run_sindy(
        samples, feature_vars, target_var, hypothesis,
        max_steps=15
    )

    print(f"\n✓ 公式: {sindy_result.formula}")
    print(f"  R² = {sindy_result.r2:.6f}")
    print(f"  RMSE = {sindy_result.rmse:.6f}")
    print(f"  MAE = {sindy_result.mae:.6f}")
    print(f"  复杂度 = {sindy_result.complexity}")
    print(f"  时间 = {sindy_result.duration_ms}ms")

    results['SINDy'] = sindy_result

    # ============================================================
    # 方法 3: 模板匹配
    # ============================================================
    print("\n" + "=" * 80)
    print("🎯 方法3: 模板匹配 (Template Matching)")
    print("=" * 80)

    template_result = run_template_matching(
        samples, feature_vars, target_var, hypothesis,
        max_steps=20
    )

    print(f"\n✓ 公式: {template_result.formula}")
    print(f"  R² = {template_result.r2:.6f}")
    print(f"  RMSE = {template_result.rmse:.6f}")
    print(f"  MAE = {template_result.mae:.6f}")
    print(f"  复杂度 = {template_result.complexity}")
    print(f"  时间 = {template_result.duration_ms}ms")
    print(f"  备注: {template_result.notes}")

    results['模板匹配'] = template_result

    # ============================================================
    # 方法 4: 物理引导搜索
    # ============================================================
    print("\n" + "=" * 80)
    print("⚛️  方法4: 物理引导搜索 (Physics-Informed)")
    print("=" * 80)

    physics_result = run_physics_informed(
        samples, feature_vars, target_var, hypothesis,
        max_steps=15
    )

    print(f"\n✓ 公式: {physics_result.formula}")
    print(f"  R² = {physics_result.r2:.6f}")
    print(f"  RMSE = {physics_result.rmse:.6f}")
    print(f"  MAE = {physics_result.mae:.6f}")
    print(f"  复杂度 = {physics_result.complexity}")
    print(f"  时间 = {physics_result.duration_ms}ms")

    results['物理引导'] = physics_result

    return results


def main():
    print("=" * 80)
    print("🏆 扫地僧竞技场 — 符号回归 vs 当前幂律 vs ML")
    print("=" * 80)

    # 构建数据集
    df = build_full_dataset()
    print(f"\n📊 数据集: {len(df)} 个晶格")
    print(f"  2D: {len(df[df['d']==2])}, 3D: {len(df[df['d']==3])}, 高维: {len(df[df['d']>=4])}")

    # 运行符号回归
    sr_results = run_symbolic_regression(df)

    # 当前最佳幂律
    print("\n" + "=" * 80)
    print("📐 当前最佳幂律公式")
    print("=" * 80)

    params, r2_power, rmse_power = test_power_law(df)
    print(f"\n公式: p_c = {params[0]:.4f} · z^({params[1]:.4f}) · d^({params[2]:.4f})")
    print(f"  R² = {r2_power:.6f}")
    print(f"  RMSE = {rmse_power:.6f}")
    print(f"  MAE ≈ {rmse_power:.6f}")

    # ML方法
    print("\n" + "=" * 80)
    print("🤖 机器学习方法 (LOOCV)")
    print("=" * 80)

    ml_results = test_ml_methods(df)
    best_ml_name = min(ml_results.items(), key=lambda x: x[1]['mae'])[0]
    best_ml_mae = ml_results[best_ml_name]['mae']

    print(f"\n最佳ML: {best_ml_name}")
    print(f"  MAE = {best_ml_mae:.6f}")

    # ============================================================
    # 综合对比
    # ============================================================
    print("\n" + "=" * 80)
    print("🏆 综合对比")
    print("=" * 80)

    all_methods = []

    # 添加SR方法
    for name, result in sr_results.items():
        all_methods.append({
            'method': f"扫地僧·{name}",
            'R²': result.r2,
            'RMSE': result.rmse,
            'MAE': result.mae,
            'params': result.complexity,
            'formula': result.formula[:60],
        })

    # 添加当前幂律
    all_methods.append({
        'method': '当前幂律 (z^b·d^c)',
        'R²': r2_power,
        'RMSE': rmse_power,
        'MAE': rmse_power,
        'params': 3,
        'formula': f"{params[0]:.2f}·z^({params[1]:.2f})·d^({params[2]:.2f})",
    })

    # 添加ML方法
    for name, res in ml_results.items():
        all_methods.append({
            'method': f"ML·{name}",
            'R²': None,
            'RMSE': None,
            'MAE': res['mae'],
            'params': 7 if '6特征' in name else 2,
            'formula': '线性组合',
        })

    # 按MAE排序
    all_methods.sort(key=lambda x: x['MAE'])

    print(f"\n{'方法':<30} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'参数':<6}")
    print("-" * 80)

    for m in all_methods:
        r2_str = f"{m['R²']:.4f}" if m['R²'] is not None else "N/A"
        rmse_str = f"{m['RMSE']:.4f}" if m['RMSE'] is not None else "N/A"
        print(f"{m['method']:<30} {r2_str:<8} {rmse_str:<8} {m['MAE']:<8.4f} {m['params']:<6}")

    # ============================================================
    # 最佳公式详情
    # ============================================================
    print("\n" + "=" * 80)
    print("🎯 最佳公式详情")
    print("=" * 80)

    best_sr = min(sr_results.values(), key=lambda r: r.mae)

    print(f"\n{'方法':<20} {best_sr.method}")
    print(f"{'公式':<20} {best_sr.formula}")
    print(f"{'R²':<20} {best_sr.r2:.6f}")
    print(f"{'MAE':<20} {best_sr.mae:.6f}")
    print(f"{'RMSE':<20} {best_sr.rmse:.6f}")
    print(f"{'复杂度':<20} {best_sr.complexity}")

    # ============================================================
    # 结论
    # ============================================================
    print("\n" + "=" * 80)
    print("💡 结论")
    print("=" * 80)

    best_sr_mae = min(r.mae for r in sr_results.values())

    print(f"\n在52个真实数据点上:")
    print(f"\n1. 当前最佳幂律:")
    print(f"   MAE = {rmse_power:.4f}, R² = {r2_power:.4f}")

    print(f"\n2. 扫地僧符号回归:")
    print(f"   最佳MAE = {best_sr_mae:.4f} ({best_sr.method})")
    print(f"   公式: {best_sr.formula}")

    print(f"\n3. ML方法:")
    print(f"   最佳MAE = {best_ml_mae:.4f}")

    print(f"\n4. 对比:")
    improvement = (rmse_power - best_sr_mae) / rmse_power * 100
    if best_sr_mae < rmse_power:
        print(f"   ✓ 扫地僧找到了更好的公式!")
        print(f"   ✓ MAE提升: {improvement:.1f}%")
    else:
        print(f"   ⚠️  当前幂律已经很好")
        print(f"   ⚠️  扫地僧未能超越 (差距: {abs(improvement):.1f}%)")

    # 保存结果
    print("\n" + "=" * 80)
    print("💾 保存结果")
    print("=" * 80)

    output_file = str(ROOT / 'sr_arena_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("扫地僧符号回归竞技场结果 (52个真实数据点)\n")
        f.write("=" * 80 + "\n\n")

        f.write("所有方法对比:\n")
        f.write("-" * 80 + "\n")
        for m in all_methods:
            f.write(f"{m['method']:<30} MAE={m['MAE']:.4f}\n")

        f.write("\n\n最佳公式详情:\n")
        f.write("-" * 80 + "\n")
        f.write(f"方法: {best_sr.method}\n")
        f.write(f"公式: {best_sr.formula}\n")
        f.write(f"R²: {best_sr.r2:.6f}\n")
        f.write(f"MAE: {best_sr.mae:.6f}\n")
        f.write(f"RMSE: {best_sr.rmse:.6f}\n")
        f.write(f"复杂度: {best_sr.complexity}\n")

    print(f"\n✓ 结果已保存: {output_file}")


if __name__ == '__main__':
    main()
