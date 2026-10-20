#!/usr/bin/env python3
"""
扫地僧竞技场 - 主运行脚本
========================
收集真实油田/渗流数据, 运行扫地僧引擎, 尝试发现渗流阈值的解析解。

用法:
  python run_monk_arena.py                    # 运行所有赛题
  python run_monk_arena.py --problem 0        # 只运行赛题0 (渗流阈值)
  python run_monk_arena.py --rounds 5         # 增加迭代轮数
"""
import sys
import os
import json
import time
import argparse
import random

# 确保可以导入 monk_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monk_engine.data_collector import (
    generate_percolation_problem,
    generate_porosity_permeability_problem,
    generate_scaling_law_problem,
    PERCOLATION_SITE_DATA,
    POROSITY_PERMEABILITY_DATA,
)
from monk_engine.orchestrator import run_monk


def print_banner():
    print(r"""
╔══════════════════════════════════════════════════════════════════╗
║                    🏛️  公式竞技场 · 扫地僧                       ║
║                  Formula Arena - The Monk Engine                 ║
╠══════════════════════════════════════════════════════════════════╣
║  4 种符号回归方法并行:                                           ║
║    1. 遗传规划 (Genetic Programming)                             ║
║    2. SINDy 稀疏回归                                             ║
║    3. 模板匹配 (Template Matching)                               ║
║    4. 物理引导搜索 (Physics-Informed)                            ║
║                                                                  ║
║  目标: 发现渗流阈值 p_c 的解析表达式                              ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_data_overview():
    """打印收集到的数据概览"""
    print("\n" + "="*60)
    print("📦 数据收集概览")
    print("="*60)

    print("\n📊 赛题A: 渗流阈值 p_c 与晶格几何")
    print(f"  数据来源: Stauffer & Aharony; Ziff (1992)")
    print(f"  数据点数: {len(PERCOLATION_SITE_DATA)}")
    print(f"  晶格类型: 蜂巢(2D), 正方(2D), 三角(2D), SC(3D), BCC(3D), FCC(3D)等")
    print(f"  维度范围: 2D ~ 9D")
    print(f"  配位数范围: z=3 ~ z=18")
    print("\n  关键数据:")
    for entry in PERCOLATION_SITE_DATA[:6]:
        print(f"    {entry['lattice']:15s} d={entry['d']} z={entry['z']:2d} → p_c={entry['pc']:.6f}")
    print(f"    ... (共 {len(PERCOLATION_SITE_DATA)} 条)")

    print("\n📊 赛题B: 油田岩心孔隙度-渗透率")
    print(f"  数据来源: Sahimi (2011); 砂岩油藏岩心分析")
    print(f"  数据点数: {len(POROSITY_PERMEABILITY_DATA)}")
    print(f"  孔隙度范围: 0.05 ~ 0.40")
    print(f"  渗透率范围: 0.001 ~ 3500 mD")

    print("\n📊 赛题C: 渗流临界指数标度律")
    print(f"  数据来源: 重整化群理论数值结果")
    print(f"  维度范围: 2D ~ 8D")
    print(f"  指数: β, γ, ν")


def save_results(results, problem, output_dir):
    """保存结果到 JSON"""
    output = {
        'problem_title': problem['title'],
        'problem_id': problem['id'],
        'domain': problem['domain'],
        'n_samples': len(problem['samples']),
        'feature_vars': problem['feature_vars'],
        'target_var': problem['target_var'],
        'best_formula': results['best_formula'],
        'best_error': results['best_error'],
        'total_time_ms': results['total_time_ms'],
        'summary': results['summary'],
        'rounds': [],
    }

    if results['best_report']:
        output['best_report'] = {
            'formula': results['best_report']['formula'],
            'rmse': results['best_report']['rmse'],
            'r2': results['best_report']['r2'],
            'interpretability': results['best_report']['interpretability'],
            'overall_score': results['best_report']['overall_score'],
            'insights': results['best_report']['insights'],
            'recommendations': results['best_report']['recommendations'],
            'error_patterns': results['best_report']['error_patterns'],
        }

    for r in results['rounds']:
        round_data = {
            'round': r['round'],
            'hypotheses_count': len(r.get('hypotheses', [])),
            'sr_results': r.get('sr_results', []),
        }
        if 'evaluation' in r:
            round_data['best_rmse'] = r['evaluation']['rmse']
            round_data['best_r2'] = r['evaluation']['r2']
        output['rounds'].append(round_data)

    filepath = os.path.join(output_dir, f"result_{problem['id']}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 结果已保存: {filepath}")

    return filepath


def run_problem(problem_idx, max_rounds, max_sr_steps, output_dir):
    """运行单个赛题"""
    problems = {
        0: ('渗流阈值 p_c', generate_percolation_problem),
        1: ('油田孔隙度-渗透率', generate_porosity_permeability_problem),
        2: ('临界指数标度律', generate_scaling_law_problem),
    }

    if problem_idx not in problems:
        print(f"❌ 无效赛题: {problem_idx}. 可选: {list(problems.keys())}")
        return

    name, gen_func = problems[problem_idx]
    problem = gen_func()

    print(f"\n{'█'*60}")
    print(f"  赛题 {problem_idx}: {name}")
    print(f"  数据点: {len(problem['samples'])}")
    print(f"  特征: {problem['feature_vars']}")
    print(f"  目标: {problem['target_var']}")
    print(f"{'█'*60}")

    # 根据赛题调整指标权重
    if problem_idx == 0:
        metrics = {'error_weight': 0.7, 'interp_weight': 0.3}
    elif problem_idx == 1:
        metrics = {'error_weight': 0.8, 'interp_weight': 0.2}
    else:
        metrics = {'error_weight': 0.6, 'interp_weight': 0.4}

    # 运行扫地僧
    results = run_monk(problem, metrics, max_rounds=max_rounds, max_sr_steps=max_sr_steps)

    # 保存结果
    save_results(results, problem, output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description='扫地僧竞技场 - 公式发现引擎')
    parser.add_argument('--problem', type=int, default=None,
                        help='赛题编号 (0=渗流阈值, 1=孔隙度-渗透率, 2=标度律)')
    parser.add_argument('--rounds', type=int, default=3, help='迭代轮数')
    parser.add_argument('--steps', type=int, default=6, help='每种SR方法的步数')
    parser.add_argument('--output-dir', type=str, default='.', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    random.seed(args.seed)

    # 设置输出目录
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print_banner()
    print_data_overview()

    start_time = time.time()

    if args.problem is not None:
        # 运行单个赛题
        run_problem(args.problem, args.rounds, args.steps, output_dir)
    else:
        # 运行所有赛题
        for idx in range(3):
            run_problem(idx, args.rounds, args.steps, output_dir)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"⏱️  总耗时: {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
