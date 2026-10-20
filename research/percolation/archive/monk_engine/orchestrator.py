"""
扫地僧编排器 - 主工作循环 (Python 版)

工作流程:
1. 数据探索 (统计分析)
2. 提出公式假设 (基于物理先验 + 数据特征)
3. 并行运行 4 种符号回归方法
4. 拟合误差评估器分析
5. 多轮迭代, 利用评估反馈改进

移植自: src/lib/monk/orchestrator.ts
"""
import json
import time
import random
from typing import Dict, List, Optional, Tuple

from .sr_methods import (
    SRResult, run_genetic_programming, run_sindy,
    run_template_matching, run_physics_informed,
)
from .evaluator import evaluate_formula_full


def data_exploration(problem: Dict) -> Dict:
    """数据探索: 基本统计量和变量关系"""
    samples = problem['samples']
    feature_vars = problem['feature_vars']
    target_var = problem['target_var']

    stats = {}
    for v in feature_vars + [target_var]:
        vals = [s[v] for s in samples]
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((x - mean)**2 for x in vals) / n) ** 0.5
        mn, mx = min(vals), max(vals)
        sorted_v = sorted(vals)
        median = sorted_v[n // 2]
        stats[v] = {'mean': mean, 'std': std, 'min': mn, 'max': mx,
                     'median': median, 'n': n}

    # 简单相关性
    correlations = []
    target_vals = [s[target_var] for s in samples]
    target_mean = sum(target_vals) / len(target_vals)
    target_std = sum((x - target_mean)**2 for x in target_vals) ** 0.5

    for v in feature_vars:
        feat_vals = [s[v] for s in samples]
        feat_mean = sum(feat_vals) / len(feat_vals)
        feat_std = sum((x - feat_mean)**2 for x in feat_vals) ** 0.5
        if target_std > 0 and feat_std > 0:
            corr = sum((s[target_var] - target_mean) * (s[v] - feat_mean)
                        for s in samples) / (len(samples) * target_std * feat_std)
            correlations.append({'var': v, 'correlation': round(corr, 4)})

    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return {'stats': stats, 'correlations': correlations}


def generate_hypotheses(problem: Dict, exploration: Dict, round_num: int,
                         feedback: Optional[str] = None) -> List[Dict]:
    """基于物理直觉和数据特征生成公式假设"""
    feature_vars = problem['feature_vars']
    target_var = problem['target_var']
    domain = problem.get('domain', 'general')
    correlations = exploration.get('correlations', [])

    hypotheses = []

    if domain == 'percolation':
        # 渗流物理启发假设
        hypotheses.extend([
            {'formula': '1 / (z_minus_1)', 'rationale': 'Bethe 晶格近似: p_c = 1/(z-1)', 'confidence': 0.7},
            {'formula': 'a * z^b + c', 'rationale': '幂律标度假设', 'confidence': 0.6},
            {'formula': 'a / z + b / d + c', 'rationale': '线性组合: 配位数和维度的贡献', 'confidence': 0.5},
            {'formula': 'a * z^b * d^c', 'rationale': '幂律乘积形式', 'confidence': 0.5},
            {'formula': 'exp(a / z + b / d + c)', 'rationale': '指数形式的修正', 'confidence': 0.4},
            {'formula': 'a / (z - b) + c / (d - e) + f', 'rationale': '双修正 Bethe 近似', 'confidence': 0.6},
            {'formula': '1 / (z - 1) * (1 - a / (b * z * d))', 'rationale': 'Bethe + 有限维修正', 'confidence': 0.7},
            {'formula': 'a * exp(-b * z / d) + c', 'rationale': '指数衰减型', 'confidence': 0.3},
        ])

    elif domain == 'petroleum':
        hypotheses.extend([
            {'formula': 'a * phi^3 / (1 - phi)^2 + b', 'rationale': 'Carman-Kozeny 方程', 'confidence': 0.8},
            {'formula': 'a * phi^b + c', 'rationale': '幂律渗透率模型', 'confidence': 0.6},
            {'formula': 'a * phi^2 * d_grain^b * C^c + d', 'rationale': '扩展 Kozeny-Carman', 'confidence': 0.7},
            {'formula': 'a * (phi - b)^c * d_grain^d * S^e', 'rationale': '渗流阈值修正模型', 'confidence': 0.6},
            {'formula': 'a * phi^3 / ((1 - phi)^2 * S^2) + b', 'rationale': '含形状因子的 Kozeny', 'confidence': 0.7},
        ])

    elif domain == 'critical_phenomena':
        hypotheses.extend([
            {'formula': '2 * beta + d * nu - 2 * beta', 'rationale': '超尺度关系简化', 'confidence': 0.5},
            {'formula': 'a * d^b * nu^c + d * beta', 'rationale': '维度-指数幂律', 'confidence': 0.5},
            {'formula': 'nu * (2 * d - beta) / a + b', 'rationale': '修正标度假设', 'confidence': 0.4},
        ])

    # 通用假设
    if feature_vars:
        x = feature_vars[0]
        hypotheses.extend([
            {'formula': f'a * {x} + b', 'rationale': '线性基线', 'confidence': 0.3},
            {'formula': f'a * {x}^b + c', 'rationale': '幂律', 'confidence': 0.4},
        ])

    # 如果有反馈, 加入改进假设
    if feedback:
        hypotheses.append({
            'formula': hypotheses[0]['formula'] if hypotheses else 'x0',
            'rationale': f'基于上轮评估反馈改进',
            'confidence': 0.5,
            'source': 'evaluator_feedback',
        })

    # 去重并排序
    seen = set()
    unique = []
    for h in hypotheses:
        if h['formula'] not in seen:
            seen.add(h['formula'])
            unique.append(h)

    unique.sort(key=lambda h: h['confidence'], reverse=True)
    return unique[:8]


def run_monk(problem: Dict, metrics: Dict, max_rounds: int = 3,
             max_sr_steps: int = 6) -> Dict:
    """
    运行扫地僧 - 完整的多轮符号回归

    返回: {
        rounds: [...],
        best_formula: str,
        best_error: float,
        best_report: dict,
        summary: str,
    }
    """
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"🔬 扫地僧开始工作: {problem['title']}")
    print(f"{'='*60}")

    # 第 1 步: 数据探索
    print("\n📊 [步骤1] 数据探索...")
    exploration = data_exploration(problem)

    for corr in exploration['correlations'][:5]:
        print(f"  {corr['var']}: 相关系数 = {corr['correlation']:.4f}")

    best_formula = ''
    best_error = float('inf')
    best_report = None
    evaluator_feedback = None

    rounds = []

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'─'*50}")
        print(f"📝 第 {round_num}/{max_rounds} 轮")
        print(f"{'─'*50}")

        round_data = {'round': round_num}

        # 第 2 步: 提出公式假设
        print(f"\n💡 [步骤2] 提出公式假设...")
        hypotheses = generate_hypotheses(problem, exploration, round_num, evaluator_feedback)
        for i, h in enumerate(hypotheses[:5]):
            print(f"  {i+1}. {h['formula'][:50]:50s} (置信度: {h['confidence']:.2f})")
            print(f"     理由: {h['rationale']}")
        round_data['hypotheses'] = hypotheses

        # 第 3 步: 并行运行 4 种符号回归方法
        print(f"\n🔧 [步骤3] 并行运行 4 种符号回归方法...")
        seed_formula = hypotheses[0]['formula'] if hypotheses else feature_vars[0] if (feature_vars := problem['feature_vars']) else 'x'

        sr_results: List[SRResult] = []

        # 方法 1: 遗传规划
        print(f"  ├─ 遗传规划...", end=' ', flush=True)
        try:
            gp_result = run_genetic_programming(
                problem['samples'], problem['feature_vars'], problem['target_var'],
                seed_formula, max_steps=max_sr_steps * 5, population_size=40)
            sr_results.append(gp_result)
            print(f"RMSE={gp_result.rmse:.4f}, R²={gp_result.r2:.4f}")
        except Exception as e:
            print(f"失败: {e}")

        # 方法 2: SINDy
        print(f"  ├─ SINDy 稀疏回归...", end=' ', flush=True)
        try:
            sindy_result = run_sindy(
                problem['samples'], problem['feature_vars'], problem['target_var'],
                seed_formula, max_steps=max_sr_steps)
            sr_results.append(sindy_result)
            print(f"RMSE={sindy_result.rmse:.4f}, R²={sindy_result.r2:.4f}")
        except Exception as e:
            print(f"失败: {e}")

        # 方法 3: 模板匹配
        print(f"  ├─ 模板匹配...", end=' ', flush=True)
        try:
            tmpl_result = run_template_matching(
                problem['samples'], problem['feature_vars'], problem['target_var'],
                seed_formula, max_steps=max_sr_steps * 3)
            sr_results.append(tmpl_result)
            print(f"RMSE={tmpl_result.rmse:.4f}, R²={tmpl_result.r2:.4f}")
        except Exception as e:
            print(f"失败: {e}")

        # 方法 4: 物理引导搜索
        print(f"  └─ 物理引导搜索...", end=' ', flush=True)
        try:
            phys_result = run_physics_informed(
                problem['samples'], problem['feature_vars'], problem['target_var'],
                seed_formula, max_steps=max_sr_steps)
            sr_results.append(phys_result)
            print(f"RMSE={phys_result.rmse:.4f}, R²={phys_result.r2:.4f}")
        except Exception as e:
            print(f"失败: {e}")

        round_data['sr_results'] = [r.to_dict() for r in sr_results]

        # 第 4 步: 评估最佳公式
        successful = [r for r in sr_results if r.success and r.rmse < 1e6 and r.r2 > -100]
        if successful:
            best_sr = min(successful, key=lambda r: r.rmse)
            print(f"\n📈 [步骤4] 评估最佳公式: {best_sr.formula[:60]}...")

            report = evaluate_formula_full(problem, best_sr.formula, best_sr, metrics)

            if report['rmse'] < best_error:
                best_error = report['rmse']
                best_formula = report['formula']
                best_report = report

            print(f"  RMSE = {report['rmse']:.6f}")
            print(f"  R²   = {report['r2']:.4f}")
            print(f"  可解释性 = {report['interpretability']:.1f}/100")
            print(f"  综合评分 = {report['overall_score']:.1f}")
            print(f"  复杂度 = {report['complexity']} 节点")

            for insight in report['insights'][:3]:
                print(f"  💡 {insight}")

            # 生成下一轮反馈
            evaluator_feedback = build_feedback(report)
            round_data['evaluation'] = report
        else:
            print(f"\n⚠️  所有方法均失败")

        rounds.append(round_data)

    # 最终汇总
    total_time = int((time.time() - start_time) * 1000)
    print(f"\n{'='*60}")
    print(f"🏆 扫地僧完成! 耗时 {total_time/1000:.1f}s")
    print(f"{'='*60}")

    if best_formula:
        print(f"\n最佳公式:")
        print(f"  {best_formula}")
        print(f"\n性能指标:")
        print(f"  RMSE = {best_error:.6f}")
        if best_report:
            print(f"  R²   = {best_report['r2']:.4f}")
            print(f"  可解释性 = {best_report['interpretability']:.1f}/100")
            print(f"  综合评分 = {best_report['overall_score']:.1f}")

    # 汇总各方法表现
    print(f"\n方法比较:")
    all_sr = []
    for r in rounds:
        all_sr.extend(r.get('sr_results', []))
    for sr in sorted(all_sr, key=lambda x: x['rmse']):
        if sr['success']:
            print(f"  [{sr['method'][:20]:20s}] RMSE={sr['rmse']:.4f} R²={sr['r2']:.4f} | {sr['formula'][:50]}")

    summary = f"扫地僧在 {max_rounds} 轮中发现最佳公式: {best_formula}"
    if best_report:
        summary += f" (RMSE={best_error:.4f}, R²={best_report['r2']:.4f})"

    return {
        'rounds': rounds,
        'best_formula': best_formula,
        'best_error': best_error,
        'best_report': best_report,
        'summary': summary,
        'total_time_ms': total_time,
    }


def build_feedback(report: Dict) -> str:
    """生成评估反馈给下一轮"""
    lines = [f"上一轮最佳公式: {report['formula']}"]
    lines.append(f"RMSE: {report['rmse']:.4f}, R²: {report['r2']:.4f}")
    lines.append("误差模式:")
    for p in report.get('error_patterns', []):
        lines.append(f"  - {p}")
    lines.append("改进建议:")
    for r in report.get('recommendations', []):
        lines.append(f"  - {r}")
    return '\n'.join(lines)
