"""
拟合误差评估器 - 详细分析公式质量
分析误差分布、高误差样本、误差模式
生成洞察和改进建议
"""
import math
from typing import Dict, List, Tuple

from .math_engine import (
    evaluate_formula_batch, formula_complexity,
    compute_rmse, compute_mae, compute_r2,
)
from .sr_methods import SRResult


def compute_interpretability(formula: str, complexity: int) -> float:
    score = 100.0
    score -= min(complexity * 2, 40)
    n_parens = formula.count('(')
    score -= min(n_parens * 1.5, 20)
    import re
    complex_funcs = len(re.findall(r'sinh|cosh|tanh|atan|asin|acos', formula))
    score -= complex_funcs * 5
    if len(formula) > 100: score -= 15
    if len(formula) > 200: score -= 15
    if complexity < 8: score += 10
    if complexity < 5: score += 5
    return max(0, min(100, score))


def residual_stats(residuals: List[float]) -> Dict:
    n = len(residuals)
    if n == 0:
        return {'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'skewness': 0, 'kurtosis': 0}
    mean = sum(residuals) / n
    var = sum((x - mean)**2 for x in residuals) / n
    std = math.sqrt(var)
    mx = max(residuals)
    mn = min(residuals)
    skewness = sum(((x - mean) / std)**3 for x in residuals) / n if std > 0 else 0
    kurtosis = sum(((x - mean) / std)**4 for x in residuals) / n - 3 if std > 0 else 0
    return {'mean': mean, 'std': std, 'max': mx, 'min': mn,
            'skewness': skewness, 'kurtosis': kurtosis}


def analyze_error_patterns(samples, predictions, targets, feature_vars) -> List[str]:
    patterns = []
    residuals = [abs(t - p) for p, t in zip(predictions, targets)]
    if not residuals:
        return patterns
    sorted_r = sorted(residuals)
    median = sorted_r[len(sorted_r) // 2]

    high_idx = [i for i, r in enumerate(residuals) if r > median * 1.5]
    low_idx = [i for i, r in enumerate(residuals) if r < median * 0.5]

    if len(high_idx) < 3 or len(low_idx) < 3:
        return patterns

    for v in feature_vars:
        high_vals = [samples[i].get(v, 0) for i in high_idx]
        low_vals = [samples[i].get(v, 0) for i in low_idx]
        high_mean = sum(high_vals) / len(high_vals)
        low_mean = sum(low_vals) / len(low_vals)
        all_vals = [s.get(v, 0) for s in samples]
        overall_mean = sum(all_vals) / len(all_vals)
        overall_std = math.sqrt(sum((x - overall_mean)**2 for x in all_vals) / len(all_vals))

        if overall_std > 0:
            diff = (high_mean - low_mean) / overall_std
            if abs(diff) > 0.5:
                direction = '较大' if diff > 0 else '较小'
                patterns.append(
                    f'当 {v} 取值{direction}时 (高误差组均值 {high_mean:.2f} vs 低误差组 {low_mean:.2f}, '
                    f'差异 {diff:.2f}σ), 模型误差显著增大。')

    return patterns


def evaluate_formula_full(problem, formula, sr_result, metrics):
    """完整评估一个公式"""
    samples = problem['samples']
    feature_vars = problem['feature_vars']
    target_var = problem['target_var']

    targets = [s[target_var] for s in samples]

    try:
        predictions = evaluate_formula_batch(formula, samples)
    except:
        predictions = [float('nan')] * len(targets)

    rmse = compute_rmse(predictions, targets)
    mae = compute_mae(predictions, targets)
    r2 = compute_r2(predictions, targets)
    complexity = formula_complexity(formula)
    interpretability = compute_interpretability(formula, complexity)

    error_score = max(0, min(100, r2 * 100))
    overall_score = metrics['error_weight'] * error_score + metrics['interp_weight'] * interpretability

    residuals = [t - p for p, t in zip(predictions, targets)]
    valid_residuals = [r for r in residuals if not math.isnan(r) and math.isinf(r)]
    r_stats = residual_stats(valid_residuals)
    error_patterns = analyze_error_patterns(samples, predictions, targets, feature_vars)

    insights = []
    insights.append(f'公式复杂度为 {complexity} 个节点, 可解释性评分 {interpretability:.1f}/100。')

    if r2 > 0.9:
        insights.append(f'R² = {r2:.4f}, 模型能解释 {r2*100:.1f}% 的方差, 拟合质量优秀。')
    elif r2 > 0.7:
        insights.append(f'R² = {r2:.4f}, 拟合质量良好, 但仍有改进空间。')
    elif r2 > 0.4:
        insights.append(f'R² = {r2:.4f}, 拟合质量一般, 公式可能遗漏了重要变量或非线性关系。')
    else:
        insights.append(f'R² = {r2:.4f}, 拟合质量较差, 公式形式可能不正确。')

    if abs(r_stats.get('skewness', 0)) > 0.5:
        skew = r_stats['skewness']
        direction = '正偏(低估居多)' if skew > 0 else '负偏(高估居多)'
        insights.append(f'残差分布{direction} (偏度={skew:.2f}), 提示存在系统性偏差。')

    for p in error_patterns:
        insights.append(p)

    recommendations = []
    if r2 < 0.8:
        recommendations.append('尝试引入更多非线性项 (如平方、对数、指数变换)。')
    if error_patterns:
        recommendations.append('针对高误差样本的特征区间, 考虑分段拟合或添加交互项。')
    if complexity > 15:
        recommendations.append('公式过于复杂, 考虑简化结构以提升可解释性。')
    if r2 > 0.95 and complexity < 10:
        recommendations.append('当前公式已很好, 可作为最终解。')

    return {
        'formula': formula,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'complexity': complexity,
        'interpretability': interpretability,
        'overall_score': overall_score,
        'residual_stats': r_stats,
        'error_patterns': error_patterns,
        'insights': insights,
        'recommendations': recommendations,
    }
