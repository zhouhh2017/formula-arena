/**
 * 拟合误差评估器 - 定制化的模式发现器
 * 分析误差的细粒度特征: 哪些样本造成误差, 这些样本有什么特征
 * 对符号回归结果进行全面洞察, 生成落盘到 wiki 的报告
 */
import { chat } from './llm';
import type { DataSample, SRResult, EvaluationReport, ProblemData } from './types';
import { evaluateFormulaBatch, formulaComplexity, computeRMSE, computeMAE, computeR2 } from './sr/utils';

// 计算可解释性评分 (0-100)
function computeInterpretability(formula: string, complexity: number): number {
  let score = 100;
  // 复杂度惩罚
  score -= Math.min(complexity * 2, 40);
  // 深层嵌套惩罚
  const nestedParens = (formula.match(/\(/g) || []).length;
  score -= Math.min(nestedParens * 1.5, 20);
  // 使用了不常见函数惩罚
  const complexFuncs = (formula.match(/(sinh|cosh|tanh|atan|asin|acos)/g) || []).length;
  score -= complexFuncs * 5;
  // 过长公式惩罚
  if (formula.length > 100) score -= 15;
  if (formula.length > 200) score -= 15;
  // 奖励简洁形式
  if (complexity < 8) score += 10;
  if (complexity < 5) score += 5;
  return Math.max(0, Math.min(100, score));
}

// 计算综合评分 (拟合误差 + 可解释性)
function computeOverallScore(rmse: number, r2: number, interpretability: number, metrics: { errorWeight: number; interpWeight: number }): number {
  // 误差分 (0-100): 基于 R² 和归一化 RMSE
  const errorScore = Math.max(0, Math.min(100, r2 * 100));
  const interpScore = interpretability;
  return metrics.errorWeight * errorScore + metrics.interpWeight * interpScore;
}

// 残差统计
function residualStats(residuals: number[]) {
  const n = residuals.length;
  const mean = residuals.reduce((s, x) => s + x, 0) / n;
  const variance = residuals.reduce((s, x) => s + (x - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const max = Math.max(...residuals);
  const min = Math.min(...residuals);
  // 偏度
  const skewness = std > 0
    ? residuals.reduce((s, x) => s + ((x - mean) / std) ** 3, 0) / n
    : 0;
  // 峰度
  const kurtosis = std > 0
    ? residuals.reduce((s, x) => s + ((x - mean) / std) ** 4, 0) / n - 3
    : 0;
  return { mean, std, max, min, skewness, kurtosis };
}

// 误差分布直方图
function errorDistribution(residuals: number[], bins = 8) {
  const max = Math.max(...residuals.map(Math.abs));
  if (max === 0) return [{ bin: '0', count: residuals.length, ratio: 1 }];
  const binSize = (max * 2) / bins;
  const dist = Array.from({ length: bins }, (_, i) => {
    const lo = -max + i * binSize;
    const hi = lo + binSize;
    const count = residuals.filter((r) => r >= lo && r < hi).length;
    return {
      bin: `[${lo.toFixed(2)}, ${hi.toFixed(2)})`,
      count,
      ratio: count / residuals.length,
    };
  });
  return dist;
}

// 识别高误差样本
function identifyHighErrorSamples(
  samples: DataSample[],
  predictions: number[],
  targets: number[],
  featureVars: string[],
  threshold: number
) {
  const residuals = predictions.map((p, i) => targets[i] - p);
  const absResiduals = residuals.map(Math.abs);
  const highErrorIdx = absResiduals
    .map((r, i) => ({ r, i }))
    .filter((x) => x.r > threshold)
    .sort((a, b) => b.r - a.r)
    .slice(0, 10)
    .map((x) => x.i);

  return highErrorIdx.map((i) => {
    const features: Record<string, number> = {};
    for (const v of featureVars) {
      features[v] = samples[i][v];
    }
    return {
      index: i,
      features,
      actual: targets[i],
      predicted: predictions[i],
      residual: residuals[i],
    };
  });
}

// 分析高误差样本的特征模式
function analyzeErrorPatterns(
  samples: DataSample[],
  predictions: number[],
  targets: number[],
  featureVars: string[]
): string[] {
  const patterns: string[] = [];
  const residuals = predictions.map((p, i) => Math.abs(targets[i] - p));
  const medianResidual = [...residuals].sort((a, b) => a - b)[Math.floor(residuals.length / 2)];
  const highIdx = residuals.map((r, i) => ({ r, i })).filter((x) => x.r > medianResidual * 1.5).map((x) => x.i);
  const lowIdx = residuals.map((r, i) => ({ r, i })).filter((x) => x.r < medianResidual * 0.5).map((x) => x.i);

  if (highIdx.length < 3 || lowIdx.length < 3) return patterns;

  for (const v of featureVars) {
    const highVals = highIdx.map((i) => samples[i][v]);
    const lowVals = lowIdx.map((i) => samples[i][v]);
    const highMean = highVals.reduce((s, x) => s + x, 0) / highVals.length;
    const lowMean = lowVals.reduce((s, x) => s + x, 0) / lowVals.length;
    const allVals = samples.map((s) => s[v]);
    const overallMean = allVals.reduce((s, x) => s + x, 0) / allVals.length;
    const overallStd = Math.sqrt(allVals.reduce((s, x) => s + (x - overallMean) ** 2, 0) / allVals.length);

    if (overallStd > 0) {
      const diff = (highMean - lowMean) / overallStd;
      if (Math.abs(diff) > 0.5) {
        const direction = diff > 0 ? '较大' : '较小';
        patterns.push(
          `当 ${v} 取值${direction}时 (高误差组均值 ${highMean.toFixed(2)} vs 低误差组 ${lowMean.toFixed(2)}, 差异 ${diff.toFixed(2)}σ), 模型误差显著增大。`
        );
      }
    }
  }

  // 检查误差是否与目标值大小相关
  const highTargetIdx = targets.map((t, i) => ({ t, i })).sort((a, b) => a.t - b.t).slice(-Math.floor(targets.length / 3)).map((x) => x.i);
  const lowTargetIdx = targets.map((t, i) => ({ t, i })).sort((a, b) => a.t - b.t).slice(0, Math.floor(targets.length / 3)).map((x) => x.i);
  const highTargetResidual = highTargetIdx.map((i) => residuals[i]).reduce((s, x) => s + x, 0) / highTargetIdx.length;
  const lowTargetResidual = lowTargetIdx.map((i) => residuals[i]).reduce((s, x) => s + x, 0) / lowTargetIdx.length;
  if (Math.abs(highTargetResidual - lowTargetResidual) > medianResidual * 0.3) {
    const direction = highTargetResidual > lowTargetResidual ? '大目标值' : '小目标值';
    patterns.push(
      `模型在${direction}区间误差更大 (高目标值组平均误差 ${highTargetResidual.toFixed(3)} vs 低目标值组 ${lowTargetResidual.toFixed(3)})。`
    );
  }

  return patterns;
}

export async function evaluateFormula(
  problem: ProblemData,
  formula: string,
  srResult: SRResult,
  metrics: { errorWeight: number; interpWeight: number }
): Promise<EvaluationReport> {
  const { samples, featureVars, targetVar } = problem;
  const targets = samples.map((s) => s[targetVar]);

  const predictions = evaluateFormulaBatch(formula, samples);
  const rmse = computeRMSE(predictions, targets);
  const mae = computeMAE(predictions, targets);
  const r2 = computeR2(predictions, targets);
  const complexity = formulaComplexity(formula);
  const interpretability = computeInterpretability(formula, complexity);
  const overallScore = computeOverallScore(rmse, r2, interpretability, metrics);

  const residuals = predictions.map((p, i) => targets[i] - p);
  const validResiduals = residuals.filter((r) => !isNaN(r) && isFinite(r));
  const rStats = residualStats(validResiduals);
  const dist = errorDistribution(validResiduals);

  const threshold = rStats.std * 1.5;
  const highErrorSamples = identifyHighErrorSamples(samples, predictions, targets, featureVars, threshold);
  const errorPatterns = analyzeErrorPatterns(samples, predictions, targets, featureVars);

  // 生成洞察
  const insights: string[] = [];
  insights.push(`公式复杂度为 ${complexity} 个节点, 可解释性评分 ${interpretability.toFixed(1)}/100。`);
  if (r2 > 0.9) {
    insights.push(`R² = ${r2.toFixed(4)}, 模型能解释 ${(r2 * 100).toFixed(1)}% 的方差, 拟合质量优秀。`);
  } else if (r2 > 0.7) {
    insights.push(`R² = ${r2.toFixed(4)}, 拟合质量良好, 但仍有改进空间。`);
  } else if (r2 > 0.4) {
    insights.push(`R² = ${r2.toFixed(4)}, 拟合质量一般, 公式可能遗漏了重要变量或非线性关系。`);
  } else {
    insights.push(`R² = ${r2.toFixed(4)}, 拟合质量较差, 公式形式可能不正确。`);
  }

  if (Math.abs(rStats.skewness) > 0.5) {
    const dir = rStats.skewness > 0 ? '正偏(低估居多)' : '负偏(高估居多)';
    insights.push(`残差分布${dir} (偏度=${rStats.skewness.toFixed(2)}), 提示存在系统性偏差。`);
  }
  if (rStats.kurtosis > 1) {
    insights.push(`残差峰度较高 (${rStats.kurtosis.toFixed(2)}), 存在异常值或局部拟合失败。`);
  }

  for (const p of errorPatterns) {
    insights.push(p);
  }

  // 生成改进建议
  const recommendations: string[] = [];
  if (r2 < 0.8) {
    recommendations.push('尝试引入更多非线性项 (如平方、对数、指数变换)。');
  }
  if (errorPatterns.length > 0) {
    recommendations.push('针对高误差样本的特征区间, 考虑分段拟合或添加交互项。');
  }
  if (complexity > 15) {
    recommendations.push('公式过于复杂, 考虑简化结构以提升可解释性。');
  }
  if (Math.abs(rStats.skewness) > 0.5) {
    recommendations.push('存在系统性偏差, 尝试对目标变量做变换 (如 log) 或添加偏置项。');
  }
  if (r2 > 0.95 && complexity < 10) {
    recommendations.push('当前公式已很好, 可作为最终解。');
  }

  return {
    formula,
    rmse,
    mae,
    r2,
    complexity,
    interpretability,
    overallScore,
    residualStats: rStats,
    highErrorSamples,
    errorPatterns,
    errorDistribution: dist,
    insights,
    recommendations,
  };
}

// 生成评估报告 wiki 内容
export async function buildEvaluationWiki(
  problem: ProblemData,
  srResults: SRResult[],
  bestReport: EvaluationReport,
  round: number
): Promise<string> {
  let md = `### 第 ${round} 轮评估报告\n\n`;

  md += `#### 符号回归结果汇总\n\n`;
  md += `| 方法 | 公式 | RMSE | R² | 复杂度 | 耗时(ms) |\n`;
  md += `|------|------|------|-----|--------|----------|\n`;
  for (const r of srResults.sort((a, b) => a.rmse - b.rmse)) {
    md += `| ${r.method} | \`${r.formula.slice(0, 50)}${r.formula.length > 50 ? '...' : ''}\` | ${r.rmse.toFixed(4)} | ${r.r2.toFixed(4)} | ${r.complexity} | ${r.durationMs} |\n`;
  }

  md += `\n#### 最佳公式分析\n\n`;
  md += `**公式**: \`${bestReport.formula}\`\n\n`;
  md += `- RMSE: ${bestReport.rmse.toFixed(4)}\n`;
  md += `- MAE: ${bestReport.mae.toFixed(4)}\n`;
  md += `- R²: ${bestReport.r2.toFixed(4)}\n`;
  md += `- 复杂度: ${bestReport.complexity}\n`;
  md += `- 可解释性: ${bestReport.interpretability.toFixed(1)}/100\n`;
  md += `- 综合评分: ${bestReport.overallScore.toFixed(2)}\n\n`;

  md += `#### 残差统计\n\n`;
  md += `- 均值: ${bestReport.residualStats.mean.toFixed(4)}\n`;
  md += `- 标准差: ${bestReport.residualStats.std.toFixed(4)}\n`;
  md += `- 范围: [${bestReport.residualStats.min.toFixed(4)}, ${bestReport.residualStats.max.toFixed(4)}]\n`;
  md += `- 偏度: ${bestReport.residualStats.skewness.toFixed(4)}\n`;
  md += `- 峰度: ${bestReport.residualStats.kurtosis.toFixed(4)}\n\n`;

  md += `#### 误差分布\n\n`;
  md += `| 区间 | 数量 | 占比 |\n|------|------|------|\n`;
  for (const d of bestReport.errorDistribution) {
    md += `| ${d.bin} | ${d.count} | ${(d.ratio * 100).toFixed(1)}% |\n`;
  }

  md += `\n#### 高误差样本 (Top ${bestReport.highErrorSamples.length})\n\n`;
  if (bestReport.highErrorSamples.length > 0) {
    const headers = ['索引', ...problem.featureVars, '实际值', '预测值', '残差'];
    md += `| ${headers.join(' | ')} |\n`;
    md += `| ${headers.map(() => '---').join(' | ')} |\n`;
    for (const s of bestReport.highErrorSamples) {
      const row = [
        String(s.index),
        ...problem.featureVars.map((v) => s.features[v].toFixed(2)),
        s.actual.toFixed(2),
        s.predicted.toFixed(2),
        s.residual.toFixed(2),
      ];
      md += `| ${row.join(' | ')} |\n`;
    }
  }

  md += `\n#### 误差模式发现\n\n`;
  for (const p of bestReport.errorPatterns) {
    md += `- ${p}\n`;
  }

  md += `\n#### 关键洞察\n\n`;
  for (const ins of bestReport.insights) {
    md += `- ${ins}\n`;
  }

  md += `\n#### 改进建议\n\n`;
  for (const rec of bestReport.recommendations) {
    md += `- ${rec}\n`;
  }

  return md;
}

// 生成给下一轮假设的反馈
export function buildEvaluatorFeedback(report: EvaluationReport): string {
  const lines: string[] = [];
  lines.push(`上一轮最佳公式: ${report.formula}`);
  lines.push(`RMSE: ${report.rmse.toFixed(4)}, R²: ${report.r2.toFixed(4)}, 可解释性: ${report.interpretability.toFixed(1)}/100`);
  lines.push(`\n误差模式:`);
  for (const p of report.errorPatterns) {
    lines.push(`- ${p}`);
  }
  lines.push(`\n改进建议:`);
  for (const rec of report.recommendations) {
    lines.push(`- ${rec}`);
  }
  return lines.join('\n');
}
