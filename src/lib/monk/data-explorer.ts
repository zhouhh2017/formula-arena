/**
 * 数据探索模块 - Coding Agent
 * 对数据进行统计分析、相关性分析、分布检查
 */
import type { DataSample, ProblemData } from './types';

export interface DataExplorationResult {
  summary: string;
  stats: {
    variable: string;
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
  }[];
  correlations: {
    var1: string;
    var2: string;
    correlation: number;
  }[];
  insights: string[];
  suggestedRelations: string[];
}

function mean(arr: number[]): number {
  return arr.reduce((s, x) => s + x, 0) / (arr.length || 1);
}

function std(arr: number[]): number {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length || 1));
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function correlation(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n < 2) return 0;
  const mx = mean(x.slice(0, n));
  const my = mean(y.slice(0, n));
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my);
    dx += (x[i] - mx) ** 2;
    dy += (y[i] - my) ** 2;
  }
  const den = Math.sqrt(dx * dy);
  return den === 0 ? 0 : num / den;
}

export function exploreData(problem: ProblemData): DataExplorationResult {
  const { samples, featureVars, targetVar } = problem;
  const allVars = [...featureVars, targetVar];

  // 统计
  const stats = allVars.map((v) => {
    const vals = samples.map((s) => s[v]).filter((x) => !isNaN(x));
    return {
      variable: v,
      mean: mean(vals),
      std: std(vals),
      min: Math.min(...vals),
      max: Math.max(...vals),
      median: median(vals),
    };
  });

  // 相关性
  const correlations: DataExplorationResult['correlations'] = [];
  for (let i = 0; i < allVars.length; i++) {
    for (let j = i + 1; j < allVars.length; j++) {
      const v1 = allVars[i];
      const v2 = allVars[j];
      const x = samples.map((s) => s[v1]);
      const y = samples.map((s) => s[v2]);
      const corr = correlation(x, y);
      correlations.push({ var1: v1, var2: v2, correlation: corr });
    }
  }
  correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

  // 洞察
  const insights: string[] = [];
  const targetStats = stats.find((s) => s.variable === targetVar)!;
  insights.push(
    `目标变量 ${targetVar} 的均值为 ${targetStats.mean.toFixed(3)}，标准差为 ${targetStats.std.toFixed(3)}，范围 [${targetStats.min.toFixed(3)}, ${targetStats.max.toFixed(3)}]。`
  );

  // 找出与目标变量相关性最高的特征
  const targetCorrs = correlations
    .filter((c) => c.var1 === targetVar || c.var2 === targetVar)
    .map((c) => ({
      feature: c.var1 === targetVar ? c.var2 : c.var1,
      corr: c.correlation,
    }))
    .sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));

  for (const tc of targetCorrs.slice(0, 3)) {
    const strength = Math.abs(tc.corr) > 0.7 ? '强' : Math.abs(tc.corr) > 0.4 ? '中等' : '弱';
    const direction = tc.corr > 0 ? '正' : '负';
    insights.push(
      `特征 ${tc.feature} 与目标 ${targetVar} 呈${strength}${direction}相关 (r=${tc.corr.toFixed(3)})。`
    );
  }

  // 特征间相关性
  const featureCorrs = correlations.filter(
    (c) => c.var1 !== targetVar && c.var2 !== targetVar
  );
  for (const fc of featureCorrs.slice(0, 2)) {
    if (Math.abs(fc.correlation) > 0.7) {
      insights.push(
        `特征 ${fc.var1} 与 ${fc.var2} 高度相关 (r=${fc.correlation.toFixed(3)})，可能存在共线性。`
      );
    }
  }

  // 建议的关系形式
  const suggestedRelations: string[] = [];
  const topCorr = targetCorrs[0];
  if (topCorr) {
    suggestedRelations.push(
      `${targetVar} ≈ a * ${topCorr.feature} + b (线性关系)`
    );
    if (Math.abs(topCorr.corr) < 0.6) {
      suggestedRelations.push(
        `${targetVar} 可能是非线性函数，建议尝试 ${topCorr.feature}^2, sqrt(${topCorr.feature}), log(${topCorr.feature}) 等`
      );
    }
  }
  if (targetCorrs.length >= 2) {
    const f1 = targetCorrs[0].feature;
    const f2 = targetCorrs[1].feature;
    suggestedRelations.push(
      `${targetVar} ≈ a * ${f1} + b * ${f2} + c (多元线性)`
    );
    suggestedRelations.push(
      `${targetVar} ≈ a * ${f1} * ${f2} + b (交互项)`
    );
  }

  // 汇总
  const summary = `数据集包含 ${samples.length} 个样本，${featureVars.length} 个特征变量 (${featureVars.join(', ')}) 和目标变量 ${targetVar}。\n` +
    stats.map((s) => `${s.variable}: μ=${s.mean.toFixed(3)}, σ=${s.std.toFixed(3)}, [${s.min.toFixed(2)}, ${s.max.toFixed(2)}]`).join('\n') +
    `\n\n与目标变量相关性排序:\n` +
    targetCorrs.map((tc) => `  ${tc.feature}: r=${tc.corr.toFixed(3)}`).join('\n');

  return { summary, stats, correlations, insights, suggestedRelations };
}

// 生成数据探索的 wiki 内容
export function buildExplorationWiki(
  problem: ProblemData,
  result: DataExplorationResult
): string {
  let md = `### 数据探索报告\n\n`;
  md += `**数据规模**: ${problem.samples.length} 个样本\n\n`;
  md += `**特征变量**: ${problem.featureVars.join(', ')}\n\n`;
  md += `**目标变量**: ${problem.targetVar}\n\n`;

  md += `#### 变量统计\n\n`;
  md += `| 变量 | 均值 | 标准差 | 最小值 | 中位数 | 最大值 |\n`;
  md += `|------|------|--------|--------|--------|--------|\n`;
  for (const s of result.stats) {
    md += `| ${s.variable} | ${s.mean.toFixed(3)} | ${s.std.toFixed(3)} | ${s.min.toFixed(3)} | ${s.median.toFixed(3)} | ${s.max.toFixed(3)} |\n`;
  }

  md += `\n#### 相关性分析 (Top 10)\n\n`;
  md += `| 变量1 | 变量2 | 相关系数 |\n|-------|-------|----------|\n`;
  for (const c of result.correlations.slice(0, 10)) {
    md += `| ${c.var1} | ${c.var2} | ${c.correlation.toFixed(3)} |\n`;
  }

  md += `\n#### 关键洞察\n\n`;
  for (const ins of result.insights) {
    md += `- ${ins}\n`;
  }

  md += `\n#### 建议的关系形式\n\n`;
  for (const rel of result.suggestedRelations) {
    md += `- ${rel}\n`;
  }

  return md;
}
