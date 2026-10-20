/**
 * 符号回归方法 4: 模板匹配 + 参数优化
 * 尝试常见的物理/数学模板，用参数优化拟合
 */
import type { DataSample, SRResult } from '../types';
import { evaluateFormulaBatch, formulaComplexity, computeRMSE, computeMAE, computeR2, optimizeFormulaParams } from './utils';

// 生成模板列表
function generateTemplates(featureVars: string[]): string[] {
  const templates: string[] = [];
  const v = featureVars;

  if (v.length >= 1) {
    const x = v[0];
    // 线性
    templates.push(`a * ${x} + b`);
    // 二次
    templates.push(`a * ${x}^2 + b * ${x} + c`);
    // 幂律
    templates.push(`a * ${x}^b`);
    // 指数
    templates.push(`a * exp(b * ${x})`);
    // 对数
    templates.push(`a * log(abs(${x})) + b`);
    // 平方根
    templates.push(`a * sqrt(abs(${x})) + b`);
    // 双曲
    templates.push(`a * ${x} / (b + ${x})`);
    // Sigmoid
    templates.push(`a / (1 + exp(-b * ${x})) + c`);
    // 倒数
    templates.push(`a / ${x} + b`);
    // Michaelis-Menten
    templates.push(`a * ${x} / (b + abs(${x}))`);
  }

  if (v.length >= 2) {
    const [x, y] = v;
    // 多元线性
    templates.push(`a * ${x} + b * ${y} + c`);
    // 交互
    templates.push(`a * ${x} * ${y} + b`);
    // 比例
    templates.push(`a * ${x} / ${y} + b`);
    // 二次型
    templates.push(`a * ${x}^2 + b * ${y}^2 + c * ${x} * ${y} + d`);
    // 幂律乘积
    templates.push(`a * ${x}^b * ${y}^c`);
    // 指数和
    templates.push(`a * exp(b * ${x}) + c * exp(d * ${y})`);
  }

  if (v.length >= 3) {
    const [x, y, z] = v;
    templates.push(`a * ${x} + b * ${y} + c * ${z} + d`);
    templates.push(`a * ${x} * ${y} + b * ${z} + c`);
  }

  if (v.length >= 4) {
    const [x, y, z, w] = v;
    templates.push(`a * ${x} + b * ${y} + c * ${z} + d * ${w} + e`);
  }

  return templates;
}

export async function runTemplateMatching(
  samples: DataSample[],
  featureVars: string[],
  targetVar: string,
  hypothesisFormula: string,
  options: { maxSteps?: number } = {}
): Promise<SRResult> {
  const startTime = Date.now();
  const targets = samples.map((s) => s[targetVar] ?? 0);

  // 生成模板
  const templates = generateTemplates(featureVars);

  // 也把假设公式作为模板之一
  templates.unshift(hypothesisFormula);

  // 限制尝试的模板数量
  const maxTemplates = Math.min(templates.length, options.maxSteps ?? 12);

  let bestFormula = hypothesisFormula;
  let bestRmse = Infinity;
  const results: { template: string; formula: string; rmse: number }[] = [];

  for (let i = 0; i < maxTemplates; i++) {
    const template = templates[i];
    try {
      const opt = optimizeFormulaParams(template, samples, featureVars, { maxIter: 120, targetVar });
      results.push({ template, formula: opt.optimizedFormula, rmse: opt.rmse });
      if (opt.rmse < bestRmse) {
        bestFormula = opt.optimizedFormula;
        bestRmse = opt.rmse;
      }
    } catch {
      // ignore
    }
  }

  const predictions = evaluateFormulaBatch(bestFormula, samples);

  // 模板描述
  const topResults = results.sort((a, b) => a.rmse - b.rmse).slice(0, 3);
  const notes = `尝试${maxTemplates}个模板, 最佳模板: "${topResults[0]?.template.slice(0, 40) || hypothesisFormula.slice(0, 40)}..."`;

  return {
    method: '模板匹配 (Template Matching)',
    formula: bestFormula,
    rmse: bestRmse,
    mae: computeMAE(predictions, targets),
    r2: computeR2(predictions, targets),
    complexity: formulaComplexity(bestFormula),
    iterations: maxTemplates,
    durationMs: Date.now() - startTime,
    success: bestRmse < 1e9,
    notes,
  };
}
