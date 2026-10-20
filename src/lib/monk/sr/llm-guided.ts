/**
 * 符号回归方法 3: LLM 引导的符号回归
 * LLM 提出候选公式，评估后迭代改进
 */
import { chat, extractFormulaFromText } from '../llm';
import type { DataSample, SRResult } from '../types';
import { evaluateFormulaBatch, formulaComplexity, computeRMSE, computeMAE, computeR2, optimizeFormulaParams, validateFormula } from './utils';

export async function runLLMGuided(
  samples: DataSample[],
  featureVars: string[],
  targetVar: string,
  hypothesisFormula: string,
  wikiContext: string,
  options: { maxSteps?: number } = {}
): Promise<SRResult> {
  const startTime = Date.now();
  const maxSteps = options.maxSteps ?? 4;
  const targets = samples.map((s) => s[targetVar] ?? 0);

  // 提供数据摘要给 LLM
  const dataSummary = summarizeData(samples, featureVars, targets);

  let bestFormula = hypothesisFormula;
  let bestRmse = Infinity;

  // 初始评估
  try {
    const opt = optimizeFormulaParams(hypothesisFormula, samples, featureVars, { maxIter: 100, targetVar });
    bestFormula = opt.optimizedFormula;
    bestRmse = opt.rmse;
  } catch {
    // ignore
  }

  const history: { formula: string; rmse: number }[] = [{ formula: bestFormula, rmse: bestRmse }];

  for (let step = 0; step < maxSteps; step++) {
    const prompt = buildPrompt(featureVars, dataSummary, history, wikiContext, step);

    const response = await chat(
      [
        {
          role: 'system',
          content: '你是一个符号回归专家。根据数据特征和历史尝试，提出一个更好的数学公式来拟合数据。只返回公式表达式，不要解释。公式必须使用给定的变量名，可以包含参数(a,b,c等)。',
        },
        { role: 'user', content: prompt },
      ],
      { temperature: 0.9 }
    );

    const candidate = extractFormulaFromText(response);
    if (!candidate) continue;

    const validation = validateFormula(candidate);
    if (!validation.valid) continue;

    // 参数优化
    try {
      const opt = optimizeFormulaParams(candidate, samples, featureVars, { maxIter: 100, targetVar });
      history.push({ formula: opt.optimizedFormula, rmse: opt.rmse });

      if (opt.rmse < bestRmse) {
        bestFormula = opt.optimizedFormula;
        bestRmse = opt.rmse;
      }
    } catch {
      // ignore
    }
  }

  const predictions = evaluateFormulaBatch(bestFormula, samples);

  return {
    method: 'LLM 引导符号回归',
    formula: bestFormula,
    rmse: bestRmse,
    mae: computeMAE(predictions, targets),
    r2: computeR2(predictions, targets),
    complexity: formulaComplexity(bestFormula),
    iterations: maxSteps,
    durationMs: Date.now() - startTime,
    success: bestRmse < 1e9,
    notes: `LLM迭代${maxSteps}次, 从假设"${hypothesisFormula.slice(0, 30)}..."出发, 尝试了${history.length}个候选`,
  };
}

function summarizeData(
  samples: DataSample[],
  featureVars: string[],
  targets: number[]
): string {
  const lines: string[] = [];
  for (const v of featureVars) {
    const vals = samples.map((s) => s[v]);
    const mean = vals.reduce((s, x) => s + x, 0) / vals.length;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    lines.push(`${v}: 均值=${mean.toFixed(2)}, 范围=[${min.toFixed(2)}, ${max.toFixed(2)}]`);
  }
  const tMean = targets.reduce((s, x) => s + x, 0) / targets.length;
  const tMin = Math.min(...targets);
  const tMax = Math.max(...targets);
  lines.push(`目标: 均值=${tMean.toFixed(2)}, 范围=[${tMin.toFixed(2)}, ${tMax.toFixed(2)}]`);

  // 前5个样本
  lines.push('\n样本示例:');
  for (let i = 0; i < Math.min(5, samples.length); i++) {
    const feats = featureVars.map((v) => `${v}=${samples[i][v].toFixed(2)}`).join(', ');
    lines.push(`  ${feats} -> 目标=${targets[i].toFixed(2)}`);
  }

  return lines.join('\n');
}

function buildPrompt(
  featureVars: string[],
  dataSummary: string,
  history: { formula: string; rmse: number }[],
  wikiContext: string,
  step: number
): string {
  const sortedHistory = [...history].sort((a, b) => a.rmse - b.rmse).slice(0, 3);
  const historyStr = sortedHistory
    .map((h, i) => `${i + 1}. 公式: ${h.formula}\n   RMSE: ${h.rmse.toFixed(4)}`)
    .join('\n');

  return `## 任务: 提出更好的拟合公式

可用变量: ${featureVars.join(', ')}
可用函数: sin, cos, exp, log, sqrt, abs, +, -, *, /, ^
可使用参数: a, b, c, d (会被自动优化)

## 数据特征
${dataSummary}

## 已有调研知识
${wikiContext.slice(0, 1500)}

## 历史尝试 (按误差排序)
${historyStr}

## 当前是第 ${step + 1} 步迭代

请提出一个**新的**公式 (与历史不同)，尝试不同的函数形式或变量组合。只返回公式表达式。`;
}
