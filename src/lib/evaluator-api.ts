/**
 * 提交评估工具 - 评估用户提交的公式
 */
import type { ProblemData } from './monk/types';
import { evaluateFormulaBatch, formulaComplexity, computeRMSE, computeMAE, computeR2, optimizeFormulaParams, validateFormula } from './monk/sr/utils';

export interface EvaluationMetrics {
  errorWeight: number;
  interpWeight: number;
}

export interface SubmissionEvaluation {
  formula: string;
  optimizedFormula: string;
  rmse: number;
  mae: number;
  r2: number;
  complexity: number;
  interpretability: number;
  overallScore: number;
  valid: boolean;
  error?: string;
}

function computeInterpretability(formula: string, complexity: number): number {
  let score = 100;
  score -= Math.min(complexity * 2, 40);
  const nestedParens = (formula.match(/\(/g) || []).length;
  score -= Math.min(nestedParens * 1.5, 20);
  const complexFuncs = (formula.match(/(sinh|cosh|tanh|atan|asin|acos)/g) || []).length;
  score -= complexFuncs * 5;
  if (formula.length > 100) score -= 15;
  if (formula.length > 200) score -= 15;
  if (complexity < 8) score += 10;
  if (complexity < 5) score += 5;
  return Math.max(0, Math.min(100, score));
}

export function evaluateSubmission(
  problem: ProblemData,
  formula: string,
  metrics: EvaluationMetrics
): SubmissionEvaluation {
  const validation = validateFormula(formula);
  if (!validation.valid) {
    return {
      formula,
      optimizedFormula: formula,
      rmse: Infinity,
      mae: Infinity,
      r2: -Infinity,
      complexity: 0,
      interpretability: 0,
      overallScore: 0,
      valid: false,
      error: validation.error,
    };
  }

  const { samples, featureVars, targetVar } = problem;
  const targets = samples.map((s) => s[targetVar]);

  // 尝试参数优化
  let optimizedFormula = formula;
  try {
    const opt = optimizeFormulaParams(formula, samples, featureVars, { maxIter: 150, targetVar });
    if (opt.rmse < Infinity) {
      optimizedFormula = opt.optimizedFormula;
    }
  } catch {
    // ignore
  }

  const predictions = evaluateFormulaBatch(optimizedFormula, samples);
  const rmse = computeRMSE(predictions, targets);
  const mae = computeMAE(predictions, targets);
  const r2 = computeR2(predictions, targets);
  const complexity = formulaComplexity(optimizedFormula);
  const interpretability = computeInterpretability(optimizedFormula, complexity);

  const errorScore = Math.max(0, Math.min(100, r2 * 100));
  const overallScore = metrics.errorWeight * errorScore + metrics.interpWeight * interpretability;

  return {
    formula,
    optimizedFormula,
    rmse,
    mae,
    r2,
    complexity,
    interpretability,
    overallScore,
    valid: true,
  };
}

// 重新计算排行榜
export async function updateLeaderboard(problemId: string) {
  const { db } = await import('@/lib/db');
  const submissions = await db.submission.findMany({
    where: { problemId, status: 'evaluated' },
    orderBy: { overallScore: 'desc' },
  });

  for (let i = 0; i < submissions.length; i++) {
    await db.submission.update({
      where: { id: submissions[i].id },
      data: { rank: i + 1 },
    });
  }
}
