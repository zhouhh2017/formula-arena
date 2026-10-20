/**
 * 符号回归方法 2: SINDy (Sparse Identification of Nonlinear Dynamics)
 * 构建候选项库，用稀疏回归选择重要项
 */
import type { DataSample, SRResult } from '../types';
import { evaluateFormulaBatch, formulaComplexity, computeRMSE, computeMAE, computeR2, optimizeFormulaParams } from './utils';

// 构建候选特征库
function buildLibrary(
  samples: DataSample[],
  featureVars: string[],
  maxDegree: number = 2
): { terms: string[]; matrix: number[][] } {
  const n = samples.length;
  const terms: string[] = ['1']; // 偏置项
  const matrix: number[][] = [];

  // 初始化矩阵
  for (let i = 0; i < n; i++) matrix.push([1]);

  // 一次项
  for (const v of featureVars) {
    terms.push(v);
    for (let i = 0; i < n; i++) {
      matrix[i].push(samples[i][v] ?? 0);
    }
  }

  // 二次项和交互项
  if (maxDegree >= 2) {
    for (let i = 0; i < featureVars.length; i++) {
      for (let j = i; j < featureVars.length; j++) {
        const v1 = featureVars[i];
        const v2 = featureVars[j];
        const termName = i === j ? `${v1}^2` : `${v1} * ${v2}`;
        terms.push(termName);
        for (let k = 0; k < n; k++) {
          matrix[k].push((samples[k][v1] ?? 0) * (samples[k][v2] ?? 0));
        }
      }
    }
  }

  // 非线性变换项
  for (const v of featureVars) {
    terms.push(`sqrt(abs(${v}))`);
    for (let i = 0; i < n; i++) {
      matrix[i].push(Math.sqrt(Math.abs(samples[i][v] ?? 0)));
    }
    terms.push(`log(abs(${v}))`);
    for (let i = 0; i < n; i++) {
      const val = Math.abs(samples[i][v] ?? 0);
      matrix[i].push(val > 1e-10 ? Math.log(val) : 0);
    }
    terms.push(`exp(${v})`);
    for (let i = 0; i < n; i++) {
      const val = samples[i][v] ?? 0;
      matrix[i].push(Math.exp(Math.min(val, 10))); // 防止溢出
    }
  }

  return { terms, matrix };
}

// STLSQ (Sequentially Thresholded Least Squares) 稀疏回归
function stlsq(
  A: number[][],
  b: number[],
  threshold: number = 0.05,
  maxIter: number = 10
): number[] {
  const n = A.length;
  const m = A[0].length;
  let coeffs = leastSquares(A, b);
  let activeIdx = Array.from({ length: m }, (_, i) => i);

  for (let iter = 0; iter < maxIter; iter++) {
    // 阈值过滤
    const newActive = activeIdx.filter((i) => Math.abs(coeffs[i]) > threshold);
    if (newActive.length === 0 || newActive.length === activeIdx.length) break;
    activeIdx = newActive;

    // 在活跃子集上重新求解
    const subA = A.map((row) => activeIdx.map((i) => row[i]));
    const subCoeffs = leastSquares(subA, b);
    coeffs = new Array(m).fill(0);
    activeIdx.forEach((idx, i) => {
      coeffs[idx] = subCoeffs[i];
    });
  }

  return coeffs;
}

// 最小二乘法 (正规方程 + 正则化)
function leastSquares(A: number[][], b: number[]): number[] {
  const n = A.length;
  const m = A[0].length;
  const lambda = 1e-6; // L2 正则化

  // 计算 A^T * A + lambda * I
  const AtA: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  const Atb: number[] = new Array(m).fill(0);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      for (let k = 0; k < m; k++) {
        AtA[j][k] += A[i][j] * A[i][k];
      }
      Atb[j] += A[i][j] * b[i];
    }
  }
  for (let i = 0; i < m; i++) AtA[i][i] += lambda;

  return solveLinearSystem(AtA, Atb);
}

// 高斯消元法解线性方程组
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    // 选主元
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) maxRow = k;
    }
    [aug[i], aug[maxRow]] = [aug[maxRow], aug[i]];

    if (Math.abs(aug[i][i]) < 1e-12) continue;

    for (let k = i + 1; k < n; k++) {
      const factor = aug[k][i] / aug[i][i];
      for (let j = i; j <= n; j++) {
        aug[k][j] -= factor * aug[i][j];
      }
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= aug[i][j] * x[j];
    }
    x[i] = Math.abs(aug[i][i]) > 1e-12 ? sum / aug[i][i] : 0;
  }
  return x;
}

export async function runSINDy(
  samples: DataSample[],
  featureVars: string[],
  targetVar: string,
  hypothesisFormula: string,
  options: { maxSteps?: number } = {}
): Promise<SRResult> {
  const startTime = Date.now();
  const targets = samples.map((s) => s[targetVar] ?? 0);

  // 标准化目标值 (帮助稀疏回归)
  const targetMean = targets.reduce((s, x) => s + x, 0) / targets.length;
  const targetStd = Math.sqrt(targets.reduce((s, x) => s + (x - targetMean) ** 2, 0) / targets.length) || 1;
  const normTargets = targets.map((t) => (t - targetMean) / targetStd);

  // 构建库
  const { terms, matrix } = buildLibrary(samples, featureVars, 2);

  // 稀疏回归
  const coeffs = stlsq(matrix, normTargets, 0.05, options.maxSteps ?? 10);

  // 反标准化系数
  const realCoeffs = coeffs.map((c) => c * targetStd);

  // 构建公式
  const activeTerms: { term: string; coeff: number }[] = [];
  for (let i = 0; i < terms.length; i++) {
    if (Math.abs(realCoeffs[i]) > 1e-4) {
      activeTerms.push({ term: terms[i], coeff: realCoeffs[i] });
    }
  }

  // 偏置项
  const bias = -targetMean / targetStd * targetStd; // = -targetMean, but let's compute properly
  // 实际上偏置已经在 terms[0] = '1' 中了

  if (activeTerms.length === 0) {
    return {
      method: 'SINDy 稀疏回归',
      formula: '0',
      rmse: 1e10,
      mae: 1e10,
      r2: 0,
      complexity: 1,
      iterations: options.maxSteps ?? 10,
      durationMs: Date.now() - startTime,
      success: false,
      notes: '稀疏回归未找到有效项',
    };
  }

  // 构建公式字符串
  let formula = activeTerms
    .map((t) => {
      const c = t.coeff;
      const sign = c >= 0 ? '' : '-';
      const absC = Math.abs(c).toFixed(4);
      if (t.term === '1') return `${sign}${absC}`;
      return `${sign}${absC} * ${t.term}`;
    })
    .join(' + ')
    .replace(/\+ -/g, '- ');

  // 如果第一项是负的，去掉前导的 "- "
  if (formula.startsWith('- ')) {
    formula = '-' + formula.slice(2);
  }

  // 参数优化微调
  try {
    const opt = optimizeFormulaParams(formula, samples, featureVars, { maxIter: 80, targetVar });
    if (opt.rmse < computeRMSE(evaluateFormulaBatch(formula, samples), targets)) {
      formula = opt.optimizedFormula;
    }
  } catch {
    // ignore
  }

  const predictions = evaluateFormulaBatch(formula, samples);
  const rmse = computeRMSE(predictions, targets);

  return {
    method: 'SINDy 稀疏回归',
    formula,
    rmse,
    mae: computeMAE(predictions, targets),
    r2: computeR2(predictions, targets),
    complexity: formulaComplexity(formula),
    iterations: options.maxSteps ?? 10,
    durationMs: Date.now() - startTime,
    success: rmse < 1e9,
    notes: `候选库${terms.length}项, 稀疏阈值0.05, 选中${activeTerms.length}项`,
  };
}
