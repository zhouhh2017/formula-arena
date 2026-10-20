/**
 * 符号回归共享工具
 * 参数优化 (最小二乘法 / Nelder-Mead)
 */
import type { DataSample } from '../types';
import { parseExpression, evalNode, type ExprNode, evaluateFormulaBatch as _evaluateFormulaBatch, formulaComplexity as _formulaComplexity, validateFormula as _validateFormula } from '../math-engine';

// Wrapper functions for re-export
export function evaluateFormulaBatch(formula: string, samples: DataSample[]): number[] {
  return _evaluateFormulaBatch(formula, samples);
}
export function formulaComplexity(formula: string): number {
  return _formulaComplexity(formula);
}
export function validateFormula(formula: string): { valid: boolean; error?: string } {
  return _validateFormula(formula);
}
export { evalNode } from '../math-engine';

// 提取公式中的参数符号 (单字母或 a/b/c... 但不是变量)
export function extractParameters(
  node: ExprNode,
  knownVars: Set<string>
): string[] {
  const params = new Set<string>();
  const CONSTANTS = new Set(['pi', 'e']);
  function walk(n: ExprNode) {
    switch (n.type) {
      case 'var':
        if (!knownVars.has(n.name) && !CONSTANTS.has(n.name)) {
          params.add(n.name);
        }
        break;
      case 'binop':
        walk(n.left);
        walk(n.right);
        break;
      case 'unaryop':
        walk(n.operand);
        break;
      case 'call':
        n.args.forEach(walk);
        break;
    }
  }
  walk(node);
  return Array.from(params);
}

// Nelder-Mead 优化
export function nelderMead(
  objective: (params: number[]) => number,
  initParams: number[],
  options: { maxIter?: number; tolerance?: number } = {}
): { params: number[]; value: number; iterations: number } {
  const { maxIter = 200, tolerance = 1e-8 } = options;
  const n = initParams.length;
  if (n === 0) return { params: [], value: objective([]), iterations: 0 };

  // 初始单纯形
  const simplex: { params: number[]; value: number }[] = [];
  for (let i = 0; i <= n; i++) {
    const p = [...initParams];
    if (i < n) {
      p[i] = p[i] !== 0 ? p[i] * 1.5 : 0.1;
    }
    simplex.push({ params: p, value: objective(p) });
  }

  const alpha = 1, gamma = 2, rho = 0.5, sigma = 0.5;
  let iter = 0;

  while (iter < maxIter) {
    simplex.sort((a, b) => a.value - b.value);
    const best = simplex[0];
    const worst = simplex[n];
    if (worst.value - best.value < tolerance) break;

    // 重心 (去掉最差点)
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        centroid[j] += simplex[i].params[j];
      }
    }
    for (let j = 0; j < n; j++) centroid[j] /= n;

    // 反射
    const reflected = centroid.map((c, j) => c + alpha * (c - worst.params[j]));
    const reflectedValue = objective(reflected);

    if (reflectedValue >= simplex[0].value && reflectedValue < simplex[n - 1].value) {
      simplex[n] = { params: reflected, value: reflectedValue };
    } else if (reflectedValue < simplex[0].value) {
      // 扩展
      const expanded = centroid.map((c, j) => c + gamma * (reflected[j] - c));
      const expandedValue = objective(expanded);
      if (expandedValue < reflectedValue) {
        simplex[n] = { params: expanded, value: expandedValue };
      } else {
        simplex[n] = { params: reflected, value: reflectedValue };
      }
    } else {
      // 收缩
      const contracted = centroid.map((c, j) => c + rho * (worst.params[j] - c));
      const contractedValue = objective(contracted);
      if (contractedValue < worst.value) {
        simplex[n] = { params: contracted, value: contractedValue };
      } else {
        // 缩小
        for (let i = 1; i <= n; i++) {
          simplex[i].params = simplex[i].params.map((p, j) =>
            (p + simplex[0].params[j]) / 2
          );
          simplex[i].value = objective(simplex[i].params);
        }
      }
    }
    iter++;
  }

  simplex.sort((a, b) => a.value - b.value);
  return { params: simplex[0].params, value: simplex[0].value, iterations: iter };
}

// 优化公式参数: 用数据拟合公式中的参数
export function optimizeFormulaParams(
  formula: string,
  samples: DataSample[],
  featureVars: string[],
  options: { maxIter?: number; targetVar?: string } = {}
): { optimizedFormula: string; rmse: number; params: Record<string, number> } {
  const node = parseExpression(formula);
  const knownVars = new Set(featureVars);
  const paramNames = extractParameters(node, knownVars);
  const targetVar = options.targetVar || '__target';

  if (paramNames.length === 0) {
    // 没有参数，直接评估
    const predictions = samples.map((s) => {
      try {
        return evalNode(node, s);
      } catch {
        return NaN;
      }
    });
    const rmse = computeRMSE(predictions, samples.map((s) => s[targetVar] ?? 0));
    return { optimizedFormula: formula, rmse, params: {} };
  }

  const targets = samples.map((s) => s[targetVar] ?? 0);

  const objective = (params: number[]): number => {
    const paramMap: Record<string, number> = {};
    paramNames.forEach((name, i) => {
      paramMap[name] = params[i];
    });
    let sumSq = 0;
    let count = 0;
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      const vars = { ...s, ...paramMap };
      try {
        const pred = evalNode(node, vars);
        if (isNaN(pred) || !isFinite(pred)) {
          count++;
          continue;
        }
        const err = pred - targets[i];
        sumSq += err * err;
        count++;
      } catch {
        count++;
      }
    }
    return count > 0 ? Math.sqrt(sumSq / count) : 1e10;
  };

  // 多次随机重启以提高找到好解的概率
  let bestResult = null;
  const restarts = 3;
  for (let r = 0; r < restarts; r++) {
    const initParams = paramNames.map(() => (r === 0 ? 0.5 : -1 + Math.random() * 2));
    const result = nelderMead(objective, initParams, {
      maxIter: Math.floor((options.maxIter ?? 150) / restarts),
    });
    if (!bestResult || result.value < bestResult.value) {
      bestResult = result;
    }
  }

  const paramMap: Record<string, number> = {};
  paramNames.forEach((name, i) => {
    paramMap[name] = bestResult!.params[i];
  });

  // 生成优化后的公式字符串 (替换参数为数值)
  let optimizedFormula = formula;
  // 按名称长度降序替换，避免子串问题
  const sortedParams = [...paramNames].sort((a, b) => b.length - a.length);
  for (const name of sortedParams) {
    const val = paramMap[name];
    const valStr = val >= 0 ? val.toFixed(4) : `(${val.toFixed(4)})`;
    optimizedFormula = optimizedFormula.replace(
      new RegExp(`\\b${escapeRegex(name)}\\b`, 'g'),
      valStr
    );
  }

  return {
    optimizedFormula,
    rmse: bestResult!.value,
    params: paramMap,
  };
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function computeRMSE(predictions: number[], targets: number[]): number {
  let sumSq = 0;
  let count = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (isNaN(predictions[i]) || !isFinite(predictions[i])) continue;
    const err = predictions[i] - targets[i];
    sumSq += err * err;
    count++;
  }
  return count > 0 ? Math.sqrt(sumSq / count) : 1e10;
}

export function computeMAE(predictions: number[], targets: number[]): number {
  let sumAbs = 0;
  let count = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (isNaN(predictions[i]) || !isFinite(predictions[i])) continue;
    sumAbs += Math.abs(predictions[i] - targets[i]);
    count++;
  }
  return count > 0 ? sumAbs / count : 1e10;
}

export function computeR2(predictions: number[], targets: number[]): number {
  const validIdx: number[] = [];
  for (let i = 0; i < predictions.length; i++) {
    if (!isNaN(predictions[i]) && isFinite(predictions[i])) {
      validIdx.push(i);
    }
  }
  if (validIdx.length < 2) return -Infinity;
  const meanT =
    validIdx.reduce((s, i) => s + targets[i], 0) / validIdx.length;
  let ssRes = 0, ssTot = 0;
  for (const i of validIdx) {
    ssRes += (targets[i] - predictions[i]) ** 2;
    ssTot += (targets[i] - meanT) ** 2;
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}
