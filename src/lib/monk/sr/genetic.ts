/**
 * 符号回归方法 1: 遗传规划 (Genetic Programming)
 * 基于表达式树的进化算法
 */
import type { DataSample, SRResult } from '../types';
import type { ExprNode } from '../math-engine';
import { evalNode, formulaComplexity, computeRMSE, computeMAE, computeR2, optimizeFormulaParams } from './utils';

interface Individual {
  tree: ExprNode;
  fitness: number;
}

const TERMINALS = ['x0', 'x1', 'x2', 'x3', 'x4', '1', '2', '0.5'];
const FUNCTIONS = ['+', '-', '*', '/', '^'];
const UNARY_FUNCS = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs'];

function randomTree(depth: number, varNames: string[], allowParams = false): ExprNode {
  const terminals = [...varNames, '1', '2', '0.5', '0.1'];
  if (allowParams) terminals.push('a', 'b', 'c');

  if (depth <= 0 || Math.random() < 0.3) {
    const t = terminals[Math.floor(Math.random() * terminals.length)];
    if (/^-?\d+\.?\d*$/.test(t)) {
      return { type: 'num', value: parseFloat(t) };
    }
    return { type: 'var', name: t };
  }

  // 20% 概率使用一元函数
  if (Math.random() < 0.2) {
    const fn = UNARY_FUNCS[Math.floor(Math.random() * UNARY_FUNCS.length)];
    return {
      type: 'call',
      name: fn,
      args: [randomTree(depth - 1, varNames, allowParams)],
    };
  }

  const op = FUNCTIONS[Math.floor(Math.random() * FUNCTIONS.length)];
  return {
    type: 'binop',
    op,
    left: randomTree(depth - 1, varNames, allowParams),
    right: randomTree(depth - 1, varNames, allowParams),
  };
}

function mutate(node: ExprNode, varNames: string[], prob = 0.2): ExprNode {
  if (Math.random() < prob) {
    return randomTree(2, varNames, true);
  }
  switch (node.type) {
    case 'num':
      if (Math.random() < 0.5) {
        return { type: 'num', value: node.value + (Math.random() - 0.5) * 0.5 };
      }
      return node;
    case 'var':
      return node;
    case 'binop':
      if (Math.random() < 0.3) {
        const ops = FUNCTIONS.filter((o) => o !== node.op);
        return { ...node, op: ops[Math.floor(Math.random() * ops.length)] };
      }
      return {
        ...node,
        left: mutate(node.left, varNames, prob),
        right: mutate(node.right, varNames, prob),
      };
    case 'unaryop':
      return { ...node, operand: mutate(node.operand, varNames, prob) };
    case 'call':
      return { ...node, args: node.args.map((a) => mutate(a, varNames, prob)) };
  }
}

function crossover(a: ExprNode, b: ExprNode): ExprNode {
  // 以 a 为基础，随机替换一个子树为 b 的子树
  if (Math.random() < 0.3) return cloneNode(b);
  switch (a.type) {
    case 'binop': {
      const choice = Math.random();
      if (choice < 0.33) return { ...a, left: crossover(a.left, b) };
      if (choice < 0.66) return { ...a, right: crossover(a.right, b) };
      return { ...a, left: cloneNode(b) };
    }
    case 'unaryop':
      return { ...a, operand: crossover(a.operand, b) };
    case 'call':
      return {
        ...a,
        args: a.args.map((arg) => (Math.random() < 0.5 ? crossover(arg, b) : arg)),
      };
    default:
      return cloneNode(b);
  }
}

function cloneNode(node: ExprNode): ExprNode {
  return JSON.parse(JSON.stringify(node));
}

function nodeToString(node: ExprNode): string {
  switch (node.type) {
    case 'num':
      return node.value.toFixed(4);
    case 'var':
      return node.name;
    case 'binop':
      return `(${nodeToString(node.left)} ${node.op} ${nodeToString(node.right)})`;
    case 'unaryop':
      return `(${node.op}${nodeToString(node.operand)})`;
    case 'call':
      return `${node.name}(${node.args.map(nodeToString).join(', ')})`;
  }
}

export async function runGeneticProgramming(
  samples: DataSample[],
  featureVars: string[],
  targetVar: string,
  hypothesisFormula: string,
  options: { maxSteps?: number; populationSize?: number } = {}
): Promise<SRResult> {
  const startTime = Date.now();
  const { maxSteps = 30, populationSize = 30 } = options;
  const targets = samples.map((s) => s[targetVar] ?? 0);

  // 评估个体
  const evaluate = (tree: ExprNode): number => {
    let sumSq = 0, count = 0;
    for (const s of samples) {
      try {
        const pred = evalNode(tree, s);
        if (isNaN(pred) || !isFinite(pred) || Math.abs(pred) > 1e10) continue;
        const err = pred - targets[count];
        sumSq += err * err;
        count++;
      } catch {
        // skip
      }
    }
    if (count < samples.length * 0.5) return 1e10; // 太多无效
    const rmse = Math.sqrt(sumSq / count);
    // 复杂度惩罚
    const complexity = countNodes(tree);
    return rmse * (1 + complexity * 0.001);
  };

  // 初始化种群: 一部分从假设公式变异，一部分随机
  const population: Individual[] = [];
  // 尝试解析假设公式作为种子
  try {
    const { parseExpression } = await import('../math-engine');
    const seedTree = parseExpression(hypothesisFormula);
    for (let i = 0; i < populationSize / 3; i++) {
      population.push({ tree: mutate(seedTree, featureVars, 0.4), fitness: 0 });
    }
  } catch {
    // ignore
  }
  while (population.length < populationSize) {
    const tree = randomTree(3, featureVars, true);
    population.push({ tree, fitness: 0 });
  }

  // 评估初始种群
  for (const ind of population) {
    ind.fitness = evaluate(ind.tree);
  }

  let best = population.reduce((a, b) => (a.fitness < b.fitness ? a : b));
  let generation = 0;

  for (let gen = 0; gen < maxSteps; gen++) {
    generation++;
    // 选择 + 交叉 + 变异
    population.sort((a, b) => a.fitness - b.fitness);
    const elite = Math.floor(populationSize * 0.2);
    const newPop = population.slice(0, elite);

    while (newPop.length < populationSize) {
      const parentA = tournament(population);
      const parentB = tournament(population);
      const child = crossover(parentA.tree, parentB.tree);
      const mutated = mutate(child, featureVars, 0.3);
      newPop.push({ tree: mutated, fitness: evaluate(mutated) });
    }

    population.length = 0;
    population.push(...newPop);
    const genBest = population.reduce((a, b) => (a.fitness < b.fitness ? a : b));
    if (genBest.fitness < best.fitness) {
      best = genBest;
    }
  }

  // 对最佳个体进行参数优化
  const bestFormulaStr = nodeToString(best.tree);
  let finalFormula = bestFormulaStr;
  let rmse = best.fitness;

  try {
    const opt = optimizeFormulaParams(bestFormulaStr, samples, featureVars, { maxIter: 100, targetVar });
    if (opt.rmse < rmse) {
      finalFormula = opt.optimizedFormula;
      rmse = opt.rmse;
    }
  } catch {
    // ignore
  }

  const predictions = samples.map((s) => {
    try {
      return evalNode(best.tree, s);
    } catch {
      return NaN;
    }
  });

  return {
    method: '遗传规划 (Genetic Programming)',
    formula: finalFormula,
    rmse,
    mae: computeMAE(predictions, targets),
    r2: computeR2(predictions, targets),
    complexity: formulaComplexity(finalFormula),
    iterations: generation,
    durationMs: Date.now() - startTime,
    success: rmse < 1e9,
    notes: `种群${populationSize}, 进化${generation}代, 从假设公式"${hypothesisFormula.slice(0, 30)}..."出发探索`,
  };
}

function tournament(population: Individual[], size = 3): Individual {
  let best = population[Math.floor(Math.random() * population.length)];
  for (let i = 1; i < size; i++) {
    const c = population[Math.floor(Math.random() * population.length)];
    if (c.fitness < best.fitness) best = c;
  }
  return best;
}

function countNodes(node: ExprNode): number {
  switch (node.type) {
    case 'num':
    case 'var':
      return 1;
    case 'binop':
      return 1 + countNodes(node.left) + countNodes(node.right);
    case 'unaryop':
      return 1 + countNodes(node.operand);
    case 'call':
      return 1 + node.args.reduce((s, a) => s + countNodes(a), 0);
  }
}

// Alias for orchestrator compatibility
export const runGenetic = runGeneticProgramming;
