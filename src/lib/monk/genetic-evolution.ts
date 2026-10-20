/**
 * 种群进化遗传算法 (跨代)
 *
 * 将公式视为"基因", 通过跨代选择/交叉/变异产生下一代假设
 *
 * 遗传操作:
 *   - selection: 从前代种群中选择最优个体作为父本
 *   - crossover: 组合两个父本公式的子树产生后代
 *   - mutation: 对公式进行随机变异 (替换子树/改变运算符/改变函数)
 *   - elitism: 保留前代最优个体直接进入下一代
 *
 * 与 llm-wiki 结合:
 *   - 变异方向受 wiki 中的 "pitfall_avoidance" 经验引导
 *   - 交叉策略受 wiki 中的 "technique" 经验引导
 *   - 适应度函数受 wiki 中的 "insight" 经验引导
 */
import { parseExpression, type ExprNode, nodeToString } from './math-engine';
import type { ExperienceRecord } from './experience-wiki';

export interface FormulaGene {
  formula: string;
  tree: ExprNode;
  fitness: number; // 综合评分 (越高越好)
  error: number;
  complexity: number;
  source: string; // 来源: 'elitism' / 'crossover' / 'mutation' / 'llm'
  parents?: string[]; // 父本公式
  generation: number;
}

export interface EvolutionPlan {
  generation: number;
  // 遗传算法参数
  populationSize: number;
  eliteCount: number;
  crossoverRate: number;
  mutationRate: number;
  // 初始种群 (从前代继承 + 变异)
  seedGenes: FormulaGene[];
  // 变异方向引导 (来自 wiki)
  mutationGuides: string[];
  // 避坑列表 (来自 wiki)
  avoidPitfalls: string[];
  // 探索重点
  focusAreas: string[];
}

// ===== 选择: 锦标赛选择 =====
export function selection(population: FormulaGene[], tournamentSize = 3): FormulaGene {
  let best = population[Math.floor(Math.random() * population.length)];
  for (let i = 1; i < tournamentSize; i++) {
    const candidate = population[Math.floor(Math.random() * population.length)];
    if (candidate.fitness > best.fitness) best = candidate;
  }
  return best;
}

// ===== 交叉: 子树交换 =====
export function crossover(parent1: FormulaGene, parent2: FormulaGene): FormulaGene {
  try {
    const tree1 = JSON.parse(JSON.stringify(parent1.tree)) as ExprNode;
    const tree2 = JSON.parse(JSON.stringify(parent2.tree)) as ExprNode;

    // 收集两棵树的所有子树节点 (可替换点)
    const nodes1 = collectNodes(tree1);
    const nodes2 = collectNodes(tree2);

    if (nodes1.length === 0 || nodes2.length === 0) {
      return { ...parent1, source: 'crossover', parents: [parent1.formula, parent2.formula] };
    }

    // 随机选择交叉点
    const crossPoint1 = nodes1[Math.floor(Math.random() * nodes1.length)];
    const crossPoint2 = nodes2[Math.floor(Math.random() * nodes2.length)];

    // 交换子树
    const subtree = JSON.parse(JSON.stringify(crossPoint2.node));
    replaceNode(tree1, crossPoint1.path, subtree);

    const newFormula = nodeToString(tree1);
    return {
      formula: newFormula,
      tree: tree1,
      fitness: 0, // 待评估
      error: Infinity,
      complexity: countNodes(tree1),
      source: 'crossover',
      parents: [parent1.formula, parent2.formula],
      generation: parent1.generation,
    };
  } catch {
    return { ...parent1, source: 'crossover', parents: [parent1.formula, parent2.formula] };
  }
}

// ===== 变异: 随机改变子树 =====
export function mutate(
  gene: FormulaGene,
  featureVars: string[],
  guides?: string[]
): FormulaGene {
  try {
    const tree = JSON.parse(JSON.stringify(gene.tree)) as ExprNode;
    const nodes = collectNodes(tree);
    if (nodes.length === 0) return { ...gene, source: 'mutation' };

    // 随机选择变异点
    const mutatePoint = nodes[Math.floor(Math.random() * nodes.length)];

    // 变异类型
    const mutationType = Math.random();
    let newSubtree: ExprNode;

    if (mutationType < 0.3) {
      // 替换为随机终端
      const terminals = [...featureVars, '1', '2', '0.5', 'a', 'b', 'c'];
      const t = terminals[Math.floor(Math.random() * terminals.length)];
      newSubtree = /^-?\d/.test(t)
        ? { type: 'num', value: parseFloat(t) }
        : { type: 'var', name: t };
    } else if (mutationType < 0.6) {
      // 替换为随机函数应用
      const funcs = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs'];
      const fn = funcs[Math.floor(Math.random() * funcs.length)];
      const argVar = featureVars[Math.floor(Math.random() * featureVars.length)];
      newSubtree = {
        type: 'call',
        name: fn,
        args: [{ type: 'var', name: argVar }],
      };
    } else {
      // 替换为随机二元运算
      const ops = ['+', '-', '*', '/'];
      const op = ops[Math.floor(Math.random() * ops.length)];
      const v1 = featureVars[Math.floor(Math.random() * featureVars.length)];
      const v2 = featureVars[Math.floor(Math.random() * featureVars.length)];
      newSubtree = {
        type: 'binop',
        op,
        left: { type: 'var', name: v1 },
        right: { type: 'var', name: v2 },
      };
    }

    replaceNode(tree, mutatePoint.path, newSubtree);
    const newFormula = nodeToString(tree);

    return {
      formula: newFormula,
      tree,
      fitness: 0,
      error: Infinity,
      complexity: countNodes(tree),
      source: 'mutation',
      parents: [gene.formula],
      generation: gene.generation,
    };
  } catch {
    return { ...gene, source: 'mutation' };
  }
}

// ===== 生成下一代种群 =====
export function evolvePopulation(
  currentPopulation: FormulaGene[],
  featureVars: string[],
  plan: {
    populationSize: number;
    eliteCount: number;
    crossoverRate: number;
    mutationRate: number;
  },
  experiences?: ExperienceRecord[]
): FormulaGene[] {
  const nextGen: FormulaGene[] = [];

  // 1. 精英保留: 最优个体直接进入下一代
  const sorted = [...currentPopulation].sort((a, b) => b.fitness - a.fitness);
  const elites = sorted.slice(0, plan.eliteCount);
  for (const e of elites) {
    nextGen.push({ ...e, source: 'elitism' });
  }

  // 2. 从避坑经验中提取变异引导
  const pitfallFormulas = new Set(
    (experiences ?? [])
      .filter((e) => e.type === 'pitfall_avoidance' || e.type === 'failure')
      .map((e) => e.formula)
      .filter(Boolean) as string[]
  );

  // 3. 交叉 + 变异填充剩余种群
  while (nextGen.length < plan.populationSize) {
    const parent1 = selection(currentPopulation);
    const parent2 = selection(currentPopulation);

    let child: FormulaGene;
    if (Math.random() < plan.crossoverRate) {
      child = crossover(parent1, parent2);
    } else {
      child = { ...parent1, source: 'crossover' };
    }

    if (Math.random() < plan.mutationRate) {
      child = mutate(child, featureVars);
    }

    // 避坑: 如果生成的公式在失败列表中, 重新变异
    let attempts = 0;
    while (pitfallFormulas.has(child.formula) && attempts < 3) {
      child = mutate(child, featureVars);
      attempts++;
    }

    nextGen.push(child);
  }

  return nextGen.slice(0, plan.populationSize);
}

// ===== 从公式字符串创建基因 =====
export function createGene(
  formula: string,
  fitness: number,
  error: number,
  complexity: number,
  generation: number,
  source: string = 'evaluated'
): FormulaGene {
  let tree: ExprNode;
  try {
    tree = parseExpression(formula);
  } catch {
    tree = { type: 'num', value: 0 };
  }
  return {
    formula,
    tree,
    fitness,
    error,
    complexity,
    source,
    generation,
  };
}

// ===== 辅助函数 =====

interface NodeRef {
  node: ExprNode;
  path: number[];
}

function collectNodes(node: ExprNode, path: number[] = []): NodeRef[] {
  const refs: NodeRef[] = [{ node, path }];

  switch (node.type) {
    case 'binop':
      refs.push(...collectNodes(node.left, [...path, 0]));
      refs.push(...collectNodes(node.right, [...path, 1]));
      break;
    case 'unaryop':
      refs.push(...collectNodes(node.operand, [...path, 0]));
      break;
    case 'call':
      node.args.forEach((arg, i) => {
        refs.push(...collectNodes(arg, [...path, i]));
      });
      break;
  }
  return refs;
}

function replaceNode(root: ExprNode, path: number[], newSubtree: ExprNode): void {
  if (path.length === 0) {
    // 替换根节点 - 需要原地修改
    Object.assign(root, newSubtree);
    return;
  }

  let current: ExprNode = root;
  for (let i = 0; i < path.length - 1; i++) {
    const idx = path[i];
    if (current.type === 'binop') {
      current = idx === 0 ? current.left : current.right;
    } else if (current.type === 'unaryop') {
      current = current.operand;
    } else if (current.type === 'call') {
      current = current.args[idx];
    }
  }

  const lastIdx = path[path.length - 1];
  if (current.type === 'binop') {
    if (lastIdx === 0) current.left = newSubtree;
    else current.right = newSubtree;
  } else if (current.type === 'unaryop') {
    current.operand = newSubtree;
  } else if (current.type === 'call') {
    current.args[lastIdx] = newSubtree;
  }
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
