/**
 * 代际规划器
 *
 * 基于 karpathy llm-wiki 理念:
 *   - ingest 前代经验 (从 wiki 读取)
 *   - 综合经验 + 遗传算法 + LLM 生成下一代探索计划
 *   - 计划落盘, 等待用户审批 (默认 autoExecute=false)
 *
 * 工作流:
 *   1. 读取前代所有经验 (success/failure/insight/pitfall)
 *   2. 用遗传算法从前代种群中选择精英 + 交叉/变异产生种子公式
 *   3. 用 LLM 综合经验生成探索策略 (focusAreas, avoidPitfalls)
 *   4. 生成 EvolutionPlan 并落盘为 MonkGeneration 记录
 */
import { db } from '@/lib/db';
import { chat } from './llm';
import { ExperienceWiki } from './experience-wiki';
import {
  createGene,
  evolvePopulation,
  type FormulaGene,
  type EvolutionPlan,
} from './genetic-evolution';
import type { ProblemData, SRResult, EvaluationReport } from './types';

export interface GenerationSummary {
  id: string;
  generation: number;
  status: string;
  bestFormula?: string;
  bestError?: number;
  bestScore?: number;
  totalExperiences: number;
  parentFormula?: string | null;
  parentError?: number | null;
  parentGen?: number | null;
  autoExecute?: boolean;
  plan?: any;
  createdAt?: Date;
  approvedAt?: Date | null;
  startedAt?: Date | null;
  completedAt?: Date | null;
}

// ===== 获取某赛题的所有代际记录 =====
export async function getGenerations(problemId: string): Promise<GenerationSummary[]> {
  const gens = await db.monkGeneration.findMany({
    where: { problemId },
    orderBy: { generation: 'asc' },
    include: { _count: { select: { experiences: true } } },
  });
  return gens.map((g) => ({
    id: g.id,
    generation: g.generation,
    status: g.status,
    bestFormula: g.bestFormula ?? undefined,
    bestError: g.bestError ?? undefined,
    bestScore: g.bestScore ?? undefined,
    totalExperiences: g._count.experiences,
    parentFormula: g.parentFormula,
    parentError: g.parentError,
    parentGen: g.parentGen,
    autoExecute: g.autoExecute,
    plan: JSON.parse(g.planJson),
    createdAt: g.createdAt,
    approvedAt: g.approvedAt,
    startedAt: g.startedAt,
    completedAt: g.completedAt,
  }));
}

// ===== 从前代结果中提取种群 (用于遗传算法) =====
function extractPopulation(
  srResults: SRResult[],
  evaluation?: EvaluationReport,
  generation: number = 1
): FormulaGene[] {
  const population: FormulaGene[] = [];

  for (const sr of srResults) {
    if (sr.success && sr.formula) {
      population.push(
        createGene(
          sr.formula,
          100 - Math.min(sr.rmse * 10, 80), // 适应度: 误差越小越高
          sr.rmse,
          sr.complexity,
          generation,
          'sr_' + sr.method
        )
      );
    }
  }

  if (evaluation) {
    population.push(
      createGene(
        evaluation.formula,
        evaluation.overallScore,
        evaluation.rmse,
        evaluation.complexity,
        generation,
        'best_evaluated'
      )
    );
  }

  return population;
}

// ===== 为下一代生成探索计划 =====
export async function planNextGeneration(
  problem: ProblemData,
  options: { autoExecute?: boolean } = {}
): Promise<{ id: string; generation: number; status: string; autoExecute: boolean; plan: EvolutionPlan }> {
  const expWiki = new ExperienceWiki(problem.id);

  // 1. 获取所有已有代际记录, 确定下一代编号
  const existingGens = await db.monkGeneration.findMany({
    where: { problemId: problem.id },
    orderBy: { generation: 'desc' },
  });
  const nextGeneration = (existingGens[0]?.generation ?? 0) + 1;

  // 2. 读取前代所有经验 (llm-wiki ingest)
  const experiences = await expWiki.getAncestralExperiences(nextGeneration);

  // 3. 从前代最佳公式构建种群
  const parentPopulation: FormulaGene[] = [];
  for (const g of existingGens) {
    if (g.bestFormula && g.bestError !== null) {
      parentPopulation.push(
        createGene(
          g.bestFormula,
          g.bestScore ?? (100 - Math.min(g.bestError * 10, 80)),
          g.bestError,
          0, // complexity unknown
          g.generation,
          'gen_' + g.generation
        )
      );
    }
  }

  // 4. 遗传算法: 进化产生种子公式
  const eliteCount = Math.min(2, parentPopulation.length);
  let seedGenes: FormulaGene[] = [];

  if (parentPopulation.length >= 2) {
    seedGenes = evolvePopulation(parentPopulation, problem.featureVars, {
      populationSize: Math.min(5, parentPopulation.length + 2),
      eliteCount,
      crossoverRate: 0.7,
      mutationRate: 0.4,
    }, experiences);
  } else if (parentPopulation.length === 1) {
    // 只有一个父本, 通过变异产生种子
    seedGenes = [parentPopulation[0]];
    for (let i = 0; i < 3; i++) {
      const mutated = evolvePopulation(parentPopulation, problem.featureVars, {
        populationSize: 1,
        eliteCount: 0,
        crossoverRate: 0,
        mutationRate: 1.0,
      }, experiences);
      seedGenes.push(...mutated);
    }
  }

  // 5. 提取避坑列表和变异引导
  const avoidPitfalls = experiences
    .filter((e) => e.type === 'pitfall_avoidance' || e.type === 'failure')
    .map((e) => `${e.title}: ${e.content.slice(0, 100)}`)
    .slice(0, 5);

  const mutationGuides = experiences
    .filter((e) => e.type === 'technique' || e.type === 'insight')
    .map((e) => `${e.title}: ${e.content.slice(0, 100)}`)
    .slice(0, 5);

  // 6. 用 LLM 生成探索策略
  const expContext = await expWiki.asContext(4000);
  const focusAreas = await generateFocusAreas(problem, expContext, parentPopulation[0]);

  const plan: EvolutionPlan = {
    generation: nextGeneration,
    populationSize: Math.max(4, seedGenes.length),
    eliteCount,
    crossoverRate: 0.7,
    mutationRate: 0.4,
    seedGenes,
    mutationGuides,
    avoidPitfalls,
    focusAreas,
  };

  // 7. 落盘为 MonkGeneration 记录
  const parentBest = parentPopulation.sort((a, b) => b.fitness - a.fitness)[0];
  const generation = await db.monkGeneration.create({
    data: {
      problemId: problem.id,
      generation: nextGeneration,
      status: 'planned',
      parentFormula: parentBest?.formula,
      parentError: parentBest?.error,
      parentGen: parentBest?.generation,
      planJson: JSON.stringify(plan),
      autoExecute: options.autoExecute ?? false,
    },
  });

  return {
    id: generation.id,
    generation: nextGeneration,
    status: 'planned',
    autoExecute: options.autoExecute ?? false,
    plan,
  };
}

// ===== LLM 生成探索重点 =====
async function generateFocusAreas(
  problem: ProblemData,
  experienceContext: string,
  parentBest?: FormulaGene
): Promise<string[]> {
  const prompt = `你是一个科学公式发现专家。基于前人的探索经验, 为下一代扫地僧规划探索重点。

## 问题
${problem.title}
目标变量: ${problem.targetVar}
特征变量: ${problem.featureVars.join(', ')}

## 前人经验
${experienceContext}

## 前代最佳结果
${parentBest ? `公式: ${parentBest.formula}\nRMSE: ${parentBest.error.toFixed(4)}\n适应度: ${parentBest.fitness.toFixed(2)}` : '无'}

请列出 3-5 个下一代应该重点探索的方向。每个方向一行, 简洁说明理由。
格式: 方向名称 - 理由`;

  const response = await chat(
    [
      { role: 'system', content: '你是科学公式发现专家, 擅长规划探索策略。' },
      { role: 'user', content: prompt },
    ],
    { temperature: 0.7 }
  );

  const lines = response
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 5 && !l.startsWith('#') && !l.startsWith('```'))
    .slice(0, 5);

  return lines.length > 0 ? lines : ['改进参数优化', '尝试非线性变换', '探索交互项'];
}

// ===== 获取代际计划详情 =====
export async function getGenerationPlan(generationId: string) {
  const gen = await db.monkGeneration.findUnique({
    where: { id: generationId },
    include: { experiences: true },
  });
  if (!gen) return null;

  return {
    id: gen.id,
    generation: gen.generation,
    status: gen.status,
    parentFormula: gen.parentFormula,
    parentError: gen.parentError,
    parentGen: gen.parentGen,
    plan: JSON.parse(gen.planJson) as EvolutionPlan,
    autoExecute: gen.autoExecute,
    bestFormula: gen.bestFormula,
    bestError: gen.bestError,
    bestScore: gen.bestScore,
    createdAt: gen.createdAt,
    approvedAt: gen.approvedAt,
    startedAt: gen.startedAt,
    completedAt: gen.completedAt,
    experiences: gen.experiences.map((e) => ({
      id: e.id,
      type: e.type,
      title: e.title,
      content: e.content,
      formula: e.formula,
      error: e.error,
      importance: e.importance,
    })),
  };
}

// Alias for API compatibility
export const getGenerationDetail = getGenerationPlan;

// ===== 更新代际计划 (用户修改) =====
export async function updateGenerationPlan(
  generationId: string,
  updates: {
    autoExecute?: boolean;
    plan?: Partial<EvolutionPlan>;
    status?: string;
  }
) {
  const current = await db.monkGeneration.findUnique({ where: { id: generationId } });
  if (!current) throw new Error('代际记录不存在');

  const data: Record<string, unknown> = {};
  if (updates.autoExecute !== undefined) data.autoExecute = updates.autoExecute;
  if (updates.status !== undefined) {
    data.status = updates.status;
    if (updates.status === 'approved') data.approvedAt = new Date();
  }
  if (updates.plan) {
    const existingPlan = JSON.parse(current.planJson);
    data.planJson = JSON.stringify({ ...existingPlan, ...updates.plan });
  }

  return db.monkGeneration.update({ where: { id: generationId }, data });
}
