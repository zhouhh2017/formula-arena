import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { runMonkGeneration } from '@/lib/monk/orchestrator';
import type { ProblemData } from '@/lib/monk/types';

// POST /api/problems/[id]/monk/run - 启动初代扫地僧
// 现在会创建 MonkGeneration 记录 (generation=1), 并记录经验到 llm-wiki
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await req.json().catch(() => ({}));
    const maxRounds = body.maxRounds ?? 2;

    const problem = await db.problem.findUnique({ where: { id } });
    if (!problem) {
      return NextResponse.json({ error: '赛题不存在' }, { status: 404 });
    }

    const problemData: ProblemData = {
      id: problem.id,
      title: problem.title,
      description: problem.description,
      domain: problem.domain,
      targetVar: problem.targetVar,
      featureVars: JSON.parse(problem.featureVars),
      samples: JSON.parse(problem.dataPoints),
      hiddenFormula: problem.hiddenFormula ?? undefined,
    };

    const metrics = JSON.parse(problem.metrics);

    // 创建初代 MonkGeneration 记录
    const gen = await db.monkGeneration.create({
      data: {
        problemId: id,
        generation: 1,
        status: 'running',
        autoExecute: true, // 初代直接执行
        startedAt: new Date(),
        planJson: JSON.stringify({
          generation: 1,
          populationSize: 4,
          eliteCount: 1,
          crossoverRate: 0.5,
          mutationRate: 0.3,
          seedGenes: [],
          mutationGuides: [],
          avoidPitfalls: [],
          focusAreas: ['data_exploration', 'deep_research', 'multi_method_sr'],
        }),
      },
    });

    // 执行初代扫地僧
    const result = await runMonkGeneration(problemData, metrics, gen.id, 1, {
      maxRounds,
      maxSRSteps: 4,
    });

    return NextResponse.json({
      runId: result.runId,
      generationId: gen.id,
      generation: 1,
      bestFormula: result.bestFormula,
      bestError: result.bestError,
      rounds: result.rounds.length,
      bestReport: result.bestReport,
    });
  } catch (e) {
    console.error('Monk run error:', e);
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
