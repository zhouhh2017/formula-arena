import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { runMonkGeneration } from '@/lib/monk/orchestrator';
import { getGenerationDetail } from '@/lib/monk/generation-planner';
import { ExperienceWiki } from '@/lib/monk/experience-wiki';
import type { ProblemData } from '@/lib/monk/types';

// POST /api/problems/[id]/generations/[genId]/execute - 执行某一代扫地僧
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string; genId: string }> }
) {
  try {
    const { id, genId } = await params;
    const problem = await db.problem.findUnique({ where: { id } });
    if (!problem) {
      return NextResponse.json({ error: '赛题不存在' }, { status: 404 });
    }

    const gen = await db.monkGeneration.findUnique({ where: { id: genId } });
    if (!gen) {
      return NextResponse.json({ error: '代际记录不存在' }, { status: 404 });
    }
    if (gen.status === 'running') {
      return NextResponse.json({ error: '该代正在运行中' }, { status: 400 });
    }
    if (gen.status === 'completed') {
      return NextResponse.json({ error: '该代已完成' }, { status: 400 });
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

    // 获取计划详情
    const detail = await getGenerationDetail(genId);
    const plan = detail.plan;

    // 从 llm-wiki 获取前人经验
    const expWiki = new ExperienceWiki(id);
    const experienceContext = await expWiki.asContext(4000);

    // 提取避坑指南
    const pitfalls = await expWiki.query({ type: 'pitfall_avoidance' });
    const avoidPitfalls = pitfalls.map(p => p.title + ': ' + p.content.slice(0, 200));

    // 执行
    const result = await runMonkGeneration(problemData, metrics, genId, gen.generation, {
      seedFormulas: plan.seedGenes?.map(g => g.formula),
      experienceContext,
      avoidPitfalls,
      maxRounds: 2,
      maxSRSteps: 4,
    });

    return NextResponse.json({
      runId: result.runId,
      bestFormula: result.bestFormula,
      bestError: result.bestError,
      rounds: result.rounds.length,
      bestReport: result.bestReport,
    });
  } catch (e) {
    console.error('Execute generation error:', e);
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
