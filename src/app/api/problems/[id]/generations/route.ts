import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { planNextGeneration, getGenerations } from '@/lib/monk/generation-planner';
import type { ProblemData } from '@/lib/monk/types';

// GET /api/problems/[id]/generations - 获取所有代际记录
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const generations = await getGenerations(id);
    return NextResponse.json({ generations });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}

// POST /api/problems/[id]/generations - 规划下一代扫地僧
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await req.json().catch(() => ({}));
    const autoExecute = body.autoExecute ?? false;

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

    const generation = await planNextGeneration(problemData, { autoExecute });

    return NextResponse.json({
      generationId: generation.id,
      generation: generation.generation,
      status: generation.status,
      autoExecute: generation.autoExecute,
      plan: generation.plan,
    });
  } catch (e) {
    console.error('Plan generation error:', e);
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
