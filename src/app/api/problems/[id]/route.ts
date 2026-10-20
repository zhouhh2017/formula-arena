import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import type { ProblemData } from '@/lib/monk/types';

// GET /api/problems/[id] - 获取赛题详情
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
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

    return NextResponse.json({
      ...problemData,
      metrics: JSON.parse(problem.metrics),
      publisherName: problem.publisherName,
      status: problem.status,
      createdAt: problem.createdAt,
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
