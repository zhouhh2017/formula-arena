import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { evaluateSubmission } from '@/lib/evaluator-api';
import type { ProblemData } from '@/lib/monk/types';

// POST /api/problems/[id]/evaluate - 评估公式 (不提交)
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await req.json();
    const { formula } = body;

    if (!formula) {
      return NextResponse.json({ error: '缺少公式' }, { status: 400 });
    }

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
    const evaluation = evaluateSubmission(problemData, formula, metrics);

    return NextResponse.json({ evaluation });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
