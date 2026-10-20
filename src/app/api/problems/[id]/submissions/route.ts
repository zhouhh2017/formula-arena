import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { evaluateSubmission, updateLeaderboard } from '@/lib/evaluator-api';
import type { ProblemData } from '@/lib/monk/types';

// GET /api/problems/[id]/submissions - 列出提交
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const submissions = await db.submission.findMany({
      where: { problemId: id },
      orderBy: { rank: 'asc' },
    });
    return NextResponse.json({ submissions });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}

// POST /api/problems/[id]/submissions - 提交公式
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await req.json();
    const { submitterName, formula, report } = body;

    if (!formula || !submitterName) {
      return NextResponse.json({ error: '缺少提交者名称或公式' }, { status: 400 });
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

    const submission = await db.submission.create({
      data: {
        problemId: id,
        submitterName,
        formula,
        report: report || '',
        fittingError: evaluation.rmse,
        r2Score: evaluation.r2,
        maeScore: evaluation.mae,
        interpretabilityScore: evaluation.interpretability,
        complexityScore: evaluation.complexity,
        overallScore: evaluation.overallScore,
        isMonk: false,
        status: evaluation.valid ? 'evaluated' : 'failed',
      },
    });

    await updateLeaderboard(id);

    return NextResponse.json({
      id: submission.id,
      evaluation,
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
