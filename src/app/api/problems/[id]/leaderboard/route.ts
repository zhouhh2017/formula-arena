import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { updateLeaderboard } from '@/lib/evaluator-api';

// GET /api/problems/[id]/leaderboard - 获取排行榜
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    await updateLeaderboard(id);
    const submissions = await db.submission.findMany({
      where: { problemId: id, status: 'evaluated' },
      orderBy: { overallScore: 'desc' },
    });

    const leaderboard = submissions.map((s, i) => ({
      rank: i + 1,
      id: s.id,
      submitterName: s.submitterName,
      formula: s.formula,
      rmse: s.fittingError,
      r2: s.r2Score,
      mae: s.maeScore,
      interpretability: s.interpretabilityScore,
      complexity: s.complexityScore,
      overallScore: s.overallScore,
      isMonk: s.isMonk,
      createdAt: s.createdAt,
    }));

    return NextResponse.json({ leaderboard });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
