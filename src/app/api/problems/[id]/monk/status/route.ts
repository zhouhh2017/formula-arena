import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET /api/problems/[id]/monk/status - 获取扫地僧运行状态
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const runs = await db.agentRun.findMany({
      where: { problemId: id },
      orderBy: { startedAt: 'desc' },
      take: 5,
      include: { steps: true },
    });

    return NextResponse.json({
      runs: runs.map((r) => ({
        id: r.id,
        round: r.round,
        status: r.status,
        currentStep: r.currentStep,
        bestFormula: r.bestFormula,
        bestError: r.bestError,
        startedAt: r.startedAt,
        completedAt: r.completedAt,
        steps: r.steps.map((s) => ({
          id: s.id,
          type: s.type,
          name: s.name,
          status: s.status,
          output: s.output,
          duration: s.duration,
          createdAt: s.createdAt,
        })),
      })),
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
