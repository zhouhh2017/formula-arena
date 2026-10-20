import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { seedDemoProblems } from '@/lib/seed';

// GET /api/problems - 列出所有赛题
export async function GET() {
  try {
    // 自动 seed 如果为空
    await seedDemoProblems();
    const problems = await db.problem.findMany({
      orderBy: { createdAt: 'desc' },
      include: {
        _count: {
          select: { submissions: true },
        },
      },
    });
    return NextResponse.json({
      problems: problems.map((p) => ({
        id: p.id,
        title: p.title,
        domain: p.domain,
        targetVar: p.targetVar,
        featureVars: JSON.parse(p.featureVars),
        status: p.status,
        publisherName: p.publisherName,
        submissionCount: p._count.submissions,
        hiddenFormula: p.hiddenFormula,
        createdAt: p.createdAt,
      })),
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}

// POST /api/problems - 创建新赛题
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      title,
      description,
      domain,
      targetVar,
      featureVars,
      samples,
      hiddenFormula,
      metrics,
      publisherName,
    } = body;

    if (!title || !description || !targetVar || !featureVars || !samples) {
      return NextResponse.json({ error: '缺少必填字段' }, { status: 400 });
    }

    const problem = await db.problem.create({
      data: {
        title,
        description,
        domain: domain || 'custom',
        dataUrl: `/api/problems/__data__`,
        targetVar,
        featureVars: JSON.stringify(featureVars),
        dataPoints: JSON.stringify(samples),
        hiddenFormula: hiddenFormula || null,
        metrics: JSON.stringify(metrics || { errorWeight: 0.6, interpWeight: 0.4 }),
        publisherName: publisherName || '匿名',
        status: 'open',
      },
    });

    return NextResponse.json({ id: problem.id });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
