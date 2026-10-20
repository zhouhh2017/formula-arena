import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET /api/problems/[id]/wiki - 获取扫地僧的 wiki 记录
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const url = new URL(req.url);
    const category = url.searchParams.get('category');
    const round = url.searchParams.get('round');

    const where: Record<string, unknown> = { problemId: id };
    if (category) where.category = category;
    if (round) where.round = parseInt(round);

    const pages = await db.wikiPage.findMany({
      where,
      orderBy: { createdAt: 'asc' },
    });

    return NextResponse.json({
      pages: pages.map((p) => ({
        id: p.id,
        title: p.title,
        content: p.content,
        category: p.category,
        round: p.round,
        tags: JSON.parse(p.tags),
        createdAt: p.createdAt,
      })),
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
