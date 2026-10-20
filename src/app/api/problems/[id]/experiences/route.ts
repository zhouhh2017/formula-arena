import { NextRequest, NextResponse } from 'next/server';
import { ExperienceWiki } from '@/lib/monk/experience-wiki';

// GET /api/problems/[id]/experiences - 获取 llm-wiki 经验记录
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const url = new URL(req.url);
    const type = url.searchParams.get('type') || undefined;

    const expWiki = new ExperienceWiki(id);
    const experiences = await expWiki.query({ type });

    return NextResponse.json({ experiences });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
