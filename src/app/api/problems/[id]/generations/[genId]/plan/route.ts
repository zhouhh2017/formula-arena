import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { updateGenerationPlan, getGenerationDetail } from '@/lib/monk/generation-planner';

// GET /api/problems/[id]/generations/[genId]/plan - 获取代际计划详情
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string; genId: string }> }
) {
  try {
    const { genId } = await params;
    const detail = await getGenerationDetail(genId);
    return NextResponse.json({ generation: detail });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}

// PATCH /api/problems/[id]/generations/[genId]/plan - 修改代际计划
export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string; genId: string }> }
) {
  try {
    const { genId } = await params;
    const body = await req.json();
    const updates: { autoExecute?: boolean; plan?: any; status?: string } = {};

    if (body.autoExecute !== undefined) updates.autoExecute = body.autoExecute;
    if (body.status !== undefined) updates.status = body.status;
    if (body.plan !== undefined) updates.plan = body.plan;

    await updateGenerationPlan(genId, updates);
    const detail = await getGenerationDetail(genId);
    return NextResponse.json({ generation: detail });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
