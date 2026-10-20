import { NextResponse } from 'next/server';
import { seedDemoProblems } from '@/lib/seed';

// POST /api/seed - 初始化示例数据
export async function POST() {
  try {
    await seedDemoProblems();
    return NextResponse.json({ success: true });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
