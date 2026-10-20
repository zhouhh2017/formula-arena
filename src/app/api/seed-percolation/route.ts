import { NextResponse } from 'next/server';
import { seedPercolationProblem } from '@/lib/seed-percolation';

// POST /api/seed-percolation - 初始化渗流赛题
export async function POST() {
  try {
    await seedPercolationProblem();
    return NextResponse.json({ success: true });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
