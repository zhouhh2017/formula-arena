/**
 * Wiki 记忆持久化层
 * 扫地僧的所有调研、假设、评估结果都落盘到 Wiki
 */
import { db } from '@/lib/db';
import type { WikiPageData } from './types';

export class WikiStore {
  constructor(private problemId: string) {}

  async write(page: Omit<WikiPageData, 'problemId'>): Promise<string> {
    const created = await db.wikiPage.create({
      data: {
        problemId: this.problemId,
        title: page.title,
        content: page.content,
        category: page.category,
        round: page.round,
        tags: JSON.stringify(page.tags ?? []),
      },
    });
    return created.id;
  }

  async list(category?: string, round?: number): Promise<WikiPageData[]> {
    const where: Record<string, unknown> = { problemId: this.problemId };
    if (category) where.category = category;
    if (round !== undefined) where.round = round;
    const pages = await db.wikiPage.findMany({
      where,
      orderBy: { createdAt: 'desc' },
    });
    return pages.map((p) => ({
      id: p.id,
      problemId: p.problemId,
      title: p.title,
      content: p.content,
      category: p.category as WikiPageData['category'],
      round: p.round,
      tags: JSON.parse(p.tags),
    }));
  }

  async readAll(): Promise<WikiPageData[]> {
    const pages = await db.wikiPage.findMany({
      where: { problemId: this.problemId },
      orderBy: { createdAt: 'asc' },
    });
    return pages.map((p) => ({
      id: p.id,
      problemId: p.problemId,
      title: p.title,
      content: p.content,
      category: p.category as WikiPageData['category'],
      round: p.round,
      tags: JSON.parse(p.tags),
    }));
  }

  // 将所有 wiki 内容汇总为上下文 (供 LLM 参考)
  async asContext(maxChars = 8000): Promise<string> {
    const pages = await this.readAll();
    if (pages.length === 0) return '(暂无历史记录)';
    let context = '';
    for (const p of pages) {
      const entry = `## [Round ${p.round}] ${p.title}\n类别: ${p.category}\n${p.content}\n\n`;
      if (context.length + entry.length > maxChars) break;
      context += entry;
    }
    return context;
  }

  async getByRound(round: number): Promise<WikiPageData[]> {
    return this.list(undefined, round);
  }
}
