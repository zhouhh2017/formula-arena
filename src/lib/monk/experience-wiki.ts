/**
 * llm-wiki 经验记忆系统
 * 基于 karpathy 的 llm-wiki 理念:
 *   - 持久化: 所有经验落盘, 不丢失
 *   - 可累积: 后代在前人经验基础上增量构建
 *   - 跨代复用: 后代通过 ingest 前人经验来避免重复踩坑
 *
 * 经验类型:
 *   - success: 成功的发现 (好公式/好方法)
 *   - failure: 失败的尝试 (走不通的路)
 *   - insight: 数据洞察 (相关性/分布特征)
 *   - pitfall_avoidance: 避坑指南 (已知陷阱)
 *   - technique: 有效技巧 (参数优化策略等)
 *   - contradiction: 矛盾发现 (假设与数据冲突)
 */
import { db } from '@/lib/db';

export interface ExperienceRecord {
  id?: string;
  problemId: string;
  generationId?: string;
  generation?: number;
  type: 'success' | 'failure' | 'insight' | 'pitfall_avoidance' | 'technique' | 'contradiction';
  title: string;
  content: string;
  tags?: string[];
  formula?: string;
  error?: number;
  importance?: number; // 0-1
}

export class ExperienceWiki {
  constructor(private problemId: string) {}

  // ===== Ingest: 记录一条经验 (核心操作) =====
  async ingest(record: Omit<ExperienceRecord, 'problemId'>): Promise<string> {
    const created = await db.experienceEntry.create({
      data: {
        problemId: this.problemId,
        generationId: record.generationId,
        type: record.type,
        title: record.title,
        content: record.content,
        tags: JSON.stringify(record.tags ?? []),
        formula: record.formula,
        error: record.error,
        importance: record.importance ?? 0.5,
      },
    });
    return created.id;
  }

  // ===== Query: 检索经验 (按类型/标签/重要性) =====
  async query(options?: {
    type?: string;
    tags?: string[];
    minImportance?: number;
    limit?: number;
  }): Promise<ExperienceRecord[]> {
    const where: Record<string, unknown> = { problemId: this.problemId };
    if (options?.type) where.type = options.type;
    if (options?.minImportance !== undefined) {
      where.importance = { gte: options.minImportance };
    }

    const entries = await db.experienceEntry.findMany({
      where,
      orderBy: { importance: 'desc' },
      take: options?.limit ?? 50,
    });

    let result = entries.map(this.deserialize);
    if (options?.tags && options.tags.length > 0) {
      result = result.filter((e) =>
        e.tags?.some((t) => options.tags!.includes(t))
      );
    }
    return result;
  }

  // ===== 获取所有经验 (供后代 ingest) =====
  async getAll(): Promise<ExperienceRecord[]> {
    const entries = await db.experienceEntry.findMany({
      where: { problemId: this.problemId },
      orderBy: [{ importance: 'desc' }, { createdAt: 'asc' }],
    });
    return entries.map(this.deserialize);
  }

  // ===== 获取指定代数之前的所有经验 (后代可见) =====
  async getAncestralExperiences(beforeGeneration: number): Promise<ExperienceRecord[]> {
    const entries = await db.experienceEntry.findMany({
      where: { problemId: this.problemId },
      orderBy: [{ importance: 'desc' }, { createdAt: 'asc' }],
    });
    // 过滤: 只返回 beforeGeneration 之前记录的经验
    const result: ExperienceRecord[] = [];
    for (const e of entries) {
      if (e.generationId) {
        const gen = await db.monkGeneration.findUnique({
          where: { id: e.generationId },
          select: { generation: true },
        });
        if (gen && gen.generation < beforeGeneration) {
          result.push(this.deserialize(e));
        }
      } else {
        result.push(this.deserialize(e));
      }
    }
    return result;
  }

  // ===== Lint: 检查经验库质量, 标记矛盾/过时 =====
  async lint(): Promise<{ contradictions: string[]; staleCount: number; total: number }> {
    const all = await this.getAll();
    const contradictions: string[] = [];

    // 检查: 同一公式既出现在 success 又出现在 failure
    const successFormulas = new Set(
      all.filter((e) => e.type === 'success' && e.formula).map((e) => e.formula)
    );
    const failureFormulas = new Set(
      all.filter((e) => e.type === 'failure' && e.formula).map((e) => e.formula)
    );
    for (const f of successFormulas) {
      if (failureFormulas.has(f)) {
        contradictions.push(`公式 "${f}" 同时被标记为成功和失败`);
      }
    }

    return {
      contradictions,
      staleCount: all.filter((e) => (e.importance ?? 0) < 0.3).length,
      total: all.length,
    };
  }

  // ===== 将经验汇总为 LLM 上下文 (供后代参考) =====
  async asContext(maxChars = 6000): Promise<string> {
    const all = await this.getAll();
    if (all.length === 0) return '(暂无前人经验)';

    const sections: Record<string, ExperienceRecord[]> = {
      success: [],
      failure: [],
      insight: [],
      pitfall_avoidance: [],
      technique: [],
      contradiction: [],
    };
    for (const e of all) {
      if (sections[e.type]) sections[e.type].push(e);
    }

    let context = `# 前人探索经验 (共 ${all.length} 条)\n\n`;

    const typeLabels: Record<string, string> = {
      success: '✅ 成功发现',
      failure: '❌ 失败尝试 (避免重复)',
      insight: '💡 数据洞察',
      pitfall_avoidance: '⚠️ 避坑指南',
      technique: '🔧 有效技巧',
      contradiction: '⚡ 矛盾发现',
    };

    for (const [type, label] of Object.entries(typeLabels)) {
      const items = sections[type];
      if (items.length === 0) continue;
      context += `## ${label} (${items.length})\n\n`;
      for (const e of items.slice(0, 8)) {
        context += `### ${e.title}\n`;
        context += `${e.content}\n`;
        if (e.formula) context += `公式: \`${e.formula}\` (误差: ${e.error?.toFixed(4) ?? 'N/A'})\n`;
        context += `\n`;
        if (context.length > maxChars) break;
      }
      if (context.length > maxChars) break;
    }

    return context.slice(0, maxChars);
  }

  // ===== 标记经验被某代引用 =====
  async markReferenced(experienceId: string, generation: number): Promise<void> {
    const entry = await db.experienceEntry.findUnique({ where: { id: experienceId } });
    if (!entry) return;
    const refs: number[] = JSON.parse(entry.referencedBy);
    if (!refs.includes(generation)) {
      refs.push(generation);
      await db.experienceEntry.update({
        where: { id: experienceId },
        data: { referencedBy: JSON.stringify(refs) },
      });
    }
  }

  private deserialize = (e: any): ExperienceRecord => ({
    id: e.id,
    problemId: e.problemId,
    generationId: e.generationId,
    type: e.type,
    title: e.title,
    content: e.content,
    tags: JSON.parse(e.tags),
    formula: e.formula ?? undefined,
    error: e.error ?? undefined,
    importance: e.importance,
  });
}
