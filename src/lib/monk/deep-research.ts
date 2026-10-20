/**
 * 深度调研模块 - DeepResearch
 * 搜索已有研究的相关公式，进行批判式记录
 */
import { webSearch, readPage, chat } from './llm';
import type { ProblemData } from './types';

export interface ResearchFinding {
  source: string;
  url: string;
  formula: string;
  description: string;
  critique: string; // 批判式评价
  relevance: number; // 0-1
}

export interface ResearchResult {
  query: string;
  findings: ResearchFinding[];
  synthesis: string;
  candidateFormulas: string[];
}

export async function deepResearch(
  problem: ProblemData,
  explorationHints: string[]
): Promise<ResearchResult> {
  const { domain, targetVar, featureVars, title } = problem;

  // 构造搜索查询
  const queries = [
    `${domain} ${targetVar} prediction formula ${featureVars.join(' ')}`,
    `${domain} empirical correlation ${targetVar} ${featureVars.slice(0, 2).join(' ')}`,
    `${title} mathematical model equation`,
  ];

  const findings: ResearchFinding[] = [];
  const seenUrls = new Set<string>();

  for (const q of queries.slice(0, 2)) {
    const results = await webSearch(q, 4);
    // 读取前2个页面
    for (const r of results.slice(0, 2)) {
      if (seenUrls.has(r.url)) continue;
      seenUrls.add(r.url);
      const page = await readPage(r.url);
      if (!page) continue;

      // 用 LLM 提取公式和批判
      const extraction = await chat([
        {
          role: 'system',
          content: '你是一个科学文献分析专家。从给定网页内容中提取与目标问题相关的数学公式、经验关系或物理定律，并对每个公式进行批判式评价（适用条件、局限性、是否可能适用于当前数据）。只返回 JSON。',
        },
        {
          role: 'user',
          content: `问题: ${title}\n领域: ${domain}\n目标变量: ${targetVar}\n特征变量: ${featureVars.join(', ')}\n数据探索提示: ${explorationHints.join('; ')}\n\n网页标题: ${page.title}\n网页内容:\n${page.content.slice(0, 3000)}\n\n请提取相关公式，返回JSON数组，每个元素: {"formula": "公式表达式(只用变量名和运算符)", "description": "公式说明", "critique": "批判式评价", "relevance": 0-1的数字}。如果没有相关公式返回 []。`,
        },
      ], { temperature: 0.3 });

      const parsed = tryParseJSON(extraction);
      if (Array.isArray(parsed)) {
        for (const f of parsed) {
          if (f.formula) {
            findings.push({
              source: page.title,
              url: r.url,
              formula: f.formula,
              description: f.description ?? '',
              critique: f.critique ?? '',
              relevance: typeof f.relevance === 'number' ? f.relevance : 0.5,
            });
          }
        }
      }
    }
  }

  // 如果没有搜索到结果，使用 LLM 的领域知识
  if (findings.length === 0) {
    const knowledgeResult = await chat([
      {
        role: 'system',
        content: '你是一个科学领域专家。根据问题描述，给出该领域已知的经典公式或经验关系，并进行批判式评价。返回JSON数组。',
      },
      {
        role: 'user',
        content: `领域: ${domain}\n问题: ${title}\n目标变量: ${targetVar}\n特征变量: ${featureVars.join(', ')}\n数据探索提示: ${explorationHints.join('; ')}\n\n请给出该领域已知的3-5个经典公式或经验关系。返回JSON数组: [{"formula": "表达式", "description": "说明", "critique": "批判", "relevance": 0-1}]`,
      },
    ], { temperature: 0.4 });
    const parsed = tryParseJSON(knowledgeResult);
    if (Array.isArray(parsed)) {
      for (const f of parsed) {
        if (f.formula) {
          findings.push({
            source: 'LLM 领域知识',
            url: '',
            formula: f.formula,
            description: f.description ?? '',
            critique: f.critique ?? '',
            relevance: typeof f.relevance === 'number' ? f.relevance : 0.5,
          });
        }
      }
    }
  }

  // 综合分析
  const topFindings = findings
    .filter((f) => f.relevance > 0.3)
    .sort((a, b) => b.relevance - a.relevance)
    .slice(0, 5);

  const synthesis = await chat([
    {
      role: 'system',
      content: '你是一个科研综述专家。综合以下调研发现，总结该问题的研究现状、已有公式的优缺点，并给出对当前数据集可能适用的公式建议。',
    },
    {
      role: 'user',
      content: `问题: ${title}\n领域: ${domain}\n目标: 用 ${featureVars.join(', ')} 预测 ${targetVar}\n\n调研发现:\n${topFindings.map((f, i) => `${i + 1}. 公式: ${f.formula}\n   说明: ${f.description}\n   批判: ${f.critique}\n   相关度: ${f.relevance}`).join('\n\n')}\n\n请综合分析，给出200-400字的综述。`,
    },
  ], { temperature: 0.5 });

  // 候选公式
  const candidateFormulas = topFindings.map((f) => f.formula);

  return {
    query: queries.join(' | '),
    findings,
    synthesis,
    candidateFormulas,
  };
}

function tryParseJSON(text: string): any {
  // 尝试从文本中提取 JSON
  const jsonMatch = text.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  const jsonStr = jsonMatch ? jsonMatch[1] : text;
  try {
    return JSON.parse(jsonStr.trim());
  } catch {
    // 尝试找到第一个 [ 到最后一个 ]
    const start = jsonStr.indexOf('[');
    const end = jsonStr.lastIndexOf(']');
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(jsonStr.slice(start, end + 1));
      } catch {
        return [];
      }
    }
    return [];
  }
}

export function buildResearchWiki(
  problem: ProblemData,
  result: ResearchResult
): string {
  let md = `### 深度调研报告\n\n`;
  md += `**搜索查询**: ${result.query}\n\n`;
  md += `**发现数量**: ${result.findings.length}\n\n`;

  md += `#### 调研发现\n\n`;
  for (const f of result.findings) {
    md += `**公式**: \`${f.formula}\`\n\n`;
    md += `- 来源: ${f.source}${f.url ? ` ([链接](${f.url}))` : ''}\n`;
    md += `- 说明: ${f.description}\n`;
    md += `- 批判: ${f.critique}\n`;
    md += `- 相关度: ${f.relevance}\n\n`;
  }

  md += `#### 综合分析\n\n${result.synthesis}\n\n`;

  md += `#### 候选公式\n\n`;
  for (const f of result.candidateFormulas) {
    md += `- \`${f}\`\n`;
  }

  return md;
}
