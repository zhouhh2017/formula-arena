/**
 * 公式假设生成模块
 * 扫地僧综合调研信息使用 LLM 提出公式假设
 */
import { chat, extractFormulaFromText } from './llm';
import type { ProblemData, Hypothesis } from './types';
import { validateFormula } from './math-engine';

export async function generateHypotheses(
  problem: ProblemData,
  wikiContext: string,
  round: number,
  evaluatorFeedback?: string
): Promise<Hypothesis[]> {
  const { targetVar, featureVars, domain } = problem;

  const systemPrompt = `你是一个科学公式发现专家。你的任务是根据已有调研和数据探索信息，提出候选的数学公式假设来预测目标变量。

要求:
1. 公式必须使用给定的特征变量名
2. 公式必须是合法的数学表达式 (支持 +, -, *, /, ^, sin, cos, exp, log, sqrt, abs 等)
3. 每个公式附带理由和置信度
4. 公式形式要多样化: 线性、非线性、交互项、对数、幂律等
5. 返回 JSON 数组格式

可用变量: ${featureVars.join(', ')}
目标变量: ${targetVar}
领域: ${domain}`;

  const userPrompt = evaluatorFeedback
    ? `## 第 ${round} 轮假设生成 (基于评估反馈改进)

${wikiContext}

## 上一轮评估反馈
${evaluatorFeedback}

请基于反馈提出 ${3 + round} 个改进的公式假设。每个假设要针对反馈中提到的问题进行改进。
返回 JSON: [{"formula": "表达式", "rationale": "理由", "confidence": 0-1}]`
    : `## 第 ${round} 轮假设生成

${wikiContext}

请提出 ${3 + round} 个多样化的公式假设。
返回 JSON: [{"formula": "表达式", "rationale": "理由", "confidence": 0-1}]`;

  const result = await chat(
    [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
    { temperature: 0.8 }
  );

  const parsed = tryParseHypotheses(result);
  const hypotheses: Hypothesis[] = [];

  for (const p of parsed) {
    const formula = p.formula || extractFormulaFromText(p.rationale || '');
    if (!formula) continue;
    const validation = validateFormula(formula);
    if (!validation.valid) continue;
    hypotheses.push({
      id: `h_${round}_${hypotheses.length + 1}`,
      round,
      formula,
      rationale: p.rationale || '',
      source: evaluatorFeedback ? 'evaluator_feedback' : 'llm',
      confidence: typeof p.confidence === 'number' ? p.confidence : 0.5,
    });
  }

  // 如果 LLM 没有给出有效公式，补充一些基础模板
  if (hypotheses.length === 0) {
    hypotheses.push(...generateFallbackHypotheses(problem, round));
  }

  return hypotheses;
}

function tryParseHypotheses(text: string): any[] {
  const jsonMatch = text.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  const jsonStr = jsonMatch ? jsonMatch[1] : text;
  try {
    const parsed = JSON.parse(jsonStr.trim());
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    const start = jsonStr.indexOf('[');
    const end = jsonStr.lastIndexOf(']');
    if (start >= 0 && end > start) {
      try {
        const parsed = JSON.parse(jsonStr.slice(start, end + 1));
        return Array.isArray(parsed) ? parsed : [];
      } catch {
        return [];
      }
    }
    return [];
  }
}

function generateFallbackHypotheses(problem: ProblemData, round: number): Hypothesis[] {
  const { featureVars, targetVar } = problem;
  const hypotheses: Hypothesis[] = [];

  // 线性
  if (featureVars.length >= 1) {
    hypotheses.push({
      id: `h_${round}_lin`,
      round,
      formula: `a * ${featureVars[0]} + b`,
      rationale: '基础线性假设',
      source: 'llm',
      confidence: 0.5,
    });
  }

  // 多元线性
  if (featureVars.length >= 2) {
    const terms = featureVars.map((v) => `c${featureVars.indexOf(v)} * ${v}`).join(' + ');
    hypotheses.push({
      id: `h_${round}_multi`,
      round,
      formula: terms,
      rationale: '多元线性假设',
      source: 'llm',
      confidence: 0.5,
    });
  }

  // 幂律
  if (featureVars.length >= 1) {
    hypotheses.push({
      id: `h_${round}_power`,
      round,
      formula: `a * ${featureVars[0]}^b`,
      rationale: '幂律关系假设',
      source: 'llm',
      confidence: 0.4,
    });
  }

  return hypotheses;
}

export function buildHypothesisWiki(
  hypotheses: Hypothesis[],
  round: number
): string {
  let md = `### 第 ${round} 轮公式假设\n\n`;
  md += `共提出 ${hypotheses.length} 个候选公式:\n\n`;
  for (const h of hypotheses) {
    md += `#### 假设 ${h.id}\n\n`;
    md += `- **公式**: \`${h.formula}\`\n`;
    md += `- **理由**: ${h.rationale}\n`;
    md += `- **来源**: ${h.source}\n`;
    md += `- **置信度**: ${h.confidence}\n\n`;
  }
  return md;
}
