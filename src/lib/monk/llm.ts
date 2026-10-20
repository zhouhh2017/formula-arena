/**
 * LLM 集成层 - 基于 z-ai-web-dev-sdk
 * 提供对话补全、网络搜索、网页阅读能力
 */
import ZAI from 'z-ai-web-dev-sdk';

let zaiInstance: ZAI | null = null;

export async function getZAI(): Promise<ZAI> {
  if (!zaiInstance) {
    zaiInstance = await ZAI.create();
  }
  return zaiInstance;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export async function chat(
  messages: ChatMessage[],
  options?: { model?: string; temperature?: number }
): Promise<string> {
  try {
    const zai = await getZAI();
    const res = await zai.chat.completions.create({
      messages,
      model: options?.model,
      temperature: options?.temperature ?? 0.7,
      stream: false,
    } as any);
    // 兼容不同返回结构
    const content =
      res?.choices?.[0]?.message?.content ??
      res?.message?.content ??
      res?.content ??
      (typeof res === 'string' ? res : JSON.stringify(res));
    return content;
  } catch (e) {
    return `[LLM调用失败: ${(e as Error).message}]`;
  }
}

export interface SearchResult {
  url: string;
  name: string;
  snippet: string;
  host_name: string;
}

export async function webSearch(query: string, num = 5): Promise<SearchResult[]> {
  try {
    const zai = await getZAI();
    const res = await zai.functions.invoke('web_search', { query, num });
    if (Array.isArray(res)) return res as SearchResult[];
    return [];
  } catch {
    return [];
  }
}

export interface PageContent {
  title: string;
  url: string;
  content: string;
  publishedTime?: string;
}

export async function readPage(url: string): Promise<PageContent | null> {
  try {
    const zai = await getZAI();
    const res = await zai.functions.invoke('page_reader', { url });
    if (res?.data?.html) {
      // 简单的 HTML -> text
      const text = htmlToText(res.data.html).slice(0, 6000);
      return {
        title: res.data.title ?? url,
        url: res.data.url,
        content: text,
        publishedTime: res.data.publishedTime,
      };
    }
    return null;
  } catch {
    return null;
  }
}

function htmlToText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

// 从 LLM 输出中提取公式 (寻找 y = ... 或 formula: ... 等模式)
export function extractFormulaFromText(text: string): string | null {
  // 尝试多种模式
  const patterns = [
    /```(?:math|formula|python)?\s*\n?\s*(y\s*=\s*[^\n`]+)\n?\s*```/i,
    /```(?:math|formula|python)?\s*\n?\s*([^\n`=]+(?:\+|\-|\*|\/|\^)[^\n`]+)\n?\s*```/i,
    /\*\*公式[^*]*:\*\*\s*`?([^\n`]+)`?/i,
    /公式[：:]\s*`?([^\n`]+)`?/i,
    /formula[：:]\s*`?([^\n`]+)`?/i,
    /y\s*=\s*([^\n`]{3,200})/i,
  ];
  for (const p of patterns) {
    const m = text.match(p);
    if (m) {
      let f = (m[1] ?? m[0]).trim();
      // 去掉前缀 y =
      f = f.replace(/^y\s*=\s*/i, '').trim();
      // 去掉尾部的说明文字
      f = f.split(/[,，。；;\n]/)[0].trim();
      if (f.length >= 2) return f;
    }
  }
  return null;
}
