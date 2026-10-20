/**
 * Provider-neutral LLM, search, and page-reading tools over MCP Streamable HTTP.
 * The configured MCP server must expose the tool names and input contracts
 * documented in README.md.
 */
import { Client, StreamableHTTPClientTransport } from '@modelcontextprotocol/client';

let clientPromise: Promise<Client> | null = null;

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface SearchResult {
  url: string;
  name: string;
  snippet: string;
  host_name: string;
}

export interface PageContent {
  title: string;
  url: string;
  content: string;
  publishedTime?: string;
}

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} must be configured to connect the MCP server`);
  return value;
}

async function getMcpClient(): Promise<Client> {
  if (!clientPromise) {
    clientPromise = (async () => {
      const endpoint = new URL(requiredEnv('MCP_SERVER_URL'));
      const token = process.env.MCP_AUTH_TOKEN?.trim();
      const client = new Client({ name: 'formula-arena', version: '1.0.0' });
      const transport = new StreamableHTTPClientTransport(endpoint, {
        requestInit: token ? { headers: { Authorization: `Bearer ${token}` } } : undefined,
      });
      await client.connect(transport);
      return client;
    })().catch((error) => {
      clientPromise = null;
      throw error;
    });
  }
  return clientPromise;
}

function toolName(envName: string, fallback: string): string {
  return process.env[envName]?.trim() || fallback;
}

async function callMcpTool(name: string, args: Record<string, unknown>): Promise<unknown> {
  const client = await getMcpClient();
  const result = await client.callTool({ name, arguments: args });
  if (result.isError) {
    const message = result.content
      .filter((item) => item.type === 'text')
      .map((item) => item.text)
      .join('\n');
    throw new Error(message || `MCP tool "${name}" failed`);
  }

  if (result.structuredContent) return result.structuredContent;

  const text = result.content
    .filter((item) => item.type === 'text')
    .map((item) => item.text)
    .join('\n')
    .trim();
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asText(value: unknown): string | null {
  if (typeof value === 'string') return value;
  const record = asRecord(value);
  if (!record) return null;
  for (const key of ['text', 'content', 'response', 'answer']) {
    if (typeof record[key] === 'string') return record[key] as string;
  }
  const choices = record.choices;
  if (Array.isArray(choices) && choices.length > 0) {
    const first = asRecord(choices[0]);
    const message = asRecord(first?.message);
    if (typeof message?.content === 'string') return message.content;
  }
  return null;
}

function parseJsonText<T>(value: unknown): T | null {
  if (typeof value !== 'string') return null;
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
}

export async function chat(
  messages: ChatMessage[],
  options?: { model?: string; temperature?: number },
): Promise<string> {
  try {
    const response = await callMcpTool(
      toolName('MCP_CHAT_TOOL', 'chat'),
      {
        messages,
        ...(options?.model ? { model: options.model } : {}),
        temperature: options?.temperature ?? 0.7,
      },
    );
    const text = asText(response);
    if (text !== null) return text;
    return JSON.stringify(response);
  } catch (error) {
    return `[MCP调用失败: ${(error as Error).message}]`;
  }
}

export async function webSearch(query: string, num = 5): Promise<SearchResult[]> {
  try {
    const response = await callMcpTool(
      toolName('MCP_SEARCH_TOOL', 'web_search'),
      { query, num },
    );
    const record = asRecord(response);
    const results = Array.isArray(response)
      ? response
      : record?.results ?? record?.items;
    if (!Array.isArray(results)) return [];
    return results
      .map((item) => {
        const result = asRecord(item);
        if (!result || typeof result.url !== 'string') return null;
        return {
          url: result.url,
          name:
            typeof result.name === 'string'
              ? result.name
              : typeof result.title === 'string'
                ? result.title
                : result.url,
          snippet: typeof result.snippet === 'string' ? result.snippet : '',
          host_name:
            typeof result.host_name === 'string'
              ? result.host_name
              : (() => {
                  try {
                    return new URL(result.url as string).hostname;
                  } catch {
                    return '';
                  }
                })(),
        } satisfies SearchResult;
      })
      .filter((item): item is SearchResult => item !== null);
  } catch {
    return [];
  }
}

export async function readPage(url: string): Promise<PageContent | null> {
  try {
    const response = await callMcpTool(
      toolName('MCP_READ_PAGE_TOOL', 'page_reader'),
      { url },
    );
    const record = asRecord(response) ?? asRecord(parseJsonText(response));
    const content = asText(record?.content) ?? asText(response);
    if (!content) return null;
    return {
      title: typeof record?.title === 'string' ? record.title : url,
      url: typeof record?.url === 'string' ? record.url : url,
      content: /<\/?(?:html|body|main|article|p|div)\b/i.test(content)
        ? htmlToText(content).slice(0, 6000)
        : content.slice(0, 6000),
      ...(typeof record?.publishedTime === 'string'
        ? { publishedTime: record.publishedTime }
        : {}),
    };
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
  const patterns = [
    /```(?:math|formula|python)?\s*\n?\s*(y\s*=\s*[^\n`]+)\n?\s*```/i,
    /```(?:math|formula|python)?\s*\n?\s*([^\n`=]+(?:\+|\-|\*|\/|\^)[^\n`]+)\n?\s*```/i,
    /\*\*公式[^*]*:\*\*\s*`?([^\n`]+)`?/i,
    /公式[：:]\s*`?([^\n`]+)`?/i,
    /formula[：:]\s*`?([^\n`]+)`?/i,
    /y\s*=\s*([^\n`]{3,200})/i,
  ];
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      let formula = (match[1] ?? match[0]).trim();
      formula = formula.replace(/^y\s*=\s*/i, '').trim();
      formula = formula.split(/[,，。；;\n]/)[0].trim();
      if (formula.length >= 2) return formula;
    }
  }
  return null;
}
