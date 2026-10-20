/**
 * 数学表达式引擎 - 安全的解析与求值
 * 支持变量、基本运算、数学函数
 */

export type ExprNode =
  | { type: 'num'; value: number }
  | { type: 'var'; name: string }
  | { type: 'binop'; op: string; left: ExprNode; right: ExprNode }
  | { type: 'unaryop'; op: string; operand: ExprNode }
  | { type: 'call'; name: string; args: ExprNode[] };

// ===== Tokenizer =====
const TOKEN_SPEC: [string, RegExp][] = [
  ['NUMBER', /\d+\.\d+|\d+\.|\.\d+|\d+/],
  ['IDENT', /[a-zA-Z_]\w*/],
  ['OP', /[+\-*/^()]/],
  ['COMMA', /,/],
  ['WS', /\s+/],
];

interface Token {
  type: string;
  value: string;
}

export function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < input.length) {
    let matched = false;
    for (const [type, regex] of TOKEN_SPEC) {
      // Anchor each alternative separately to avoid partial matches
      const alternatives = regex.source.split('|');
      let matchedThis = false;
      for (const alt of alternatives) {
        const re = new RegExp('^(?:' + alt + ')');
        const m = re.exec(input.slice(i));
        if (m) {
          if (type !== 'WS') {
            tokens.push({ type, value: m[0] });
          }
          i += m[0].length;
          matched = true;
          matchedThis = true;
          break;
        }
      }
      if (matchedThis) break;
    }
    if (!matched) {
      throw new Error(`无法识别的字符: "${input[i]}" 在位置 ${i}`);
    }
  }
  return tokens;
}

// ===== Parser (递归下降) =====
class Parser {
  private pos = 0;
  constructor(private tokens: Token[]) {}

  private peek(): Token | null {
    return this.tokens[this.pos] ?? null;
  }

  private consume(): Token {
    const t = this.tokens[this.pos];
    if (!t) throw new Error('意外的表达式结束');
    this.pos++;
    return t;
  }

  private expect(type: string, value?: string): Token {
    const t = this.peek();
    if (!t || t.type !== type || (value && t.value !== value)) {
      throw new Error(`期望 ${type} ${value ?? ''}, 得到 ${t?.value ?? 'EOF'}`);
    }
    return this.consume();
  }

  parse(): ExprNode {
    const node = this.parseExpr();
    if (this.pos < this.tokens.length) {
      throw new Error(`多余的 token: ${this.tokens[this.pos].value}`);
    }
    return node;
  }

  private parseExpr(): ExprNode {
    let left = this.parseTerm();
    while (this.peek()?.type === 'OP' && (this.peek()!.value === '+' || this.peek()!.value === '-')) {
      const op = this.consume().value;
      const right = this.parseTerm();
      left = { type: 'binop', op, left, right };
    }
    return left;
  }

  private parseTerm(): ExprNode {
    let left = this.parseFactor();
    while (this.peek()?.type === 'OP' && (this.peek()!.value === '*' || this.peek()!.value === '/')) {
      const op = this.consume().value;
      const right = this.parseFactor();
      left = { type: 'binop', op, left, right };
    }
    return left;
  }

  private parseFactor(): ExprNode {
    const base = this.parseUnary();
    if (this.peek()?.type === 'OP' && this.peek()!.value === '^') {
      this.consume();
      const exp = this.parseFactor();
      return { type: 'binop', op: '^', left: base, right: exp };
    }
    return base;
  }

  private parseUnary(): ExprNode {
    if (this.peek()?.type === 'OP' && (this.peek()!.value === '+' || this.peek()!.value === '-')) {
      const op = this.consume().value;
      const operand = this.parseUnary();
      return { type: 'unaryop', op, operand };
    }
    return this.parsePrimary();
  }

  private parsePrimary(): ExprNode {
    const t = this.peek();
    if (!t) throw new Error('意外的表达式结束');

    if (t.type === 'NUMBER') {
      this.consume();
      return { type: 'num', value: parseFloat(t.value) };
    }

    if (t.type === 'IDENT') {
      this.consume();
      if (this.peek()?.type === 'OP' && this.peek()!.value === '(') {
        this.consume();
        const args: ExprNode[] = [];
        if (!(this.peek()?.type === 'OP' && this.peek()!.value === ')')) {
          args.push(this.parseExpr());
          while (this.peek()?.type === 'COMMA') {
            this.consume();
            args.push(this.parseExpr());
          }
        }
        this.expect('OP', ')');
        return { type: 'call', name: t.value, args };
      }
      return { type: 'var', name: t.value };
    }

    if (t.type === 'OP' && t.value === '(') {
      this.consume();
      const node = this.parseExpr();
      this.expect('OP', ')');
      return node;
    }

    throw new Error(`意外的 token: ${t.value}`);
  }
}

export function parseExpression(input: string): ExprNode {
  const tokens = tokenize(input);
  return new Parser(tokens).parse();
}

// ===== Evaluator =====
const MATH_FUNCS: Record<string, (args: number[]) => number> = {
  sin: (a) => Math.sin(a[0]),
  cos: (a) => Math.cos(a[0]),
  tan: (a) => Math.tan(a[0]),
  exp: (a) => Math.exp(a[0]),
  log: (a) => Math.log(Math.abs(a[0])),
  ln: (a) => Math.log(Math.abs(a[0])),
  log10: (a) => Math.log10(Math.abs(a[0])),
  sqrt: (a) => Math.sqrt(Math.max(0, a[0])),
  abs: (a) => Math.abs(a[0]),
  pow: (a) => Math.pow(a[0], a[1]),
  max: (a) => Math.max(...a),
  min: (a) => Math.min(...a),
  sinh: (a) => Math.sinh(a[0]),
  cosh: (a) => Math.cosh(a[0]),
  tanh: (a) => Math.tanh(a[0]),
  atan: (a) => Math.atan(a[0]),
  asin: (a) => Math.asin(a[0]),
  acos: (a) => Math.acos(a[0]),
};

const CONSTANTS: Record<string, number> = {
  pi: Math.PI,
  e: Math.E,
};

export function evalNode(node: ExprNode, vars: Record<string, number>): number {
  switch (node.type) {
    case 'num':
      return node.value;
    case 'var':
      if (node.name in vars) return vars[node.name];
      if (node.name in CONSTANTS) return CONSTANTS[node.name];
      throw new Error(`未知变量: ${node.name}`);
    case 'binop': {
      const l = evalNode(node.left, vars);
      const r = evalNode(node.right, vars);
      switch (node.op) {
        case '+': return l + r;
        case '-': return l - r;
        case '*': return l * r;
        case '/': return l / (r === 0 ? 1e-10 : r);
        case '^': return Math.pow(l, r);
      }
      throw new Error(`未知运算符: ${node.op}`);
    }
    case 'unaryop': {
      const v = evalNode(node.operand, vars);
      return node.op === '-' ? -v : v;
    }
    case 'call': {
      const fn = MATH_FUNCS[node.name];
      if (!fn) throw new Error(`未知函数: ${node.name}`);
      const args = node.args.map((a) => evalNode(a, vars));
      return fn(args);
    }
  }
}

export function evaluateFormula(formula: string, vars: Record<string, number>): number {
  try {
    const node = parseExpression(formula);
    return evalNode(node, vars);
  } catch {
    return NaN;
  }
}

export function evaluateFormulaBatch(
  formula: string,
  samples: Record<string, number>[]
): number[] {
  const node = parseExpression(formula);
  return samples.map((s) => {
    try {
      return evalNode(node, s);
    } catch {
      return NaN;
    }
  });
}

export function nodeToString(node: ExprNode): string {
  switch (node.type) {
    case 'num':
      return Number.isInteger(node.value) ? String(node.value) : node.value.toFixed(4);
    case 'var':
      return node.name;
    case 'binop': {
      const l = nodeToString(node.left);
      const r = nodeToString(node.right);
      if (node.op === '^') return `${l}^${r}`;
      return `${l} ${node.op} ${r}`;
    }
    case 'unaryop':
      return `${node.op}${nodeToString(node.operand)}`;
    case 'call':
      return `${node.name}(${node.args.map(nodeToString).join(', ')})`;
  }
}

export function formulaToString(formula: string): string {
  try {
    return nodeToString(parseExpression(formula));
  } catch {
    return formula;
  }
}

export function countNodes(node: ExprNode): number {
  switch (node.type) {
    case 'num':
    case 'var':
      return 1;
    case 'binop':
      return 1 + countNodes(node.left) + countNodes(node.right);
    case 'unaryop':
      return 1 + countNodes(node.operand);
    case 'call':
      return 1 + node.args.reduce((s, a) => s + countNodes(a), 0);
  }
}

export function formulaComplexity(formula: string): number {
  try {
    return countNodes(parseExpression(formula));
  } catch {
    return 999;
  }
}

export function extractVariables(node: ExprNode): Set<string> {
  const vars = new Set<string>();
  function walk(n: ExprNode) {
    switch (n.type) {
      case 'var':
        if (!(n.name in CONSTANTS)) vars.add(n.name);
        break;
      case 'binop':
        walk(n.left);
        walk(n.right);
        break;
      case 'unaryop':
        walk(n.operand);
        break;
      case 'call':
        n.args.forEach(walk);
        break;
    }
  }
  walk(node);
  return vars;
}

export function getFormulaVariables(formula: string): string[] {
  try {
    return Array.from(extractVariables(parseExpression(formula)));
  } catch {
    return [];
  }
}

export function validateFormula(formula: string): { valid: boolean; error?: string } {
  try {
    parseExpression(formula);
    return { valid: true };
  } catch (e) {
    return { valid: false, error: (e as Error).message };
  }
}
