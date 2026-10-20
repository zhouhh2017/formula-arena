"""
数学表达式引擎 - 安全的解析与求值
支持变量、基本运算、数学函数

移植自: src/lib/monk/math-engine.ts
"""
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Set

# ===== AST 节点类型 =====
@dataclass
class NumNode:
    value: float

@dataclass
class VarNode:
    name: str

@dataclass
class BinOpNode:
    op: str
    left: 'ExprNode'
    right: 'ExprNode'

@dataclass
class UnaryOpNode:
    op: str
    operand: 'ExprNode'

@dataclass
class CallNode:
    name: str
    args: List['ExprNode']

ExprNode = Union[NumNode, VarNode, BinOpNode, UnaryOpNode, CallNode]

# ===== Tokenizer =====
TOKEN_SPEC = [
    ('NUMBER', re.compile(r'\d+\.\d+|\d+\.|\.\d+|\d+')),
    ('IDENT', re.compile(r'[a-zA-Z_]\w*')),
    ('OP', re.compile(r'[+\-*/^()]')),
    ('COMMA', re.compile(r',')),
    ('WS', re.compile(r'\s+')),
]

@dataclass
class Token:
    type: str
    value: str

def tokenize(input_str: str) -> List[Token]:
    tokens = []
    i = 0
    while i < len(input_str):
        matched = False
        for ttype, regex in TOKEN_SPEC:
            m = regex.match(input_str, i)
            if m:
                if ttype != 'WS':
                    tokens.append(Token(ttype, m.group()))
                i = m.end()
                matched = True
                break
        if not matched:
            raise ValueError(f'无法识别的字符: "{input_str[i]}" 在位置 {i}')
    return tokens

# ===== Parser (递归下降) =====
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, ttype: str, value: str = None) -> Token:
        t = self.peek()
        if t is None or t.type != ttype or (value and t.value != value):
            raise ValueError(f'期望 {ttype} {value or ""}, 得到 {t.value if t else "EOF"}')
        return self.consume()

    def parse(self) -> ExprNode:
        node = self.parse_expr()
        if self.pos < len(self.tokens):
            raise ValueError(f'多余的 token: {self.tokens[self.pos].value}')
        return node

    def parse_expr(self) -> ExprNode:
        left = self.parse_term()
        while self.peek() and self.peek().type == 'OP' and self.peek().value in ('+', '-'):
            op = self.consume().value
            right = self.parse_term()
            left = BinOpNode(op, left, right)
        return left

    def parse_term(self) -> ExprNode:
        left = self.parse_factor()
        while self.peek() and self.peek().type == 'OP' and self.peek().value in ('*', '/'):
            op = self.consume().value
            right = self.parse_factor()
            left = BinOpNode(op, left, right)
        return left

    def parse_factor(self) -> ExprNode:
        base = self.parse_unary()
        if self.peek() and self.peek().type == 'OP' and self.peek().value == '^':
            self.consume()
            exp = self.parse_factor()
            return BinOpNode('^', base, exp)
        return base

    def parse_unary(self) -> ExprNode:
        if self.peek() and self.peek().type == 'OP' and self.peek().value in ('+', '-'):
            op = self.consume().value
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        return self.parse_primary()

    def parse_primary(self) -> ExprNode:
        t = self.peek()
        if t is None:
            raise ValueError('意外的表达式结束')

        if t.type == 'NUMBER':
            self.consume()
            return NumNode(float(t.value))

        if t.type == 'IDENT':
            self.consume()
            if self.peek() and self.peek().type == 'OP' and self.peek().value == '(':
                self.consume()
                args = []
                if not (self.peek() and self.peek().type == 'OP' and self.peek().value == ')'):
                    args.append(self.parse_expr())
                    while self.peek() and self.peek().type == 'COMMA':
                        self.consume()
                        args.append(self.parse_expr())
                self.expect('OP', ')')
                return CallNode(t.value, args)
            return VarNode(t.value)

        if t.type == 'OP' and t.value == '(':
            self.consume()
            node = self.parse_expr()
            self.expect('OP', ')')
            return node

        raise ValueError(f'意外的 token: {t.value}')

def parse_expression(input_str: str) -> ExprNode:
    tokens = tokenize(input_str)
    return Parser(tokens).parse()

# ===== Evaluator =====
MATH_FUNCS = {
    'sin': lambda a: math.sin(a[0]),
    'cos': lambda a: math.cos(a[0]),
    'tan': lambda a: math.tan(a[0]),
    'exp': lambda a: math.exp(a[0]),
    'log': lambda a: math.log(max(abs(a[0]), 1e-300)),
    'ln': lambda a: math.log(max(abs(a[0]), 1e-300)),
    'log10': lambda a: math.log10(max(abs(a[0]), 1e-300)),
    'sqrt': lambda a: math.sqrt(max(0, a[0])),
    'abs': lambda a: abs(a[0]),
    'pow': lambda a: a[0] ** a[1],
    'max': lambda a: max(a),
    'min': lambda a: min(a),
    'sinh': lambda a: math.sinh(a[0]),
    'cosh': lambda a: math.cosh(a[0]),
    'tanh': lambda a: math.tanh(a[0]),
    'atan': lambda a: math.atan(a[0]),
    'asin': lambda a: math.asin(max(-1, min(1, a[0]))),
    'acos': lambda a: math.acos(max(-1, min(1, a[0]))),
}

CONSTANTS = {
    'pi': math.pi,
    'e': math.e,
}

def eval_node(node: ExprNode, vars: Dict[str, float]) -> float:
    if isinstance(node, NumNode):
        return node.value
    elif isinstance(node, VarNode):
        if node.name in vars:
            return vars[node.name]
        if node.name in CONSTANTS:
            return CONSTANTS[node.name]
        raise ValueError(f'未知变量: {node.name}')
    elif isinstance(node, BinOpNode):
        l = eval_node(node.left, vars)
        r = eval_node(node.right, vars)
        if node.op == '+': return l + r
        elif node.op == '-': return l - r
        elif node.op == '*': return l * r
        elif node.op == '/': return l / (r if r != 0 else 1e-10)
        elif node.op == '^':
            try:
                result = l ** r
                if isinstance(result, complex):
                    return float('nan')
                return float(result)
            except (OverflowError, ValueError, ZeroDivisionError):
                return float('nan')
    elif isinstance(node, UnaryOpNode):
        v = eval_node(node.operand, vars)
        return -v if node.op == '-' else v
    elif isinstance(node, CallNode):
        fn = MATH_FUNCS.get(node.name)
        if fn is None:
            raise ValueError(f'未知函数: {node.name}')
        args = [eval_node(a, vars) for a in node.args]
        try:
            return fn(args)
        except (ValueError, OverflowError, ZeroDivisionError):
            return float('nan')
    return float('nan')

def evaluate_formula(formula: str, vars: Dict[str, float]) -> float:
    try:
        node = parse_expression(formula)
        return eval_node(node, vars)
    except:
        return float('nan')

def evaluate_formula_batch(formula: str, samples: List[Dict[str, float]]) -> List[float]:
    node = parse_expression(formula)
    results = []
    for s in samples:
        try:
            results.append(eval_node(node, s))
        except:
            results.append(float('nan'))
    return results

def node_to_string(node: ExprNode) -> str:
    if isinstance(node, NumNode):
        if node.value == int(node.value):
            return str(int(node.value))
        return f'{node.value:.4f}'
    elif isinstance(node, VarNode):
        return node.name
    elif isinstance(node, BinOpNode):
        l = node_to_string(node.left)
        r = node_to_string(node.right)
        if node.op == '^':
            return f'{l}^{r}'
        return f'{l} {node.op} {r}'
    elif isinstance(node, UnaryOpNode):
        return f'{node.op}{node_to_string(node.operand)}'
    elif isinstance(node, CallNode):
        args = ', '.join(node_to_string(a) for a in node.args)
        return f'{node.name}({args})'
    return ''

def formula_to_string(formula: str) -> str:
    try:
        return node_to_string(parse_expression(formula))
    except:
        return formula

def count_nodes(node: ExprNode) -> int:
    if isinstance(node, (NumNode, VarNode)):
        return 1
    elif isinstance(node, BinOpNode):
        return 1 + count_nodes(node.left) + count_nodes(node.right)
    elif isinstance(node, UnaryOpNode):
        return 1 + count_nodes(node.operand)
    elif isinstance(node, CallNode):
        return 1 + sum(count_nodes(a) for a in node.args)
    return 0

def formula_complexity(formula: str) -> int:
    try:
        return count_nodes(parse_expression(formula))
    except:
        return 999

def extract_variables(node: ExprNode) -> Set[str]:
    vars_set = set()
    def walk(n):
        if isinstance(n, VarNode):
            if n.name not in CONSTANTS:
                vars_set.add(n.name)
        elif isinstance(n, BinOpNode):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, UnaryOpNode):
            walk(n.operand)
        elif isinstance(n, CallNode):
            for a in n.args:
                walk(a)
    walk(node)
    return vars_set

def get_formula_variables(formula: str) -> List[str]:
    try:
        return list(extract_variables(parse_expression(formula)))
    except:
        return []

def validate_formula(formula: str) -> dict:
    try:
        parse_expression(formula)
        return {'valid': True}
    except Exception as e:
        return {'valid': False, 'error': str(e)}

# ===== 统计工具 =====
def compute_rmse(predictions: List[float], targets: List[float]) -> float:
    sum_sq = 0.0
    count = 0
    for p, t in zip(predictions, targets):
        if not (math.isnan(p) or math.isinf(p)):
            sum_sq += (p - t) ** 2
            count += 1
    return math.sqrt(sum_sq / count) if count > 0 else 1e10

def compute_mae(predictions: List[float], targets: List[float]) -> float:
    sum_abs = 0.0
    count = 0
    for p, t in zip(predictions, targets):
        if not (math.isnan(p) or math.isinf(p)):
            sum_abs += abs(p - t)
            count += 1
    return sum_abs / count if count > 0 else 1e10

def compute_r2(predictions: List[float], targets: List[float]) -> float:
    valid_idx = [i for i, (p, t) in enumerate(zip(predictions, targets))
                 if not (math.isnan(p) or math.isinf(p))]
    if len(valid_idx) < 2:
        return float('-inf')
    mean_t = sum(targets[i] for i in valid_idx) / len(valid_idx)
    ss_res = sum((targets[i] - predictions[i]) ** 2 for i in valid_idx)
    ss_tot = sum((targets[i] - mean_t) ** 2 for i in valid_idx)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0
