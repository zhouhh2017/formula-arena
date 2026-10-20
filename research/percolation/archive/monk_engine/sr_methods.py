"""
符号回归方法 - 4 种并行方法
1. 遗传规划 (Genetic Programming)
2. SINDy 稀疏回归
3. 模板匹配 (Template Matching)
4. 物理引导搜索 (Physics-Informed, 替代 LLM-guided)

移植自: src/lib/monk/sr/*.ts
"""
import math
import random
import copy
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

from .math_engine import (
    ExprNode, NumNode, VarNode, BinOpNode, UnaryOpNode, CallNode,
    parse_expression, eval_node, evaluate_formula_batch,
    formula_complexity, node_to_string, count_nodes,
    compute_rmse, compute_mae, compute_r2,
)


class SRResult:
    def __init__(self, method: str, formula: str, rmse: float, mae: float,
                 r2: float, complexity: int, iterations: int,
                 duration_ms: int, success: bool, notes: str = ''):
        self.method = method
        self.formula = formula
        self.rmse = rmse
        self.mae = mae
        self.r2 = r2
        self.complexity = complexity
        self.iterations = iterations
        self.duration_ms = duration_ms
        self.success = success
        self.notes = notes

    def to_dict(self):
        return {
            'method': self.method,
            'formula': self.formula,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'complexity': self.complexity,
            'iterations': self.iterations,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'notes': self.notes,
        }


# ============================================================
# 共享工具: Nelder-Mead 参数优化
# ============================================================

def extract_parameters(node: ExprNode, known_vars: set) -> List[str]:
    """提取公式中的参数符号 (非已知变量)"""
    params = set()
    CONST = {'pi', 'e'}
    def walk(n):
        if isinstance(n, VarNode):
            if n.name not in known_vars and n.name not in CONST:
                params.add(n.name)
        elif isinstance(n, BinOpNode):
            walk(n.left); walk(n.right)
        elif isinstance(n, UnaryOpNode):
            walk(n.operand)
        elif isinstance(n, CallNode):
            for a in n.args: walk(a)
    walk(node)
    return list(params)


def nelder_mead(objective, init_params, max_iter=200, tol=1e-8):
    """Nelder-Mead 单纯形优化"""
    n = len(init_params)
    if n == 0:
        return init_params, objective([]), 0

    simplex = []
    for i in range(n + 1):
        p = list(init_params)
        if i < n:
            p[i] = p[i] * 1.5 if p[i] != 0 else 0.1
        simplex.append((p, objective(p)))

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    iters = 0

    for _ in range(max_iter):
        simplex.sort(key=lambda x: x[1])
        best_val = simplex[0][1]
        worst_val = simplex[n][1]
        if worst_val - best_val < tol:
            break

        centroid = [0.0] * n
        for i in range(n):
            for j in range(n):
                centroid[j] += simplex[i][0][j]
        centroid = [c / n for c in centroid]

        # 反射
        reflected = [centroid[j] + alpha * (centroid[j] - simplex[n][0][j]) for j in range(n)]
        reflected_val = objective(reflected)

        if reflected_val >= simplex[0][1] and reflected_val < simplex[n - 1][1]:
            simplex[n] = (reflected, reflected_val)
        elif reflected_val < simplex[0][1]:
            expanded = [centroid[j] + gamma * (reflected[j] - centroid[j]) for j in range(n)]
            expanded_val = objective(expanded)
            if expanded_val < reflected_val:
                simplex[n] = (expanded, expanded_val)
            else:
                simplex[n] = (reflected, reflected_val)
        else:
            contracted = [centroid[j] + rho * (simplex[n][0][j] - centroid[j]) for j in range(n)]
            contracted_val = objective(contracted)
            if contracted_val < simplex[n][1]:
                simplex[n] = (contracted, contracted_val)
            else:
                for i in range(1, n + 1):
                    new_p = [(simplex[i][0][j] + simplex[0][0][j]) / 2 for j in range(n)]
                    simplex[i] = (new_p, objective(new_p))
        iters += 1

    simplex.sort(key=lambda x: x[1])
    return simplex[0][0], simplex[0][1], iters


def optimize_formula_params(formula: str, samples: List[Dict], feature_vars: List[str],
                            target_var: str, max_iter: int = 150) -> Tuple[str, float, Dict]:
    """优化公式中的参数"""
    node = parse_expression(formula)
    known_vars = set(feature_vars)
    param_names = extract_parameters(node, known_vars)

    targets = [s.get(target_var, 0) for s in samples]

    if not param_names:
        preds = evaluate_formula_batch(formula, samples)
        rmse = compute_rmse(preds, targets)
        return formula, rmse, {}

    def objective(params):
        param_map = {name: params[i] for i, name in enumerate(param_names)}
        sum_sq = 0.0
        count = 0
        for i, s in enumerate(samples):
            vars_dict = {**s, **param_map}
            try:
                pred = eval_node(node, vars_dict)
                if math.isnan(pred) or math.isinf(pred) or math.iscomplex(pred):
                    count += 1
                    continue
                err = pred - targets[i]
                sum_sq += err * err
                count += 1
            except:
                count += 1
        return math.sqrt(sum_sq / count) if count > 0 else 1e10

    best_result = None
    for r in range(3):
        init = [0.5 if r == 0 else -1 + random.random() * 2 for _ in param_names]
        params, val, iters = nelder_mead(objective, init, max_iter=max_iter // 3)
        if best_result is None or val < best_result[1]:
            best_result = (params, val)

    param_map = {name: best_result[0][i] for i, name in enumerate(param_names)}

    optimized = formula
    for name in sorted(param_names, key=len, reverse=True):
        val = param_map[name]
        val_str = f'{val:.6f}' if val >= 0 else f'({val:.6f})'
        optimized = re.sub(rf'\b{re.escape(name)}\b', val_str, optimized)

    return optimized, best_result[1], param_map


import re


# ============================================================
# 方法 1: 遗传规划 (Genetic Programming)
# ============================================================

FUNCTIONS = ['+', '-', '*', '/']
UNARY_FUNCS = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs']

def random_tree(depth: int, var_names: List[str], allow_params=False) -> ExprNode:
    terminals = list(var_names) + ['1', '2', '0.5', '0.1']
    if allow_params:
        terminals += ['a', 'b', 'c']

    if depth <= 0 or random.random() < 0.3:
        t = random.choice(terminals)
        if re.match(r'^-?\d+\.?\d*$', t):
            return NumNode(float(t))
        return VarNode(t)

    if random.random() < 0.2:
        fn = random.choice(UNARY_FUNCS)
        return CallNode(fn, [random_tree(depth - 1, var_names, allow_params)])

    op = random.choice(FUNCTIONS)
    return BinOpNode(op,
                     random_tree(depth - 1, var_names, allow_params),
                     random_tree(depth - 1, var_names, allow_params))


def mutate_tree(node: ExprNode, var_names: List[str], prob=0.2) -> ExprNode:
    if random.random() < prob:
        return random_tree(2, var_names, True)
    if isinstance(node, NumNode):
        if random.random() < 0.5:
            return NumNode(node.value + (random.random() - 0.5) * 0.5)
        return node
    elif isinstance(node, VarNode):
        return node
    elif isinstance(node, BinOpNode):
        if random.random() < 0.3:
            ops = [o for o in FUNCTIONS if o != node.op]
            return BinOpNode(random.choice(ops),
                             copy.deepcopy(node.left), copy.deepcopy(node.right))
        return BinOpNode(node.op,
                         mutate_tree(node.left, var_names, prob),
                         mutate_tree(node.right, var_names, prob))
    elif isinstance(node, UnaryOpNode):
        return UnaryOpNode(node.op, mutate_tree(node.operand, var_names, prob))
    elif isinstance(node, CallNode):
        return CallNode(node.name, [mutate_tree(a, var_names, prob) for a in node.args])
    return node


def crossover_tree(a: ExprNode, b: ExprNode) -> ExprNode:
    if random.random() < 0.3:
        return copy.deepcopy(b)
    if isinstance(a, BinOpNode):
        r = random.random()
        if r < 0.33:
            return BinOpNode(a.op, crossover_tree(a.left, b), copy.deepcopy(a.right))
        elif r < 0.66:
            return BinOpNode(a.op, copy.deepcopy(a.left), crossover_tree(a.right, b))
        else:
            return BinOpNode(a.op, copy.deepcopy(b), copy.deepcopy(a.right))
    elif isinstance(a, UnaryOpNode):
        return UnaryOpNode(a.op, crossover_tree(a.operand, b))
    elif isinstance(a, CallNode):
        return CallNode(a.name, [crossover_tree(arg, b) if random.random() < 0.5 else copy.deepcopy(arg)
                                  for arg in a.args])
    return copy.deepcopy(b)


def tree_to_formula(node: ExprNode) -> str:
    if isinstance(node, NumNode):
        return f'{node.value:.4f}'
    elif isinstance(node, VarNode):
        return node.name
    elif isinstance(node, BinOpNode):
        l = tree_to_formula(node.left)
        r = tree_to_formula(node.right)
        return f'({l} {node.op} {r})'
    elif isinstance(node, UnaryOpNode):
        return f'({node.op}{tree_to_formula(node.operand)})'
    elif isinstance(node, CallNode):
        args = ','.join(tree_to_formula(a) for a in node.args)
        return f'{node.name}({args})'
    return '0'


def run_genetic_programming(samples, feature_vars, target_var, hypothesis_formula,
                            max_steps=30, population_size=30):
    """遗传规划符号回归"""
    start_time = time.time()
    targets = [s.get(target_var, 0) for s in samples]

    def evaluate(tree):
        sum_sq = 0.0
        count = 0
        for s in samples:
            try:
                pred = eval_node(tree, s)
                if math.isnan(pred) or math.isinf(pred) or abs(pred) > 1e10:
                    count += 1
                    continue
                err = pred - targets[count]
                sum_sq += err * err
                count += 1
            except:
                pass
        if count < len(samples) * 0.5:
            return 1e10
        rmse = math.sqrt(sum_sq / count)
        if math.isnan(rmse) or math.isinf(rmse) or rmse > 1e8:
            return 1e10
        c = count_nodes(tree)
        return rmse * (1 + c * 0.001)

    def tournament(pop, size=3):
        best = random.choice(pop)
        for _ in range(size - 1):
            c = random.choice(pop)
            if c[1] < best[1]:
                best = c
        return best

    # 初始化种群
    population = []
    try:
        seed_tree = parse_expression(hypothesis_formula)
        for _ in range(population_size // 3):
            t = mutate_tree(seed_tree, feature_vars, 0.4)
            population.append((t, evaluate(t)))
    except:
        pass
    while len(population) < population_size:
        t = random_tree(3, feature_vars, True)
        population.append((t, evaluate(t)))

    best = min(population, key=lambda x: x[1])
    generation = 0

    for gen in range(max_steps):
        generation = gen + 1
        population.sort(key=lambda x: x[1])
        elite_count = max(2, population_size // 5)
        new_pop = list(population[:elite_count])

        while len(new_pop) < population_size:
            pa = tournament(population)
            pb = tournament(population)
            child = crossover_tree(pa[0], pb[0])
            mutated = mutate_tree(child, feature_vars, 0.3)
            new_pop.append((mutated, evaluate(mutated)))

        population = new_pop
        gen_best = min(population, key=lambda x: x[1])
        if gen_best[1] < best[1]:
            best = gen_best

    best_formula = tree_to_formula(best[0])
    rmse = best[1]

    # 验证 best 公式是否有效 - 避免除零/无穷
    try:
        test_preds = evaluate_formula_batch(best_formula, samples)
        valid_count = sum(1 for p in test_preds if not (math.isnan(p) or math.isinf(p)))
        if valid_count < len(samples) * 0.5:
            sorted_pop = sorted(population, key=lambda x: x[1])
            for candidate, fitness in sorted_pop:
                alt_formula = tree_to_formula(candidate)
                try:
                    alt_preds = evaluate_formula_batch(alt_formula, samples)
                    alt_valid = sum(1 for p in alt_preds if not (math.isnan(p) or math.isinf(p)))
                    if alt_valid >= len(samples) * 0.5:
                        best_formula = alt_formula
                        rmse = fitness
                        break
                except:
                    continue
    except:
        pass

    try:
        opt_formula, opt_rmse, _ = optimize_formula_params(
            best_formula, samples, feature_vars, target_var, max_iter=100)
        if opt_rmse < rmse:
            best_formula = opt_formula
            rmse = opt_rmse
    except:
        pass

    preds = evaluate_formula_batch(best_formula, samples)
    duration = int((time.time() - start_time) * 1000)

    return SRResult(
        method='遗传规划 (Genetic Programming)',
        formula=best_formula,
        rmse=rmse,
        mae=compute_mae(preds, targets),
        r2=compute_r2(preds, targets),
        complexity=formula_complexity(best_formula),
        iterations=generation,
        duration_ms=duration,
        success=rmse < 1e9,
        notes=f'种群{population_size}, 进化{generation}代',
    )


# ============================================================
# 方法 2: SINDy 稀疏回归
# ============================================================

def build_library(samples, feature_vars, max_degree=2):
    """构建候选特征库"""
    n = len(samples)
    terms = ['1']
    matrix = [[1.0] for _ in range(n)]

    for v in feature_vars:
        terms.append(v)
        for i in range(n):
            matrix[i].append(samples[i].get(v, 0))

    if max_degree >= 2:
        for i, v1 in enumerate(feature_vars):
            for j, v2 in enumerate(feature_vars):
                if j >= i:
                    term_name = f'{v1}^2' if i == j else f'{v1}*{v2}'
                    terms.append(term_name)
                    for k in range(n):
                        matrix[k].append(samples[k].get(v1, 0) * samples[k].get(v2, 0))

    for v in feature_vars:
        terms.append(f'sqrt(abs({v}))')
        for i in range(n):
            matrix[i].append(math.sqrt(abs(samples[i].get(v, 0))))

        terms.append(f'log(abs({v}))')
        for i in range(n):
            val = abs(samples[i].get(v, 0))
            matrix[i].append(math.log(val) if val > 1e-10 else 0)

        terms.append(f'1/{v}')
        for i in range(n):
            val = samples[i].get(v, 0)
            matrix[i].append(1.0 / val if abs(val) > 1e-10 else 0)

    return terms, matrix


def stlsq(A, b, threshold=0.05, max_iter=10):
    """STLSQ 稀疏回归"""
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    lam = 1e-6

    AtA = A.T @ A + lam * np.eye(A.shape[1])
    Atb = A.T @ b
    try:
        coeffs = np.linalg.solve(AtA, Atb)
    except:
        coeffs = np.zeros(A.shape[1])

    active = list(range(len(coeffs)))

    for _ in range(max_iter):
        new_active = [i for i in active if abs(coeffs[i]) > threshold]
        if not new_active or len(new_active) == len(active):
            break
        active = new_active

        sub_A = A[:, active]
        sub_AtA = sub_A.T @ sub_A + lam * np.eye(len(active))
        sub_Atb = sub_A.T @ b
        try:
            sub_coeffs = np.linalg.solve(sub_AtA, sub_Atb)
        except:
            break
        coeffs = np.zeros(len(coeffs))
        for idx, i in enumerate(active):
            coeffs[i] = sub_coeffs[idx]

    return coeffs


def run_sindy(samples, feature_vars, target_var, hypothesis_formula, max_steps=10):
    """SINDy 稀疏回归"""
    start_time = time.time()
    targets = np.array([s.get(target_var, 0) for s in samples], dtype=np.float64)

    target_mean = np.mean(targets)
    target_std = np.std(targets) or 1.0
    norm_targets = (targets - target_mean) / target_std

    terms, matrix = build_library(samples, feature_vars, 2)
    coeffs = stlsq(matrix, norm_targets.tolist(), 0.05, max_steps)

    real_coeffs = coeffs * target_std

    active_terms = []
    for i, (term, coeff) in enumerate(zip(terms, real_coeffs)):
        if abs(coeff) > 1e-4:
            active_terms.append((term, float(coeff)))

    if not active_terms:
        return SRResult(
            method='SINDy 稀疏回归',
            formula='0', rmse=1e10, mae=1e10, r2=0,
            complexity=1, iterations=max_steps,
            duration_ms=int((time.time() - start_time) * 1000),
            success=False, notes='稀疏回归未找到有效项',
        )

    formula_parts = []
    for term, coeff in active_terms:
        if term == '1':
            formula_parts.append(f'{coeff:.6f}')
        else:
            clean_term = term.replace('^', '**')
            if coeff >= 0:
                formula_parts.append(f'{coeff:.6f} * ({clean_term})')
            else:
                formula_parts.append(f'({coeff:.6f}) * ({clean_term})')

    formula = ' + '.join(formula_parts)

    # 用引擎兼容的公式
    engine_parts = []
    for term, coeff in active_terms:
        if term == '1':
            engine_parts.append(f'{coeff:.6f}')
        else:
            engine_term = term
            if '**' in engine_term:
                engine_term = engine_term.replace('**', '^')
            if coeff >= 0:
                engine_parts.append(f'{coeff:.6f} * {engine_term}')
            else:
                engine_parts.append(f'({coeff:.6f}) * {engine_term}')

    engine_formula = ' + '.join(engine_parts)

    try:
        preds = evaluate_formula_batch(engine_formula, samples)
        rmse = compute_rmse(preds, targets.tolist())
    except:
        rmse = 1e10
        preds = targets.tolist()

    duration = int((time.time() - start_time) * 1000)

    return SRResult(
        method='SINDy 稀疏回归',
        formula=engine_formula,
        rmse=rmse,
        mae=compute_mae(preds, targets.tolist()),
        r2=compute_r2(preds, targets.tolist()),
        complexity=formula_complexity(engine_formula),
        iterations=max_steps,
        duration_ms=duration,
        success=rmse < 1e9,
        notes=f'候选库{len(terms)}项, 选中{len(active_terms)}项',
    )


# ============================================================
# 方法 3: 模板匹配 + 参数优化
# ============================================================

def generate_templates(feature_vars):
    """生成物理/数学模板"""
    templates = []
    v = feature_vars

    if len(v) >= 1:
        x = v[0]
        templates.extend([
            f'a * {x} + b',
            f'a * {x}^2 + b * {x} + c',
            f'a * {x}^b',
            f'a * exp(b * {x})',
            f'a * log(abs({x})) + b',
            f'a * sqrt(abs({x})) + b',
            f'a * {x} / (b + {x})',
            f'a / (1 + exp(-b * {x})) + c',
            f'a / {x} + b',
            f'a * {x} / (b + abs({x}))',
            f'a * {x}^b + c',
            f'a * {x}^b / (1 + {x}^b) + c',
        ])

    if len(v) >= 2:
        x, y = v[0], v[1]
        templates.extend([
            f'a * {x} + b * {y} + c',
            f'a * {x} * {y} + b',
            f'a * {x} / {y} + b',
            f'a * {x}^b * {y}^c',
            f'a * {x}^2 + b * {y}^2 + c * {x} * {y} + d',
        ])

    if len(v) >= 3:
        x, y, z = v[0], v[1], v[2]
        templates.extend([
            f'a * {x} + b * {y} + c * {z} + d',
            f'a * {x} * {y} + b * {z} + c',
            f'a * {x}^b * {y}^c + d * {z}',
        ])

    if len(v) >= 4:
        x, y, z, w = v[0], v[1], v[2], v[3]
        templates.extend([
            f'a * {x} + b * {y} + c * {z} + d * {w} + e',
        ])

    return templates


def run_template_matching(samples, feature_vars, target_var, hypothesis_formula, max_steps=12):
    """模板匹配符号回归"""
    start_time = time.time()
    targets = [s.get(target_var, 0) for s in samples]

    templates = generate_templates(feature_vars)
    templates.insert(0, hypothesis_formula)

    # 添加渗流相关模板
    percolation_templates = [
        f'a * ({feature_vars[0]} - b)^c' if feature_vars else '',
        f'a * ({feature_vars[0]} - b)^c + d' if feature_vars else '',
    ]
    templates.extend([t for t in percolation_templates if t])

    max_templates = min(len(templates), max_steps)
    best_formula = hypothesis_formula
    best_rmse = float('inf')
    results = []

    for i in range(max_templates):
        template = templates[i]
        try:
            opt_formula, opt_rmse, _ = optimize_formula_params(
                template, samples, feature_vars, target_var, max_iter=120)
            results.append((template, opt_formula, opt_rmse))
            if opt_rmse < best_rmse:
                best_formula = opt_formula
                best_rmse = opt_rmse
        except:
            pass

    try:
        preds = evaluate_formula_batch(best_formula, samples)
    except:
        preds = targets

    duration = int((time.time() - start_time) * 1000)
    top = sorted(results, key=lambda x: x[2])[:3]
    notes = f'尝试{max_templates}个模板, 最佳: "{top[0][0][:40] if top else "..."}"'

    return SRResult(
        method='模板匹配 (Template Matching)',
        formula=best_formula,
        rmse=best_rmse,
        mae=compute_mae(preds, targets),
        r2=compute_r2(preds, targets),
        complexity=formula_complexity(best_formula),
        iterations=max_templates,
        duration_ms=duration,
        success=best_rmse < 1e9,
        notes=notes,
    )


# ============================================================
# 方法 4: 物理引导搜索 (Physics-Informed)
# ============================================================

def run_physics_informed(samples, feature_vars, target_var, hypothesis_formula, max_steps=10):
    """基于物理直觉的公式搜索"""
    start_time = time.time()
    targets = [s.get(target_var, 0) for s in samples]

    # 物理启发的候选公式族
    physics_candidates = []

    if len(feature_vars) >= 1:
        x = feature_vars[0]
        # 幂律标度 (渗流典型)
        for exp_val in [0.5, 1.0, 1.5, 2.0, 4/3, 0.41, 5/16]:
            physics_candidates.append(f'a * ({x} - b)^{exp_val:.4f}')
            physics_candidates.append(f'a * ({x} - b)^{exp_val:.4f} + c')
        # 指数衰减
        physics_candidates.append(f'a * exp(-b / {x}) + c')
        physics_candidates.append(f'a * exp(-b / ({x} - c))')
        # 对数标度
        physics_candidates.append(f'a * log({x} / b) + c')
        # 双曲型
        physics_candidates.append(f'a * {x}^2 / (b + {x}^2)')
        # 组合幂律+线性
        physics_candidates.append(f'a * {x} + b * {x}^c + d')

    if len(feature_vars) >= 2:
        x, y = feature_vars[0], feature_vars[1]
        physics_candidates.append(f'a * {x}^b * {y}^c + d')
        physics_candidates.append(f'a * ({x} - b)^c * {y}^d')
        physics_candidates.append(f'a * ({x} * {y})^b + c')

    # 清理空字符串
    physics_candidates = [c for c in physics_candidates if c]

    best_formula = hypothesis_formula
    best_rmse = float('inf')
    tested = 0

    for candidate in physics_candidates[:max_steps]:
        try:
            opt_formula, opt_rmse, params = optimize_formula_params(
                candidate, samples, feature_vars, target_var, max_iter=100)
            tested += 1
            if opt_rmse < best_rmse:
                best_formula = opt_formula
                best_rmse = opt_rmse
        except:
            pass

    try:
        preds = evaluate_formula_batch(best_formula, samples)
    except:
        preds = targets

    duration = int((time.time() - start_time) * 1000)

    return SRResult(
        method='物理引导搜索 (Physics-Informed)',
        formula=best_formula,
        rmse=best_rmse,
        mae=compute_mae(preds, targets),
        r2=compute_r2(preds, targets),
        complexity=formula_complexity(best_formula),
        iterations=tested,
        duration_ms=duration,
        success=best_rmse < 1e9,
        notes=f'测试{tested}个物理启发公式',
    )
