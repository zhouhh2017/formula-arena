from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
增强版迭代符号回归 - 以当前幂律为种子, 多轮进化
================================================
策略:
1. 以 p_c = 3.96·z^(-0.90)·d^(-0.83) 为种子
2. 多轮迭代: 变异 + 交叉 + 参数优化
3. 增加物理启发的候选模板
4. 每轮选最好的继续进化
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import random
import copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize
from large_scale_benchmark import build_full_dataset

random.seed(42)
np.random.seed(42)


# ===== 数据准备 =====
def prepare_data(df):
    samples = []
    for _, row in df.iterrows():
        samples.append({
            'z': row['z'],
            'd': row['d'],
            'z_inv': row['z_inv'],
            'd_inv': row['d_inv'],
            'log_z': row['log_z'],
            'log_d': row['log_d'],
            'zd': row['zd'],
            'pc': row['pc'],
        })
    return samples


# ===== 公式评估 =====
def eval_formula(formula, samples):
    """安全评估公式"""
    import re

    # 将公式转换为 Python 可执行形式
    py_formula = formula
    py_formula = py_formula.replace('^', '**')

    results = []
    for s in samples:
        try:
            local_vars = dict(s)
            # 添加安全数学函数
            local_vars['log'] = lambda x: math.log(abs(x)) if abs(x) > 1e-15 else -30
            local_vars['sqrt'] = lambda x: math.sqrt(abs(x))
            local_vars['exp'] = lambda x: math.exp(max(-500, min(500, x)))
            local_vars['sin'] = math.sin
            local_vars['cos'] = math.cos
            local_vars['abs'] = abs
            local_vars['pi'] = math.pi
            local_vars['e'] = math.e

            val = eval(py_formula, {"__builtins__": {}}, local_vars)
            if math.isnan(val) or math.isinf(val) or abs(val) > 1e10:
                results.append(None)
            else:
                results.append(val)
        except:
            results.append(None)

    return results


def compute_metrics(preds, targets):
    """计算 R², RMSE, MAE"""
    valid_pairs = [(p, t) for p, t in zip(preds, targets) if p is not None]
    if len(valid_pairs) < len(targets) * 0.5:
        return -1e10, 1e10, 1e10

    preds_v = np.array([p for p, t in valid_pairs])
    targets_v = np.array([t for p, t in valid_pairs])

    ss_res = np.sum((preds_v - targets_v) ** 2)
    ss_tot = np.sum((targets_v - np.mean(targets_v)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((preds_v - targets_v) ** 2))
    mae = np.mean(np.abs(preds_v - targets_v))

    return r2, rmse, mae


# ===== 优化公式参数 =====
def optimize_params(formula, samples, param_names, max_restarts=30):
    """用 scipy 优化公式中的参数"""
    targets = np.array([s['pc'] for s in samples])

    def objective(params):
        param_map = {name: params[i] for i, name in enumerate(param_names)}
        modified = formula
        # 替换参数
        for name in sorted(param_names, key=len, reverse=True):
            val = param_map[name]
            modified = modified.replace(name, f'({val:.10f})')

        preds = eval_formula(modified, samples)
        valid = [(p, t) for p, t in zip(preds, targets) if p is not None]
        if len(valid) < len(targets) * 0.5:
            return 1e10
        p_arr = np.array([p for p, t in valid])
        t_arr = np.array([t for p, t in valid])
        return np.sqrt(np.mean((p_arr - t_arr) ** 2))

    best_rmse = float('inf')
    best_params = None

    for _ in range(max_restarts):
        x0 = np.random.randn(len(param_names)) * 2
        try:
            result = scipy_minimize(objective, x0, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_rmse:
                best_rmse = result.fun
                best_params = result.x.copy()
        except:
            pass

    if best_params is not None:
        optimized = formula
        for i, name in enumerate(sorted(param_names, key=len, reverse=True)):
            val = best_params[i]
            optimized = optimized.replace(name, f'({val:.6f})')
        return optimized, best_rmse

    return formula, 1e10


# ===== 候选公式库 =====
def generate_candidate_formulas():
    """生成大量候选公式"""
    candidates = []

    # === 第1类: 改进幂律 ===
    # 基础幂律
    candidates.append(("a * z^b * d^c", ['a', 'b', 'c']))
    candidates.append(("a * z^b * d^c + e", ['a', 'b', 'c', 'e']))

    # 带修正项的幂律
    candidates.append(("a * z^b * d^c * (1 + e / z)", ['a', 'b', 'c', 'e']))
    candidates.append(("a * z^b * d^c * (1 + e / d)", ['a', 'b', 'c', 'e']))
    candidates.append(("a * z^b * d^c * (1 + e / (z*d))", ['a', 'b', 'c', 'e']))
    candidates.append(("a * z^b * d^c + e / (z * d)", ['a', 'b', 'c', 'e']))

    # 分维度幂律
    candidates.append(("a * z^b / (1 + z^c * d^e)", ['a', 'b', 'c', 'e']))
    candidates.append(("a / (z^b * d^c + e)", ['a', 'b', 'c', 'e']))

    # === 第2类: 对数/指数混合 ===
    candidates.append(("a * exp(b * log(z) + c * log(d))", ['a', 'b', 'c']))
    candidates.append(("a * exp(-b * z) * exp(-c * d) + e", ['a', 'b', 'c', 'e']))
    candidates.append(("a / (b * log(z) + c * log(d) + e)", ['a', 'b', 'c', 'e']))
    candidates.append(("a / (b * z^0.5 + c * d^0.5 + e)", ['a', 'b', 'c', 'e']))

    # === 第3类: 有理函数 ===
    candidates.append(("(a + b * z) / (c + d * z + e * d)", ['a', 'b', 'c', 'd', 'e']))
    candidates.append(("a / (b + c * z + d * d) ", ['a', 'b', 'c', 'd']))
    candidates.append(("a * z / (b + z) * d / (c + d)", ['a', 'b', 'c']))
    candidates.append(("a / (b + z^c) * d / (e + d)", ['a', 'b', 'c', 'd', 'e']))

    # === 第4类: 分段/条件型 (用平滑近似) ===
    # p_c ~ A / (z-1) 的改进
    candidates.append(("a / (z - b) * (c / d)^e", ['a', 'b', 'c', 'd', 'e']))
    candidates.append(("a / (z - b + c * d)", ['a', 'b', 'c']))
    candidates.append(("a / (z^b - c + d * d)", ['a', 'b', 'c', 'd']))

    # === 第5类: Bethe 近似改进 ===
    candidates.append(("a / (z - b) * d^c", ['a', 'b', 'c']))
    candidates.append(("a / (z - b) * d^c + e", ['a', 'b', 'c', 'e']))
    candidates.append(("a / (z - b) * (1 + c / d)", ['a', 'b', 'c']))
    candidates.append(("a / (z - b)^c * d^e", ['a', 'b', 'c', 'e']))

    # === 第6类: 渗流理论启发 ===
    # p_c 与维度关系接近 1/d
    candidates.append(("a / (z * d + b)", ['a', 'b']))
    candidates.append(("a / (z * d^b + c)", ['a', 'b', 'c']))
    candidates.append(("a * (z * d)^b + c", ['a', 'b', 'c']))
    candidates.append(("a * (z^b + c * d^e)", ['a', 'b', 'c', 'e']))

    # === 第7类: 混合特征 ===
    candidates.append(("a * z_inv^b * d_inv^c", ['a', 'b', 'c']))
    candidates.append(("a * z_inv^b * d_inv^c + e", ['a', 'b', 'c', 'e']))
    candidates.append(("a / (b * z^c + d * d^e)", ['a', 'b', 'c', 'd', 'e']))

    # === 第8类: 更复杂的非线性 ===
    candidates.append(("a * z^b * d^c * exp(e * z_inv)", ['a', 'b', 'c', 'e']))
    candidates.append(("a * z^b * d^c * exp(e * d_inv)", ['a', 'b', 'c', 'e']))
    candidates.append(("a * z^b * d^c * (1 + e * log(z))", ['a', 'b', 'c', 'e']))
    candidates.append(("a * z^b * d^c * (1 + e * log(d))", ['a', 'b', 'c', 'e']))

    return candidates


# ===== 迭代搜索 =====
def iterative_search(samples, n_rounds=5, top_k=5):
    """多轮迭代搜索"""
    targets = np.array([s['pc'] for s in samples])
    candidates = generate_candidate_formulas()

    print(f"\n生成了 {len(candidates)} 个候选公式模板")
    print(f"开始 {n_rounds} 轮迭代搜索...\n")

    all_results = []

    # === 第1轮: 遍历所有候选 ===
    print("=" * 80)
    print("第 1 轮: 遍历候选公式 + 参数优化")
    print("=" * 80)

    round_results = []
    for i, (formula, params) in enumerate(candidates):
        try:
            optimized, rmse = optimize_params(formula, samples, params, max_restarts=15)
            if rmse < 1e9:
                preds = eval_formula(optimized, samples)
                r2, rmse_val, mae = compute_metrics(preds, targets.tolist())
                if r2 > -10:
                    round_results.append({
                        'formula': optimized,
                        'template': formula,
                        'params': params,
                        'r2': r2,
                        'rmse': rmse_val,
                        'mae': mae,
                        'round': 1,
                    })
        except:
            pass

        if (i + 1) % 10 == 0:
            print(f"  已测试 {i+1}/{len(candidates)} 个...")

    round_results.sort(key=lambda x: x['mae'])
    print(f"\n  找到 {len(round_results)} 个有效公式")
    if round_results:
        best = round_results[0]
        print(f"  最佳: MAE={best['mae']:.4f}, R²={best['r2']:.4f}")
        print(f"  公式: {best['formula'][:80]}")

    all_results.extend(round_results[:top_k])

    # === 后续轮次: 基于 top_k 变异 ===
    for round_num in range(2, n_rounds + 1):
        print(f"\n{'=' * 80}")
        print(f"第 {round_num} 轮: 基于 Top-{top_k} 变异")
        print(f"{'=' * 80}")

        # 取上一轮最好的 top_k
        top_formulas = sorted(all_results, key=lambda x: x['mae'])[:top_k]

        round_results = []
        for parent in top_formulas:
            # 生成变异体
            mutants = mutate_formula(parent['template'], parent['params'])
            for formula, params in mutants:
                try:
                    optimized, rmse = optimize_params(formula, samples, params, max_restarts=10)
                    if rmse < 1e9:
                        preds = eval_formula(optimized, samples)
                        r2, rmse_val, mae = compute_metrics(preds, targets.tolist())
                        if r2 > -10:
                            round_results.append({
                                'formula': optimized,
                                'template': formula,
                                'params': params,
                                'r2': r2,
                                'rmse': rmse_val,
                                'mae': mae,
                                'round': round_num,
                            })
                except:
                    pass

        round_results.sort(key=lambda x: x['mae'])
        print(f"  找到 {len(round_results)} 个变异体")
        if round_results:
            best = round_results[0]
            print(f"  最佳: MAE={best['mae']:.4f}, R²={best['r2']:.4f}")
            print(f"  公式: {best['formula'][:80]}")

        all_results.extend(round_results[:top_k])

    return all_results


def mutate_formula(template, params):
    """基于模板生成变异体"""
    mutants = []

    # 变异1: 添加修正项
    for correction in [' + e / z', ' + e / d', ' + e / (z*d)',
                        ' * (1 + e / z)', ' * (1 + e / d)',
                        ' + e * z_inv', ' + e * d_inv',
                        ' * exp(e * z_inv)', ' * exp(e * d_inv)']:
        new_template = template + correction
        new_params = params + ['e']
        mutants.append((new_template, new_params))

    # 变异2: 改变指数
    for p in params:
        if p in ['b', 'c']:
            # 试试固定为物理启发值
            for val in [-1, -0.5, -2/3, -1.5, 0.5]:
                new_template = template.replace(p, f'{val:.4f}')
                new_params = [x for x in params if x != p]
                mutants.append((new_template, new_params))

    # 变异3: 替换函数
    replacements = [
        ('z^b', 'exp(b * log(z))'),
        ('d^c', 'exp(c * log(d))'),
        ('z^b', 'sqrt(z)^b2'),  # 需要额外参数
        ('a * z^b * d^c', 'a * (z*d)^b * (z/d)^c'),
    ]
    for old, new in replacements:
        if old in template:
            new_template = template.replace(old, new)
            mutants.append((new_template, params))

    return mutants


def main():
    print("=" * 80)
    print("🔬 增强版迭代符号回归 (52个真实数据点)")
    print("=" * 80)

    # 构建数据集
    df = build_full_dataset()
    samples = prepare_data(df)
    targets = [s['pc'] for s in samples]

    print(f"\n数据: {len(samples)} 个晶格")
    print(f"p_c 范围: {min(targets):.5f} - {max(targets):.5f}")

    # 运行迭代搜索
    all_results = iterative_search(samples, n_rounds=3, top_k=5)

    # 排序
    all_results.sort(key=lambda x: x['mae'])

    # ============================================================
    # 最终对比
    # ============================================================
    print("\n" + "=" * 80)
    print("🏆 最终 Top-10 公式")
    print("=" * 80)

    for i, r in enumerate(all_results[:10]):
        print(f"\n#{i+1} (Round {r['round']})")
        print(f"  R² = {r['r2']:.6f}")
        print(f"  RMSE = {r['rmse']:.6f}")
        print(f"  MAE = {r['mae']:.6f}")
        print(f"  模板: {r['template']}")
        print(f"  公式: {r['formula'][:100]}")

    # 对比基准
    print("\n" + "=" * 80)
    print("📊 对比基准")
    print("=" * 80)

    # 当前幂律
    from large_scale_benchmark import test_power_law, test_ml_methods
    params, r2_power, rmse_power = test_power_law(df)
    ml_results = test_ml_methods(df)
    best_ml_mae = min(r['mae'] for r in ml_results.values())

    print(f"\n  当前幂律:      MAE = {rmse_power:.4f}, R² = {r2_power:.4f}")
    print(f"  最佳 ML:       MAE = {best_ml_mae:.4f}")

    if all_results:
        best_sr = all_results[0]
        print(f"  扫地僧(增强):  MAE = {best_sr['mae']:.4f}, R² = {best_sr['r2']:.4f}")
        print(f"\n  最佳公式: {best_sr['formula'][:100]}")

        improvement = (rmse_power - best_sr['mae']) / rmse_power * 100
        if best_sr['mae'] < rmse_power:
            print(f"\n  ✓ 扫地僧找到更好的公式! MAE 提升: {improvement:.1f}%")
        else:
            print(f"\n  ⚠️  当前幂律仍然是最好的 (差距: {abs(improvement):.1f}%)")

    # 保存
    output_file = str(ROOT / 'enhanced_sr_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("增强版迭代符号回归结果\n")
        f.write("=" * 80 + "\n\n")

        f.write("Top-10 公式:\n")
        for i, r in enumerate(all_results[:10]):
            f.write(f"\n#{i+1}:\n")
            f.write(f"  R² = {r['r2']:.6f}\n")
            f.write(f"  RMSE = {r['rmse']:.6f}\n")
            f.write(f"  MAE = {r['mae']:.6f}\n")
            f.write(f"  模板: {r['template']}\n")
            f.write(f"  公式: {r['formula']}\n")

    print(f"\n💾 结果已保存: {output_file}")


if __name__ == '__main__':
    main()
