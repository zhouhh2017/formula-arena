#!/usr/bin/env python3
"""
修复泛化问题 — 分组验证失败分析 + 改进方案
============================================
核心发现:
1. Girth-Bethe 在标准晶格上泛化失败 (MAE 爆炸)
2. 原因: 参数范围不同 + 公式形式可能不对
3. 改进: 分段建模 + 物理约束 + 贝叶斯先验
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize, differential_evolution
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr


def load_data():
    """加载并准备数据"""
    from physics_52_validation import build_full_physics_dataset
    df = build_full_physics_dataset()

    # 改进复杂邻域的物理量
    zone_structure = {
        1: {'z': 4, 'n_tri': 0, 'n_sq': 1, 'g': 4},
        2: {'z': 4, 'n_tri': 2, 'n_sq': 2, 'g': 3},
        3: {'z': 4, 'n_tri': 0, 'n_sq': 2, 'g': 4},
        4: {'z': 4, 'n_tri': 4, 'n_sq': 4, 'g': 3},
        5: {'z': 4, 'n_tri': 2, 'n_sq': 3, 'g': 3},
        6: {'z': 4, 'n_tri': 6, 'n_sq': 6, 'g': 3},
    }

    complex_mask = df['name'].str.startswith('sq-', na=False)
    for idx in df[complex_mask].index:
        name = df.loc[idx, 'name']
        zones_str = name.replace('sq-', '')
        zones = [int(x) for x in zones_str.split(',')]

        total_tri, total_sq, total_z = 0, 0, 0
        for zone in zones:
            if zone in zone_structure:
                s = zone_structure[zone]
                total_z += s['z']
                total_tri += s['n_tri']
                total_sq += s['n_sq']

        if total_z > 0:
            df.loc[idx, 'n_tri'] = total_tri
            df.loc[idx, 'n_sq'] = total_sq
            df.loc[idx, 'g'] = 3 if total_tri > 0 else 4
            df.loc[idx, 'C'] = 2 * total_tri / (total_z * (total_z - 1)) if total_z > 1 else 0

    df['z_eff'] = df['z'] - 2 * df['n_tri'] / df['z']

    # 分组
    df['group'] = 'standard'
    df.loc[df['name'].str.startswith('arch_'), 'group'] = 'archimedean'
    df.loc[df['name'].str.startswith('sq-'), 'group'] = 'complex'

    return df


def diagnose_generalization_failure(df):
    """
    诊断为什么分组验证失败

    核心问题: 不同类型晶格的 p_c 由不同物理机制控制?
    """

    print("=" * 80)
    print("🔍 诊断: 为什么泛化失败?")
    print("=" * 80)

    # 1. 各组数据统计
    print("\n--- 各组数据分布 ---")
    for group in ['standard', 'archimedean', 'complex']:
        subset = df[df['group'] == group]
        print(f"\n{group} (n={len(subset)}):")
        print(f"  z:  {subset['z'].min()} - {subset['z'].max()} (mean={subset['z'].mean():.1f})")
        print(f"  d:  {subset['d'].unique()}")
        print(f"  g:  {subset['g'].value_counts().to_dict()}")
        print(f"  C:  {subset['C'].min():.3f} - {subset['C'].max():.3f}")
        print(f"  p_c: {subset['pc'].min():.4f} - {subset['pc'].max():.4f}")

    # 2. 各组 p_c vs z 的关系
    print("\n--- p_c vs z 的关系 (各组对比) ---")
    for group in ['standard', 'archimedean', 'complex']:
        subset = df[df['group'] == group]
        if len(subset) >= 3:
            corr, p = spearmanr(subset['z'], subset['pc'])
            print(f"  {group}: ρ={corr:.3f}, p={p:.4f}")

    # 3. 关键问题: 同 z 不同组的 p_c 差异
    print("\n--- 同 z 不同组的 p_c ---")
    for z_val in [3, 4, 5, 6]:
        subset = df[df['z'] == z_val]
        if len(subset) >= 2:
            print(f"\nz = {z_val}:")
            for _, row in subset.iterrows():
                print(f"  {row['group']:>12} | {row['name']:<20} | g={row['g']} C={row['C']:.3f} | p_c={row['pc']:.4f}")

    # 4. 分析: Archimedean 为什么不同?
    print("\n--- Archimedean 的特殊性 ---")
    arch = df[df['group'] == 'archimedean']
    std_z3 = df[(df['group'] == 'standard') & (df['z'] == 3)]

    print(f"\n标准 z=3: p_c = {std_z3['pc'].values}")
    print(f"Archimedean z=3: p_c = {arch[arch['z']==3]['pc'].values}")
    print("\n→ 同 z=3, Archimedean 的 p_c 显著更高!")
    print("→ 说明 (z, d, g, C) 不能完全解释 p_c 的差异")
    print("→ 可能缺少关键物理量: 面类型分布, 对偶性, 对称性")


def test_alternative_formulas(df):
    """
    测试替代公式形式, 寻找更好的泛化

    策略:
    1. 增加更多物理特征
    2. 测试不同的函数形式
    3. 用贝叶斯约束防止参数爆炸
    """

    print("\n" + "=" * 80)
    print("🧪 测试替代公式 (改进泛化)")
    print("=" * 80)

    z = df['z'].values.astype(float)
    d = df['d'].values.astype(float)
    pc = df['pc'].values.astype(float)
    g = df['g'].values.astype(float)
    C = df['C'].values.astype(float)
    n_tri = df['n_tri'].values.astype(float)

    formulas = {}

    # === 公式 1: 基础幂律 (基准) ===
    def f_power(p, z, d):
        return p[0] * np.power(z, -p[1]) * np.power(d, -p[2])

    r = robust_fit(f_power, [z, d], pc, "基础幂律",
                   bounds=[(0.1, 10), (0, 2), (0, 2)], n_restarts=100)
    if r:
        formulas['基础幂律'] = r

    # === 公式 2: Girth-Bethe (约束参数) ===
    def f_gb(p, z, g, d):
        return p[0] / (np.power(np.maximum(z-1, 0.5), p[1]) *
                       np.power(np.maximum(g, 1), p[2]) *
                       np.power(d, p[3]))

    # 用物理先验约束: α≈1, β≈0, γ≈1
    r = robust_fit(f_gb, [z, g, d], pc, "Girth-Bethe (约束)",
                   bounds=[(1, 5), (0.3, 1.5), (-0.5, 0.5), (0.3, 1.5)],
                   n_restarts=100,
                   prior_center=[3.0, 1.0, 0.0, 1.0],
                   prior_strength=0.5)
    if r:
        formulas['Girth-Bethe (约束)'] = r

    # === 公式 3: 有效配位数 + 维度 ===
    def f_zeff(p, z_eff, d):
        return p[0] / (np.power(np.maximum(z_eff-1, 0.5), p[1]) *
                       np.power(d, p[2]))

    z_eff = (z - 2 * n_tri / np.maximum(z, 1))
    r = robust_fit(f_zeff, [z_eff, d], pc, "有效配位数+维度",
                   bounds=[(0.1, 10), (0.3, 1.5), (0.3, 1.5)], n_restarts=100)
    if r:
        formulas['有效配位数+维度'] = r

    # === 公式 4: 综合 (z, d, g, C) ===
    def f_full(p, z, d, g, C):
        base = p[0] / (np.power(z, p[1]) * np.power(d, p[2]))
        correction = 1 + p[3] * C + p[4] / np.maximum(g, 1)
        return base * correction

    r = robust_fit(f_full, [z, d, g, C], pc, "综合 (z,d,g,C)",
                   bounds=[(0.1, 15), (0.3, 2), (0.3, 2), (-2, 2), (-2, 2)],
                   n_restarts=80)
    if r:
        formulas['综合 (z,d,g,C)'] = r

    # === 公式 5: 对数线性 ===
    log_z = np.log(np.maximum(z, 1))
    log_d = np.log(np.maximum(d, 1))
    log_g = np.log(np.maximum(g, 1))

    def f_loglin(p, lz, ld, lg):
        return np.exp(p[0] + p[1]*lz + p[2]*ld + p[3]*lg)

    r = robust_fit(f_loglin, [log_z, log_d, log_g], pc, "对数线性 (log z, log d, log g)",
                   bounds=[(-5, 5), (-3, 0), (-3, 0), (-1, 1)], n_restarts=80)
    if r:
        formulas['对数线性'] = r

    # === 公式 6: 分段 (2D vs 3D+) ===
    is_2d = (d == 2).astype(float)
    def f_piecewise(p, z, d, is_2d):
        base = p[0] / (np.power(z, p[1]) * np.power(d, p[2]))
        shift = p[3] * is_2d
        return base + shift

    r = robust_fit(f_piecewise, [z, d, is_2d], pc, "分段 (2D+偏移)",
                   bounds=[(0.1, 10), (0.3, 2), (0.3, 2), (-0.3, 0.3)], n_restarts=80)
    if r:
        formulas['分段 (2D+偏移)'] = r

    return formulas


def robust_fit(func, inputs, target, name, bounds, n_restarts=50,
               prior_center=None, prior_strength=0.0):
    """
    稳健拟合 — 带先验约束

    如果 prior_center 不为 None, 添加正则化:
    loss = RMSE + prior_strength * ||params - prior_center||^2
    """

    best_rmse = float('inf')
    best_params = None

    for _ in range(n_restarts):
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])

        def objective(p):
            try:
                pred = func(p, *inputs)
                valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
                if valid.sum() < len(target) * 0.8:
                    return 1e10
                rmse = np.sqrt(np.mean((pred[valid] - target[valid])**2))

                # 添加先验约束
                if prior_center is not None and prior_strength > 0:
                    penalty = prior_strength * np.sum((p - prior_center)**2)
                    rmse += penalty

                return rmse
            except:
                return 1e10

        try:
            result = scipy_minimize(objective, x0, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'fatol': 1e-14})
            if result.fun < best_rmse:
                pred = func(result.x, *inputs)
                valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
                if valid.sum() > len(target) * 0.8:
                    best_rmse = result.fun
                    best_params = result.x.copy()
        except:
            pass

    # 差分进化
    try:
        def de_objective(p):
            try:
                pred = func(p, *inputs)
                valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
                if valid.sum() < len(target) * 0.8:
                    return 1e10
                rmse = np.sqrt(np.mean((pred[valid] - target[valid])**2))
                if prior_center is not None and prior_strength > 0:
                    penalty = prior_strength * np.sum((p - prior_center)**2)
                    rmse += penalty
                return rmse
            except:
                return 1e10

        result_de = differential_evolution(de_objective, bounds,
                                           maxiter=3000, tol=1e-14, seed=42)
        if result_de.fun < best_rmse:
            pred = func(result_de.x, *inputs)
            valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
            if valid.sum() > len(target) * 0.8:
                best_rmse = result_de.fun
                best_params = result_de.x.copy()
    except:
        pass

    if best_params is not None:
        pred = func(best_params, *inputs)
        ss_res = np.sum((pred - target)**2)
        ss_tot = np.sum((target - target.mean())**2)
        r2 = 1 - ss_res / ss_tot
        mae = np.mean(np.abs(pred - target))

        return {
            'name': name, 'params': best_params, 'func': func,
            'r2': r2, 'mae': mae, 'rmse': best_rmse,
            'n_params': len(best_params)
        }
    return None


def group_cv_robust(df, formulas):
    """
    稳健的分组交叉验证

    改进:
    1. 约束参数范围 (用物理先验)
    2. 检测异常参数 (如果超出范围, 标记失败)
    3. 报告每组的最佳公式
    """

    print("\n" + "=" * 80)
    print("🔄 稳健的分组交叉验证")
    print("=" * 80)

    groups = ['standard', 'archimedean', 'complex']
    group_indices = {g: df[df['group'] == g].index.tolist() for g in groups}

    z = df['z'].values.astype(float)
    d = df['d'].values.astype(float)
    g_arr = df['g'].values.astype(float)
    C = df['C'].values.astype(float)
    pc = df['pc'].values.astype(float)
    n_tri = df['n_tri'].values.astype(float)
    z_eff = z - 2 * n_tri / np.maximum(z, 1)
    log_z = np.log(np.maximum(z, 1))
    log_d = np.log(np.maximum(d, 1))
    log_g = np.log(np.maximum(g_arr, 1))
    is_2d = (d == 2).astype(float)

    # 准备各公式的输入
    formula_inputs = {
        '基础幂律': [z, d],
        'Girth-Bethe (约束)': [z, g_arr, d],
        '有效配位数+维度': [z_eff, d],
        '综合 (z,d,g,C)': [z, d, g_arr, C],
        '对数线性': [log_z, log_d, log_g],
        '分段 (2D+偏移)': [z, d, is_2d],
    }

    # 参数范围约束 (物理先验)
    param_bounds = {
        '基础幂律': [(0.1, 10), (0, 2), (0, 2)],
        'Girth-Bethe (约束)': [(1, 5), (0.3, 1.5), (-0.5, 0.5), (0.3, 1.5)],
        '有效配位数+维度': [(0.1, 10), (0.3, 1.5), (0.3, 1.5)],
        '综合 (z,d,g,C)': [(0.1, 15), (0.3, 2), (0.3, 2), (-2, 2), (-2, 2)],
        '对数线性': [(-5, 5), (-3, 0), (-3, 0), (-1, 1)],
        '分段 (2D+偏移)': [(0.1, 10), (0.3, 2), (0.3, 2), (-0.3, 0.3)],
    }

    # 先验中心
    prior_centers = {
        '基础幂律': [4.0, 0.9, 0.8],
        'Girth-Bethe (约束)': [3.0, 1.0, 0.0, 1.0],
        '有效配位数+维度': [2.0, 1.0, 1.0],
        '综合 (z,d,g,C)': [4.0, 0.9, 0.8, 0.0, 0.0],
        '对数线性': [1.5, -0.9, -0.8, 0.0],
        '分段 (2D+偏移)': [4.0, 0.9, 0.8, 0.0],
    }

    all_cv_results = {}

    for test_group in groups:
        train_mask = df['group'] != test_group
        test_mask = df['group'] == test_group

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        print(f"\n--- 测试集: {test_group} (n={len(test_idx)}) ---")
        print(f"{'公式':<25} {'训练MAE':<10} {'测试MAE':<10} {'状态':<10}")
        print("-" * 60)

        cv_results = {}

        for fname, fdata in formulas.items():
            func = fdata['func']
            inputs = formula_inputs[fname]
            bounds = param_bounds[fname]
            prior = prior_centers[fname]

            # 在训练集上拟合 (带先验约束)
            train_inputs = [x[train_idx] for x in inputs]
            train_target = pc[train_idx]

            best_rmse = float('inf')
            best_p = None

            for _ in range(60):
                x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])

                def objective(p):
                    try:
                        pred = func(p, *train_inputs)
                        valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
                        if valid.sum() < len(train_target) * 0.8:
                            return 1e10
                        rmse = np.sqrt(np.mean((pred[valid] - train_target[valid])**2))
                        penalty = 0.3 * np.sum((p - prior)**2)
                        return rmse + penalty
                    except:
                        return 1e10

                try:
                    result = scipy_minimize(objective, x0, method='Nelder-Mead',
                                           options={'maxiter': 3000, 'fatol': 1e-12})
                    if result.fun < best_rmse:
                        pred = func(result.x, *train_inputs)
                        valid = ~np.isnan(pred) & ~np.isinf(pred)
                        if valid.sum() > len(train_target) * 0.8:
                            best_rmse = result.fun
                            best_p = result.x.copy()
                except:
                    pass

            if best_p is not None:
                train_pred = func(best_p, *train_inputs)
                train_mae = np.mean(np.abs(train_pred - train_target))

                test_inputs = [x[test_idx] for x in inputs]
                test_target = pc[test_idx]
                test_pred = func(best_p, *test_inputs)

                if np.all(~np.isnan(test_pred)) and np.all(~np.isinf(test_pred)) and np.all(np.abs(test_pred) < 100):
                    test_mae = np.mean(np.abs(test_pred - test_target))
                    status = "✓" if test_mae < 0.1 else "⚠️"
                    cv_results[fname] = {'train_mae': train_mae, 'test_mae': test_mae, 'params': best_p}
                    print(f"  {fname:<23} {train_mae:<10.4f} {test_mae:<10.4f} {status}")
                else:
                    print(f"  {fname:<23} {train_mae:<10.4f} {'NaN':<10} ✗ 预测异常")

        all_cv_results[test_group] = cv_results

    # 汇总
    print("\n--- 汇总: 各测试集的最佳公式 ---")
    for test_group in groups:
        if all_cv_results[test_group]:
            best = min(all_cv_results[test_group].items(), key=lambda x: x[1]['test_mae'])
            print(f"  {test_group}: {best[0]} (MAE={best[1]['test_mae']:.4f})")


def final_recommendation(df):
    """
    最终推荐: 综合 LOOCV + 分组验证 + 物理解释
    """

    print("\n" + "=" * 80)
    print("🎯 最终推荐")
    print("=" * 80)

    z = df['z'].values.astype(float)
    d = df['d'].values.astype(float)
    g = df['g'].values.astype(float)
    C = df['C'].values.astype(float)
    pc = df['pc'].values.astype(float)
    n_tri = df['n_tri'].values.astype(float)
    z_eff = z - 2 * n_tri / np.maximum(z, 1)

    # 最佳公式候选
    formulas = {}

    # 1. 基础幂律 (最简洁)
    def f1(p, z, d):
        return p[0] * np.power(z, -p[1]) * np.power(d, -p[2])
    r = robust_fit(f1, [z, d], pc, "基础幂律",
                   [(0.1, 10), (0, 2), (0, 2)], n_restarts=100)
    if r:
        formulas['基础幂律'] = {**r, 'inputs': [z, d], 'bounds': [(0.1, 10), (0, 2), (0, 2)]}

    # 2. 有效配位数 + 维度 (物理)
    def f2(p, z_eff, d):
        return p[0] / (np.power(np.maximum(z_eff-1, 0.5), p[1]) * np.power(d, p[2]))
    r = robust_fit(f2, [z_eff, d], pc, "有效配位数+维度",
                   [(0.1, 10), (0.3, 1.5), (0.3, 1.5)], n_restarts=100)
    if r:
        formulas['有效配位数+维度'] = {**r, 'inputs': [z_eff, d], 'bounds': [(0.1, 10), (0.3, 1.5), (0.3, 1.5)]}

    # 3. 综合公式
    def f3(p, z, d, g, C):
        base = p[0] / (np.power(z, p[1]) * np.power(d, p[2]))
        correction = 1 + p[3] * C + p[4] / np.maximum(g, 1)
        return base * correction
    r = robust_fit(f3, [z, d, g, C], pc, "综合 (z,d,g,C)",
                   [(0.1, 15), (0.3, 2), (0.3, 2), (-2, 2), (-2, 2)], n_restarts=80)
    if r:
        formulas['综合'] = {**r, 'inputs': [z, d, g, C], 'bounds': [(0.1, 15), (0.3, 2), (0.3, 2), (-2, 2), (-2, 2)]}

    # LOOCV
    print("\n--- LOOCV 验证 ---")
    for fname, fdata in formulas.items():
        func = fdata['func']
        inputs = fdata['inputs']
        n_p = fdata['n_params']

        loo = LeaveOneOut()
        errors = []
        for train_idx, test_idx in loo.split(z):
            inputs_tr = [x[train_idx] for x in inputs]
            target_tr = pc[train_idx]

            best_rmse = float('inf')
            best_p = None
            for _ in range(20):
                x0 = np.random.randn(n_p) * 2
                try:
                    result = scipy_minimize(
                        lambda p: np.sqrt(np.mean((func(p, *inputs_tr) - target_tr)**2)),
                        x0, method='Nelder-Mead',
                        options={'maxiter': 2000, 'fatol': 1e-12}
                    )
                    if result.fun < best_rmse:
                        pred = func(result.x, *inputs_tr)
                        if np.all(~np.isnan(pred)) and np.all(~np.isinf(pred)):
                            best_rmse = result.fun
                            best_p = result.x.copy()
                except:
                    pass

            if best_p is not None:
                test_inputs = [x[test_idx] for x in inputs]
                pred = func(best_p, *test_inputs)
                if not (np.isnan(pred[0]) or np.isinf(pred[0])):
                    errors.append(abs(pred[0] - pc[test_idx[0]]))

        loocv_mae = np.mean(errors) if errors else 1e10
        fdata['loocv_mae'] = loocv_mae
        print(f"  {fname}: 训练MAE={fdata['mae']:.4f}, LOOCV MAE={loocv_mae:.4f}")

    # 排序
    ranked = sorted(formulas.items(), key=lambda x: x[1]['loocv_mae'])

    print("\n--- 最终排名 ---")
    for i, (name, fdata) in enumerate(ranked):
        print(f"  #{i+1}: {name}")
        print(f"       训练MAE={fdata['mae']:.4f}, LOOCV={fdata['loocv_mae']:.4f}, "
              f"R²={fdata['r2']:.4f}, params={fdata['n_params']}")
        # 生成公式字符串
        p = fdata['params']
        if name == '基础幂律':
            formula_str = f"{p[0]:.3f} / (z^{p[1]:.3f} · d^{p[2]:.3f})"
        elif name == '有效配位数+维度':
            formula_str = f"{p[0]:.3f} / ((z_eff-1)^{p[1]:.3f} · d^{p[2]:.3f})"
        elif name == '综合':
            formula_str = f"{p[0]:.3f}/(z^{p[1]:.3f}·d^{p[2]:.3f}) · (1+{p[3]:.3f}·C+{p[4]:.3f}/g)"
        else:
            formula_str = "..."
        print(f"       公式: {formula_str}")

    # 推荐
    best_name, best_data = ranked[0]
    print(f"\n🏆 推荐公式: {best_name}")
    if best_name == '基础幂律':
        p = best_data['params']
        print(f"   p_c = {p[0]:.3f} / (z^{p[1]:.3f} · d^{p[2]:.3f})")
    elif best_name == '有效配位数+维度':
        p = best_data['params']
        print(f"   p_c = {p[0]:.3f} / ((z_eff-1)^{p[1]:.3f} · d^{p[2]:.3f})")
    elif best_name == '综合':
        p = best_data['params']
        print(f"   p_c = {p[0]:.3f}/(z^{p[1]:.3f}·d^{p[2]:.3f}) · (1+{p[3]:.3f}·C+{p[4]:.3f}/g)")
    print(f"   LOOCV MAE = {best_data['loocv_mae']:.4f}")


def main():
    print("=" * 80)
    print("🔧 修复泛化问题 — 综合分析与改进")
    print("=" * 80)

    df = load_data()
    print(f"\n数据: {len(df)} 个晶格")

    # 1. 诊断
    diagnose_generalization_failure(df)

    # 2. 测试替代公式
    formulas = test_alternative_formulas(df)

    print("\n--- 全量拟合结果 ---")
    for name, r in formulas.items():
        print(f"  {name}: R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

    # 3. 分组验证
    group_cv_robust(df, formulas)

    # 4. 最终推荐
    final_recommendation(df)


if __name__ == '__main__':
    main()
