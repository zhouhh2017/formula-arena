from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
物理公式在52个真实数据点上的验证
================================
包含: 标准晶格 + Archimedean + 复杂邻域
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize, differential_evolution
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def build_full_physics_dataset():
    """
    构建 52 个晶格的完整物理数据集

    数据来源:
    1. 标准晶格 (Stauffer 1994): 14 个
    2. Archimedean (Suding-Ziff 1998): 7 个
    3. 复杂邻域 (Malarz 2023): 31 个
    """

    data = []

    # ===== 1. 标准晶格 =====
    standard = [
        # 2D
        {'name': 'honeycomb', 'd': 2, 'z': 3, 'pc': 0.697040,
         'g': 6, 'n_tri': 0, 'n_sq': 0, 'C': 0.0},
        {'name': 'square', 'd': 2, 'z': 4, 'pc': 0.592746,
         'g': 4, 'n_tri': 0, 'n_sq': 1, 'C': 0.0},
        {'name': 'triangular', 'd': 2, 'z': 6, 'pc': 0.500000,
         'g': 3, 'n_tri': 6, 'n_sq': 0, 'C': 6*2/(6*5)},  # C = 12/30 = 0.4
        {'name': 'kagome', 'd': 2, 'z': 4, 'pc': 0.652703,
         'g': 3, 'n_tri': 2, 'n_sq': 0, 'C': 2*2/(4*3)},  # C = 4/12 = 0.333

        # 3D
        {'name': 'sc', 'd': 3, 'z': 6, 'pc': 0.311600,
         'g': 4, 'n_tri': 0, 'n_sq': 3, 'C': 0.0},
        {'name': 'bcc', 'd': 3, 'z': 8, 'pc': 0.245800,
         'g': 4, 'n_tri': 0, 'n_sq': 6, 'C': 0.0},
        {'name': 'fcc', 'd': 3, 'z': 12, 'pc': 0.118800,
         'g': 3, 'n_tri': 8, 'n_sq': 6, 'C': 8*2/(12*11)},  # C = 16/132 ≈ 0.121
        {'name': 'diamond', 'd': 3, 'z': 4, 'pc': 0.431900,
         'g': 6, 'n_tri': 0, 'n_sq': 0, 'C': 0.0},

        # 高维
        {'name': 'hypersc_4', 'd': 4, 'z': 8, 'pc': 0.196737,
         'g': 4, 'n_tri': 0, 'n_sq': 12, 'C': 0.0},
        {'name': 'hypersc_5', 'd': 5, 'z': 10, 'pc': 0.140810,
         'g': 4, 'n_tri': 0, 'n_sq': 20, 'C': 0.0},
        {'name': 'hypersc_6', 'd': 6, 'z': 12, 'pc': 0.109460,
         'g': 4, 'n_tri': 0, 'n_sq': 30, 'C': 0.0},
        {'name': 'hypersc_7', 'd': 7, 'z': 14, 'pc': 0.089270,
         'g': 4, 'n_tri': 0, 'n_sq': 42, 'C': 0.0},
        {'name': 'hypersc_8', 'd': 8, 'z': 16, 'pc': 0.075050,
         'g': 4, 'n_tri': 0, 'n_sq': 56, 'C': 0.0},
        {'name': 'hypersc_9', 'd': 9, 'z': 18, 'pc': 0.064530,
         'g': 4, 'n_tri': 0, 'n_sq': 72, 'C': 0.0},
    ]
    data.extend(standard)

    # ===== 2. Archimedean 晶格 =====
    archimedean = [
        {'name': 'arch_3_12_12', 'd': 2, 'z': 3, 'pc': 0.807904,
         'g': 3, 'n_tri': 1, 'n_sq': 0, 'C': 1*2/(3*2)},  # C = 2/6 ≈ 0.333
        {'name': 'arch_4_6_12', 'd': 2, 'z': 4, 'pc': 0.747806,
         'g': 4, 'n_tri': 0, 'n_sq': 1, 'C': 0.0},
        {'name': 'arch_4_8_8', 'd': 2, 'z': 4, 'pc': 0.729724,
         'g': 4, 'n_tri': 0, 'n_sq': 1, 'C': 0.0},
        {'name': 'arch_3_4_6_4', 'd': 2, 'z': 5, 'pc': 0.579498,
         'g': 3, 'n_tri': 1, 'n_sq': 2, 'C': 1*2/(5*4)},  # C = 2/20 = 0.1
        {'name': 'arch_3_4_6_4_b', 'd': 2, 'z': 5, 'pc': 0.621819,
         'g': 3, 'n_tri': 1, 'n_sq': 2, 'C': 0.1},
        {'name': 'arch_3_3_3_4_4', 'd': 2, 'z': 5, 'pc': 0.550213,
         'g': 3, 'n_tri': 3, 'n_sq': 2, 'C': 3*2/(5*4)},  # C = 6/20 = 0.3
        {'name': 'arch_3_3_4_3_4', 'd': 2, 'z': 5, 'pc': 0.550806,
         'g': 3, 'n_tri': 3, 'n_sq': 2, 'C': 0.3},
    ]
    data.extend(archimedean)

    # ===== 3. 复杂邻域 (Malarz 2023) =====
    # 正方晶格的邻居组合, 需要估算物理量
    # 关键: 正方晶格基础 g=4, 加入对角线邻居会引入三角形

    malarz_csv = str(ROOT / 'malarz_2023_data.csv')
    if os.path.exists(malarz_csv):
        df_m = pd.read_csv(malarz_csv)
        for _, row in df_m.iterrows():
            z = row['z']
            # 正方晶格基础: z=4, g=4, n_tri=0
            # 加入对角线邻居 (第2配位区): 引入三角形
            # 估算: 每增加4个对角线邻居, 增加4个三角形
            # 对于 z=4+4k (k 个额外配位区):
            #   n_tri ≈ 4k (粗略)
            #   g = 3 如果有三角形, 否则 4
            extra = z - 4
            if extra > 0:
                n_tri_est = min(extra, 4)  # 粗略估计
                g_est = 3 if n_tri_est > 0 else 4
                C_est = n_tri_est * 2 / (z * (z-1))
            else:
                n_tri_est = 0
                g_est = 4
                C_est = 0.0

            data.append({
                'name': row['name'],
                'd': 2, 'z': z, 'pc': row['pc'],
                'g': g_est, 'n_tri': n_tri_est, 'n_sq': 1,
                'C': C_est,
            })

    df = pd.DataFrame(data)

    # 派生物理量
    df['p_bethe'] = 1.0 / (df['z'] - 1)
    df['z_eff'] = df['z'] - 2 * df['n_tri'] / df['z']
    df['g_inv'] = 1.0 / df['g']
    df['d_inv'] = 1.0 / df['d']

    return df


def fit_and_eval(func, inputs, target, name, bounds, n_restarts=80):
    """拟合并返回结果"""
    best_rmse = float('inf')
    best_params = None

    for _ in range(n_restarts):
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        try:
            result = scipy_minimize(
                lambda p: np.sqrt(np.mean((func(p, *inputs) - target)**2)),
                x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if result.fun < best_rmse:
                pred = func(result.x, *inputs)
                valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
                if valid.sum() > len(target) * 0.8:
                    best_rmse = result.fun
                    best_params = result.x.copy()
        except:
            pass

    try:
        result_de = differential_evolution(
            lambda p: np.sqrt(np.mean((func(p, *inputs) - target)**2)),
            bounds, maxiter=3000, tol=1e-14, seed=42
        )
        pred = func(result_de.x, *inputs)
        valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
        if valid.sum() > len(target) * 0.8 and result_de.fun < best_rmse:
            best_rmse = result_de.fun
            best_params = result_de.x.copy()
    except:
        pass

    if best_params is None:
        return None

    pred = func(best_params, *inputs)
    ss_res = np.sum((pred - target)**2)
    ss_tot = np.sum((target - target.mean())**2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(pred - target))

    return {
        'name': name,
        'params': best_params,
        'func': func,
        'r2': r2,
        'mae': mae,
        'rmse': best_rmse,
        'n_params': len(best_params),
    }


def loocv_refit(func, inputs_list, target, n_params, n_refit=15):
    """LOOCV with refit"""
    n = len(target)
    errors = []

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        inputs_tr = [x[mask] for x in inputs_list]
        target_tr = target[mask]
        inputs_te = [x[~mask] for x in inputs_list]
        target_te = target[~mask]

        best_rmse = float('inf')
        best_p = None
        for _ in range(n_refit):
            x0 = np.random.randn(n_params) * 2
            try:
                result = scipy_minimize(
                    lambda p: np.sqrt(np.mean((func(p, *inputs_tr) - target_tr)**2)),
                    x0, method='Nelder-Mead',
                    options={'maxiter': 2000, 'fatol': 1e-12}
                )
                if result.fun < best_rmse:
                    pred = func(result.x, *inputs_tr)
                    valid = ~np.isnan(pred) & ~np.isinf(pred)
                    if valid.sum() > len(target_tr) * 0.8:
                        best_rmse = result.fun
                        best_p = result.x.copy()
            except:
                pass

        if best_p is not None:
            pred = func(best_p, *inputs_te)
            if not (np.isnan(pred[0]) or np.isinf(pred[0])):
                errors.append(abs(pred[0] - target_te[0]))

    return np.mean(errors) if errors else 1e10


def main():
    print("=" * 80)
    print("🧠 物理公式在 52 个真实数据点上的验证")
    print("=" * 80)

    df = build_full_physics_dataset()
    print(f"\n数据: {len(df)} 个晶格")
    print(f"  2D: {len(df[df['d']==2])}, 3D: {len(df[df['d']==3])}, 高维: {len(df[df['d']>=4])}")

    z = df['z'].values.astype(float)
    d = df['d'].values.astype(float)
    pc = df['pc'].values.astype(float)
    C = df['C'].values.astype(float)
    g = df['g'].values.astype(float)
    n_tri = df['n_tri'].values.astype(float)

    results = {}

    # ===== 物理公式 1: 谱理论+聚类 =====
    print("\n--- 理论: 谱图论 + 聚类系数 ---")
    def f1(p, z, C):
        return p[0] / (z - p[1] + p[2] * C * z)
    r = fit_and_eval(f1, [z, C], pc, "谱理论+聚类",
                     [(0.1, 10), (-2, 3), (-5, 5)])
    if r:
        results['谱理论+聚类'] = r
        print(f"  p_c = {r['params'][0]:.3f} / (z - {r['params'][1]:.3f} + {r['params'][2]:.3f}·C·z)")
        print(f"  R² = {r['r2']:.4f}, MAE = {r['mae']:.4f}")

    # ===== 物理公式 2: Girth-Bethe =====
    print("\n--- 理论: Girth-Bethe ---")
    def f2(p, z, g, d):
        return p[0] / (np.power(np.maximum(z-1, 0.5), p[1]) * np.power(g, p[2]) * np.power(d, p[3]))
    r = fit_and_eval(f2, [z, g, d], pc, "Girth-Bethe",
                     [(0, 20), (0, 3), (-2, 2), (-2, 2)])
    if r:
        results['Girth-Bethe'] = r
        print(f"  p_c = {r['params'][0]:.3f} / ((z-1)^{r['params'][1]:.3f} · g^{r['params'][2]:.3f} · d^{r['params'][3]:.3f})")
        print(f"  R² = {r['r2']:.4f}, MAE = {r['mae']:.4f}")

    # ===== 物理公式 3: Bethe+环修正 =====
    print("\n--- 理论: Bethe + 环修正 ---")
    def f3(p, z, n_tri):
        z_eff = z - p[1] * n_tri / np.maximum(z, 1)
        return p[0] / np.maximum(z_eff - 1, 0.1)
    r = fit_and_eval(f3, [z, n_tri], pc, "Bethe+环修正",
                     [(0.1, 5), (0, 5)])
    if r:
        results['Bethe+环修正'] = r
        print(f"  p_c = {r['params'][0]:.3f} / (z_eff - 1), z_eff = z - {r['params'][1]:.3f}·n_tri/z")
        print(f"  R² = {r['r2']:.4f}, MAE = {r['mae']:.4f}")

    # ===== 物理公式 4: 随机游走+girth =====
    print("\n--- 理论: 随机游走 + Girth ---")
    def f4(p, z, g):
        return p[0] * np.power(z, -p[1]) * (1 + p[2] * np.power(np.maximum(z, 1), 1 - g/2))
    r = fit_and_eval(f4, [z, g], pc, "随机游走+girth",
                     [(0.1, 10), (0, 3), (0, 10)])
    if r:
        results['随机游走+girth'] = r
        print(f"  p_c = {r['params'][0]:.3f} · z^(-{r['params'][1]:.3f}) · (1 + {r['params'][2]:.3f}·z^(1-g/2))")
        print(f"  R² = {r['r2']:.4f}, MAE = {r['mae']:.4f}")

    # ===== 物理公式 5: 标度+上临界维 =====
    print("\n--- 理论: 标度 + 上临界维 ---")
    def f5(p, z, d):
        return (1.0 / np.maximum(z - 1, 0.5)) * (1 + p[0] * np.exp(-p[1] * d))
    r = fit_and_eval(f5, [z, d], pc, "标度+上临界维",
                     [(-3, 5), (0, 3)])
    if r:
        results['标度+上临界维'] = r
        print(f"  p_c = (1 + {r['params'][0]:.3f}·exp(-{r['params'][1]:.3f}·d)) / (z-1)")
        print(f"  R² = {r['r2']:.4f}, MAE = {r['mae']:.4f}")

    # ===== 物理公式 6: 综合多因子 =====
    print("\n--- 理论: 综合多因子 ---")
    def f6(p, z, d, C):
        return p[0] / (np.power(z, p[1]) * np.power(d, p[2])) * (1 + p[3] * C * np.power(z, p[4]))
    r = fit_and_eval(f6, [z, d, C], pc, "综合多因子",
                     [(0, 20), (0, 3), (0, 3), (-5, 5), (-3, 3)])
    if r:
        results['综合多因子'] = r
        print(f"  p_c = {r['params'][0]:.3f}/(z^{r['params'][1]:.3f}·d^{r['params'][2]:.3f})·(1+{r['params'][3]:.3f}·C·z^{r['params'][4]:.3f})")
        print(f"  R² = {r['r2']:.4f}, MAE = {r['mae']:.4f}")

    # ===== LOOCV 验证 =====
    print("\n" + "=" * 80)
    print("🔬 LOOCV 严格验证 (每折重拟合)")
    print("=" * 80)

    loocv_results = {}
    for name, r in results.items():
        func = r['func']
        n_p = r['n_params']
        if '聚类' in name:
            inputs = [z, C]
        elif 'Girth' in name:
            inputs = [z, g, d]
        elif '环修正' in name:
            inputs = [z, n_tri]
        elif '游走' in name:
            inputs = [z, g]
        elif '标度' in name:
            inputs = [z, d]
        elif '综合' in name:
            inputs = [z, d, C]
        else:
            inputs = [z, d]

        loocv_mae = loocv_refit(func, inputs, pc, n_p)
        loocv_results[name] = loocv_mae
        gap = loocv_mae - r['mae']
        print(f"\n{name}:")
        print(f"  训练 MAE = {r['mae']:.4f}")
        print(f"  LOOCV MAE = {loocv_mae:.4f}")
        print(f"  泛化差距 = {gap:.4f}")

    # ===== ML 基线 =====
    print(f"\n--- ML 基线 ---")
    X = df[['z', 'd', 'C', 'g', 'n_tri', 'z_eff', 'p_bethe']].values
    loo = LeaveOneOut()
    lr = LinearRegression()
    lr_pred = []
    for train_idx, test_idx in loo.split(X):
        lr.fit(X[train_idx], pc[train_idx])
        lr_pred.append(lr.predict(X[test_idx])[0])
    mae_ml = mean_absolute_error(pc, lr_pred)
    print(f"ML (7物理特征): LOOCV MAE = {mae_ml:.4f}")

    # ===== 汇总 =====
    print("\n" + "=" * 80)
    print("🏆 最终排名 (52个数据点)")
    print("=" * 80)

    all_r = []
    for name, r in results.items():
        all_r.append({
            'name': name, 'n_params': r['n_params'],
            'train_mae': r['mae'], 'train_r2': r['r2'],
            'loocv_mae': loocv_results.get(name, 1e10),
            'physics': True
        })
    all_r.append({
        'name': 'ML (7物理特征)', 'n_params': 7,
        'train_mae': mae_ml, 'train_r2': None,
        'loocv_mae': mae_ml, 'physics': False
    })
    all_r.sort(key=lambda x: x['loocv_mae'])

    print(f"\n{'#':<3} {'公式':<25} {'参数':<4} {'训练MAE':<10} {'LOOCV':<10} {'物理':<4}")
    print("-" * 60)
    for i, r in enumerate(all_r):
        phys = "✓" if r['physics'] else "✗"
        print(f"#{i+1:<2} {r['name']:<25} {r['n_params']:<4} {r['train_mae']:<10.4f} {r['loocv_mae']:<10.4f} {phys}")

    print(f"\n💡 关键结论:")
    print(f"  最佳物理公式: {all_r[0]['name']}")
    print(f"  LOOCV MAE = {all_r[0]['loocv_mae']:.4f}")
    if all_r[0]['physics']:
        print(f"  ✅ 物理公式优于 ML!")


if __name__ == '__main__':
    main()
