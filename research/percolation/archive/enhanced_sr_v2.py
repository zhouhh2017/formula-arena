from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
精简版增强符号回归 - 重点测试最有希望的公式族
============================================
策略: 直接测试物理启发的公式族，每个做更充分的优化
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize, differential_evolution
from large_scale_benchmark import build_full_dataset

np.random.seed(42)


def prepare_data(df):
    z = df['z'].values.astype(float)
    d = df['d'].values.astype(float)
    pc = df['pc'].values.astype(float)
    z_inv = 1.0 / z
    d_inv = 1.0 / d
    log_z = np.log(z)
    log_d = np.log(d)
    return z, d, pc, z_inv, d_inv, log_z, log_d


def compute_r2_rmse_mae(pred, target):
    valid = ~np.isnan(pred) & ~np.isinf(pred) & (np.abs(pred) < 1e10)
    if valid.sum() < len(target) * 0.5:
        return -1e10, 1e10, 1e10
    p = pred[valid]
    t = target[valid]
    ss_res = np.sum((p - t)**2)
    ss_tot = np.sum((t - t.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((p - t)**2))
    mae = np.mean(np.abs(p - t))
    return r2, rmse, mae


# ===== 公式族定义 =====
def test_formula_family(name, func, z, d, pc, bounds, n_restarts=50):
    """测试一个公式族，用多起点优化"""

    best_result = None
    best_rmse = float('inf')

    for _ in range(n_restarts):
        x0 = np.random.uniform(0.1, 5, size=len(bounds))
        signs = np.random.choice([-1, 1], size=len(bounds))
        x0 = x0 * signs

        try:
            result = scipy_minimize(
                lambda params: np.sqrt(np.mean((func(params, z, d) - pc)**2)),
                x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if result.fun < best_rmse:
                pred = func(result.x, z, d)
                r2, rmse, mae = compute_r2_rmse_mae(pred, pc)
                if r2 > -10 and rmse < best_rmse:
                    best_rmse = rmse
                    best_result = {
                        'params': result.x.copy(),
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                    }
        except:
            pass

    # 也尝试差分进化
    try:
        result_de = differential_evolution(
            lambda params: np.sqrt(np.mean((func(params, z, d) - pc)**2)),
            bounds, maxiter=2000, tol=1e-14, seed=42,
            mutation=(0.5, 1.5), recombination=0.9
        )
        if result_de.fun < best_rmse:
            pred = func(result_de.x, z, d)
            r2, rmse, mae = compute_r2_rmse_mae(pred, pc)
            if r2 > -10:
                best_rmse = rmse
                best_result = {
                    'params': result_de.x.copy(),
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                }
    except:
        pass

    return best_result, name


def main():
    print("=" * 80)
    print("🔬 精简版增强符号回归 (52个真实数据点)")
    print("=" * 80)

    df = build_full_dataset()
    z, d, pc, z_inv, d_inv, log_z, log_d = prepare_data(df)

    print(f"\n数据: {len(df)} 个晶格")
    print(f"p_c 范围: {pc.min():.5f} - {pc.max():.5f}")

    results = []

    # ===== 公式族 1: 幂律 + 修正项 =====
    print("\n--- 公式族: 幂律 + 修正项 ---")

    # 1a: a * z^b * d^c (基础)
    def f1(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2])
    r, name = test_formula_family("a*z^b*d^c", f1, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0)], n_restarts=80)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 1b: a * z^b * d^c + e
    def f2(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2]) + p[3]
    r, name = test_formula_family("a*z^b*d^c+e", f2, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0),(-1,1)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})+{r['params'][3]:.4f}"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 1c: a * z^b * d^c * (1 + e/z)
    def f3(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2]) * (1 + p[3]/z)
    r, name = test_formula_family("a*z^b*d^c*(1+e/z)", f3, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0),(-5,5)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})*(1+{r['params'][3]:.4f}/z)"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 1d: a * z^b * d^c * (1 + e/d)
    def f4(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2]) * (1 + p[3]/d)
    r, name = test_formula_family("a*z^b*d^c*(1+e/d)", f4, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0),(-5,5)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})*(1+{r['params'][3]:.4f}/d)"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 1e: a * z^b * d^c * exp(e/(z*d))
    def f5(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2]) * np.exp(p[3]/(z*d))
    r, name = test_formula_family("a*z^b*d^c*exp(e/(z*d))", f5, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0),(-10,10)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})*exp({r['params'][3]:.4f}/(z*d))"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # ===== 公式族 2: Bethe 近似改进 =====
    print("\n--- 公式族: Bethe 近似改进 ---")

    # 2a: a / (z - b) * d^c
    def f6(p, z, d): return p[0] / (z - p[1]) * np.power(d, p[2])
    r, name = test_formula_family("a/(z-b)*d^c", f6, z, d, pc,
                                   [(-10,10),(0,3),(-5,0)], n_restarts=80)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}/(z-{r['params'][1]:.4f})*d^({r['params'][2]:.4f})"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 2b: a / (z - b)^c * d^e
    def f7(p, z, d): return p[0] / np.power(np.abs(z - p[1]), p[2]) * np.power(d, p[3])
    r, name = test_formula_family("a/(z-b)^c*d^e", f7, z, d, pc,
                                   [(-10,10),(0,3),(0,3),(-5,0)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}/(z-{r['params'][1]:.4f})^{r['params'][2]:.4f}*d^({r['params'][3]:.4f})"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # ===== 公式族 3: 有理函数 =====
    print("\n--- 公式族: 有理函数 ---")

    # 3a: a / (b + z^c * d^e)
    def f8(p, z, d): return p[0] / (p[1] + np.power(z, p[2]) * np.power(d, p[3]))
    r, name = test_formula_family("a/(b+z^c*d^e)", f8, z, d, pc,
                                   [(-10,10),(-5,10),(-3,0),(-3,0)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}/({r['params'][1]:.4f}+z^{r['params'][2]:.4f}*d^{r['params'][3]:.4f})"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 3b: (a + b*z) / (c + d*z + e*d)
    def f9(p, z, d): return (p[0] + p[1]*z) / (p[2] + p[3]*z + p[4]*d)
    r, name = test_formula_family("(a+b*z)/(c+d*z+e*d)", f9, z, d, pc,
                                   [(-10,10),(-5,5),(-5,10),(-5,5),(-5,5)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"({r['params'][0]:.4f}+{r['params'][1]:.4f}*z)/({r['params'][2]:.4f}+{r['params'][3]:.4f}*z+{r['params'][4]:.4f}*d)"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # ===== 公式族 4: 对数/指数 =====
    print("\n--- 公式族: 对数/指数混合 ---")

    # 4a: a * exp(b*log(z) + c*log(d))  等价于 a * z^b * d^c
    # 跳过 (同1a)

    # 4b: a / (b*log(z) + c*log(d) + e)
    def f10(p, z, d): return p[0] / (p[1]*np.log(z) + p[2]*np.log(d) + p[3])
    r, name = test_formula_family("a/(b*log(z)+c*log(d)+e)", f10, z, d, pc,
                                   [(-10,10),(-5,5),(-5,5),(-5,10)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}/({r['params'][1]:.4f}*log(z)+{r['params'][2]:.4f}*log(d)+{r['params'][3]:.4f})"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 4c: a * exp(-b*z^c * d^e)
    def f11(p, z, d): return p[0] * np.exp(-np.abs(p[1]) * np.power(z, p[2]) * np.power(d, p[3]))
    r, name = test_formula_family("a*exp(-b*z^c*d^e)", f11, z, d, pc,
                                   [(-5,5),(-5,5),(-3,0),(-3,0)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*exp(-{abs(r['params'][1]):.4f}*z^{r['params'][2]:.4f}*d^{r['params'][3]:.4f})"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # ===== 公式族 5: 复合非线性 =====
    print("\n--- 公式族: 复合非线性 ---")

    # 5a: a * z^b * d^c + e * z_inv * d_inv
    def f12(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2]) + p[3] / (z * d)
    r, name = test_formula_family("a*z^b*d^c + e/(z*d)", f12, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0),(-5,5)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})+{r['params'][3]:.4f}/(z*d)"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 5b: a * z^b * d^c + e * log(z) * log(d)
    def f13(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2]) + p[3] * np.log(z) * np.log(d)
    r, name = test_formula_family("a*z^b*d^c + e*log(z)*log(d)", f13, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0),(-5,5)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})+{r['params'][3]:.4f}*log(z)*log(d)"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 5c: a * (z*d)^b + c
    def f14(p, z, d): return p[0] * np.power(z*d, p[1]) + p[2]
    r, name = test_formula_family("a*(z*d)^b+c", f14, z, d, pc,
                                   [(-10,10),(-3,0),(-1,1)], n_restarts=80)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*(z*d)^({r['params'][1]:.4f})+{r['params'][2]:.4f}"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 5d: a * z^b * d^c + e * z^f
    def f15(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2]) + p[3] * np.power(z, p[4])
    r, name = test_formula_family("a*z^b*d^c + e*z^f", f15, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0),(-5,5),(-3,0)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})+{r['params'][3]:.4f}*z^({r['params'][4]:.4f})"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 5e: a * z^b * d^c + e * d^f
    def f16(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2]) + p[3] * np.power(d, p[4])
    r, name = test_formula_family("a*z^b*d^c + e*d^f", f16, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0),(-5,5),(-3,0)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})+{r['params'][3]:.4f}*d^({r['params'][4]:.4f})"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # 5f: a * z^b / (1 + c*z^e * d^f)
    def f17(p, z, d): return p[0] * np.power(z, p[1]) / (1 + p[2] * np.power(z, p[3]) * np.power(d, p[4]))
    r, name = test_formula_family("a*z^b/(1+c*z^e*d^f)", f17, z, d, pc,
                                   [(-10,10),(-5,2),(-5,5),(-3,0),(-3,0)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})/(1+{r['params'][2]:.4f}*z^{r['params'][3]:.4f}*d^{r['params'][4]:.4f})"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # ===== 公式族 6: 2D/3D 分段 (用 sigmoid 平滑) =====
    print("\n--- 公式族: 维度依赖分段 ---")

    # 6a: a*z^b*d^c * (1 + e * exp(-f*(d-2.5)))
    def f18(p, z, d): return p[0] * np.power(z, p[1]) * np.power(d, p[2]) * (1 + p[3] * 1/(1+np.exp(-p[4]*(d-2.5))))
    r, name = test_formula_family("a*z^b*d^c*(1+e*sigmoid(f*(d-2.5)))", f18, z, d, pc,
                                   [(-10,10),(-5,0),(-5,0),(-2,2),(0,5)], n_restarts=60)
    if r:
        results.append({**r, 'name': name, 'formula': f"{r['params'][0]:.4f}*z^({r['params'][1]:.4f})*d^({r['params'][2]:.4f})*(1+{r['params'][3]:.4f}*sigmoid({r['params'][4]:.4f}*(d-2.5)))"})
        print(f"  {name}: MAE={r['mae']:.4f}, R²={r['r2']:.4f}")

    # ===== 汇总 =====
    results.sort(key=lambda x: x['mae'])

    print("\n" + "=" * 80)
    print("🏆 Top-10 公式排名")
    print("=" * 80)

    for i, r in enumerate(results[:10]):
        print(f"\n#{i+1}: {r['name']}")
        print(f"  R² = {r['r2']:.6f}")
        print(f"  RMSE = {r['rmse']:.6f}")
        print(f"  MAE = {r['mae']:.6f}")
        print(f"  公式: {r['formula']}")

    # 对比基准
    print("\n" + "=" * 80)
    print("📊 对比基准")
    print("=" * 80)

    from large_scale_benchmark import test_power_law, test_ml_methods
    params, r2_power, rmse_power = test_power_law(df)
    ml_results = test_ml_methods(df)
    best_ml_mae = min(r['mae'] for r in ml_results.values())

    print(f"\n  当前幂律 (3参数):    MAE = {rmse_power:.4f}, R² = {r2_power:.4f}")
    print(f"  最佳 ML (7参数):     MAE = {best_ml_mae:.4f}")

    if results:
        best = results[0]
        n_params = best['formula'].count('*') + 1  # 粗略估计
        print(f"  扫地僧增强 (最优):   MAE = {best['mae']:.4f}, R² = {best['r2']:.4f}")
        print(f"  公式: {best['formula']}")

        improvement = (rmse_power - best['mae']) / rmse_power * 100
        if best['mae'] < rmse_power:
            print(f"\n  ✅ 扫地僧找到更好的公式! MAE 提升: {improvement:.1f}%")
        elif abs(improvement) < 5:
            print(f"\n  ⚖️  性能持平 (差距: {abs(improvement):.1f}%)")
        else:
            print(f"\n  ⚠️  当前幂律仍更优 (差距: {abs(improvement):.1f}%)")

    # 保存
    output_file = str(ROOT / 'enhanced_sr_v2_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("精简版增强符号回归结果 (52个真实数据点)\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(results):
            f.write(f"\n#{i+1}: {r['name']}\n")
            f.write(f"  R² = {r['r2']:.6f}\n")
            f.write(f"  RMSE = {r['rmse']:.6f}\n")
            f.write(f"  MAE = {r['mae']:.6f}\n")
            f.write(f"  公式: {r['formula']}\n")

    print(f"\n💾 结果已保存: {output_file}")


if __name__ == '__main__':
    main()
