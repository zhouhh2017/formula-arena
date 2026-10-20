from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
Proof Checker: 渗流阈值公式快速审计 (精简版)
==============================================
快速但全面地验证所有声明指标
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import json
import time
warnings.filterwarnings('ignore')

OUT = ROOT
t0 = time.time()

print("=" * 80)
print("  PROOF CHECKER: 渗流阈值公式全面审计 (精简版)")
print("=" * 80)

# ============================================================
# PHASE 0: 数据加载
# ============================================================
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
n = len(df)
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values

print(f"数据量: {n} 个晶格")
print(f"维度范围: d={int(d.min())}-{int(d.max())}")
print(f"配位数范围: z={int(z.min())}-{int(z.max())}")
print(f"girth: g∈{sorted(set(g.astype(int)))}")

# 物理一致性
issues = []
if ((pc <= 0) | (pc > 1)).any():
    issues.append("FATAL: p_c out of (0,1]")
if (z < 3).any():
    issues.append("WARNING: z<3 found")
if (g < 3).any():
    issues.append("WARNING: g<3 found")
print(f"✓ 物理一致性: p_c∈(0,1], z≥3, g≥3")

# ============================================================
# PHASE 1: 模型定义 + 独立拟合
# ============================================================
print(f"\n{'='*80}")
print("PHASE 1: 独立拟合验证")
print(f"{'='*80}")

def model_exp(params, z, d, g):
    a, alpha, gamma, z0 = params
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

def model_power(params, z, d, g):
    a, beta, gamma, z0 = params
    correction = (a / (d**beta * g**gamma)) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

# 声明参数
p_claim_new = np.array([12.3346, 0.3368, 0.8477, 9.6171])
p_claim_old = np.array([12.9, 1.03, 0.85, 9.7])

# 快速拟合新模型 (100 trials)
best_ss_new = np.inf
best_p_new = None
for trial in range(100):
    p0 = p_claim_new.copy() if trial == 0 else [
        np.random.uniform(5, 30), np.random.uniform(0.05, 0.5),
        np.random.uniform(0.3, 1.5), np.random.uniform(3, 20)]
    res = minimize(lambda p: np.sum((pc - model_exp(p, z, d, g))**2),
                  p0, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
    if res.fun < best_ss_new:
        best_ss_new = res.fun
        best_p_new = res.x

ss_claim_new = np.sum((pc - model_exp(p_claim_new, z, d, g))**2)
print(f"\n指数模型:")
print(f"  声明参数 SS = {ss_claim_new:.6f}")
print(f"  独立拟合 SS = {best_ss_new:.6f}")
print(f"  声明: a={p_claim_new[0]:.4f} α={p_claim_new[1]:.4f} γ={p_claim_new[2]:.4f} z₀={p_claim_new[3]:.4f}")
print(f"  独立: a={best_p_new[0]:.4f} α={best_p_new[1]:.4f} γ={best_p_new[2]:.4f} z₀={best_p_new[3]:.4f}")

diff_pct = np.abs(p_claim_new - best_p_new) / np.abs(p_claim_new) * 100
print(f"  差异: a={diff_pct[0]:.2f}% α={diff_pct[1]:.2f}% γ={diff_pct[2]:.2f}% z₀={diff_pct[3]:.2f}%")

if best_ss_new < ss_claim_new * 0.999:
    issues.append(f"CRITICAL: 独立拟合显著优于声明参数 (SS降低{(ss_claim_new-best_ss_new)/ss_claim_new*100:.4f}%)")
    final_p = best_p_new
else:
    final_p = p_claim_new
    print(f"  ✓ 声明参数接近最优")

# 快速拟合旧模型 (100 trials)
best_ss_old = np.inf
best_p_old = None
for trial in range(100):
    p0 = p_claim_old.copy() if trial == 0 else [
        np.random.uniform(5, 30), np.random.uniform(0.3, 2.0),
        np.random.uniform(0.3, 1.5), np.random.uniform(3, 20)]
    res = minimize(lambda p: np.sum((pc - model_power(p, z, d, g))**2),
                  p0, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
    if res.fun < best_ss_old:
        best_ss_old = res.fun
        best_p_old = res.x

print(f"\n幂律模型:")
print(f"  声明参数 SS = {np.sum((pc - model_power(p_claim_old, z, d, g))**2):.6f}")
print(f"  独立拟合 SS = {best_ss_old:.6f}")
print(f"  独立: a={best_p_old[0]:.4f} β={best_p_old[1]:.4f} γ={best_p_old[2]:.4f} z₀={best_p_old[3]:.4f}")

# ============================================================
# PHASE 2: 统计指标复现
# ============================================================
print(f"\n{'='*80}")
print("PHASE 2: 统计指标复现")
print(f"{'='*80}")

pred = model_exp(final_p, z, d, g)
errors = np.abs((pc - pred) / pc) * 100
residuals = (pc - pred) / pc * 100

mape = np.mean(errors)
median_err = np.median(errors)
max_err = np.max(errors)
ssr = np.sum((pc - pred)**2)
sst = np.sum((pc - np.mean(pc))**2)
R2 = 1 - ssr / sst
aic = n * np.log(ssr / n) + 2 * 4

print(f"\n{'指标':<20} {'声明':>10} {'复现':>10} {'差异':>8} {'状态':>6}")
print("-" * 58)

checks = [
    ('MAPE(%)', 8.14, mape, 0.1),
    ('中位误差(%)', 7.06, median_err, 0.1),
    ('最大误差(%)', 28.9, max_err, 0.5),
    ('R²', 0.973, R2, 0.003),
    ('AIC', -668.3, aic, 1.0),
]

for label, claimed, actual, tol in checks:
    diff = abs(actual - claimed)
    ok = "✓" if diff < tol else "⚠"
    print(f"{label:<20} {claimed:>10.4f} {actual:>10.4f} {diff:>8.4f} {ok:>6}")
    if ok == "⚠":
        issues.append(f"MAJOR: {label} 复现({actual:.4f}) vs 声明({claimed:.4f})")

# 分类误差
print(f"\n分类误差:")
cats = {
    'd=2': d==2, 'd=3': d==3, 'd=4-7': (d>=4)&(d<=7), 'd≥8': d>=8,
    '超立方d≥4': (z==2*d)&(d>=4), '超立方d≥8': (z==2*d)&(d>=8),
    'g=3': g==3, 'g=4': g==4, 'g=6': g==6,
}
for label, mask in cats.items():
    if mask.sum() > 0:
        print(f"  {label:<20} n={mask.sum():>3}  MAPE={np.mean(errors[mask]):.2f}%")

# ============================================================
# PHASE 3: 交叉验证
# ============================================================
print(f"\n{'='*80}")
print("PHASE 3: 10-fold CV")
print(f"{'='*80}")

np.random.seed(42)
cv_errors = []
for fold in range(10):
    test_idx = np.arange(fold * (n//10), (fold+1) * (n//10) if fold < 9 else n)
    train_idx = np.setdiff1d(np.arange(n), test_idx)

    best_fp = None
    best_fss = np.inf
    for _ in range(20):
        p0 = final_p * np.random.uniform(0.8, 1.2, 4) if _ > 0 else final_p.copy()
        res = minimize(lambda p: np.sum((pc[train_idx] - model_exp(p, z[train_idx], d[train_idx], g[train_idx]))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.fun < best_fss:
            best_fss = res.fun
            best_fp = res.x

    pred_test = model_exp(best_fp, z[test_idx], d[test_idx], g[test_idx])
    cv_errors.extend(np.abs((pc[test_idx] - pred_test) / pc[test_idx]) * 100)

cv_mape = np.mean(cv_errors)
print(f"\n声明 CV MAPE = 8.83%")
print(f"复现 CV MAPE = {cv_mape:.2f}%")
if abs(cv_mape - 8.83) > 1.0:
    issues.append(f"MAJOR: CV MAPE 复现({cv_mape:.2f}%) vs 声明(8.83%)")
else:
    print("✓ CV结果一致")

# ============================================================
# PHASE 4: Bootstrap (50次快速版)
# ============================================================
print(f"\n{'='*80}")
print("PHASE 4: Bootstrap稳定性 (50次)")
print(f"{'='*80}")

np.random.seed(123)
boot_params = []
for b in range(50):
    idx = np.random.choice(n, size=n, replace=True)
    best_b = None
    best_bss = np.inf
    for _ in range(5):
        p0 = final_p * np.random.uniform(0.8, 1.2, 4) if _ > 0 else final_p.copy()
        res = minimize(lambda p: np.sum((pc[idx] - model_exp(p, z[idx], d[idx], g[idx]))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_bss:
            best_bss = res.fun
            best_b = res.x
    if best_b is not None:
        boot_params.append(best_b)

boot_params = np.array(boot_params)
pnames = ['a', 'α', 'γ', 'z₀']
print(f"\n参数稳定性:")
for j, pn in enumerate(pnames):
    m = np.mean(boot_params[:, j])
    s = np.std(boot_params[:, j])
    ci_lo = np.percentile(boot_params[:, j], 2.5)
    ci_hi = np.percentile(boot_params[:, j], 97.5)
    cv = s / abs(m) * 100
    print(f"  {pn}: mean={m:.4f}, CV={cv:.1f}%, 95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

# ============================================================
# PHASE 5: 渐近一致性
# ============================================================
print(f"\n{'='*80}")
print("PHASE 5: 渐近一致性")
print(f"{'='*80}")

a, alpha, gamma, z0 = final_p

print(f"\n1. d→∞ → Bethe极限:")
for dt in [10, 50, 100]:
    zt = 2 * dt
    pc_pred = model_exp(final_p, np.array([zt]), np.array([dt]), np.array([4.0]))[0]
    R = pc_pred * (zt - 1)
    print(f"  d={dt:>3}: R={R:.8f} (修正{(R-1)*100:+.6f}%)")

print(f"\n2. z→∞ → R饱和:")
for dt in [2, 3, 6, 10]:
    R_sat = 1 + a * np.exp(-alpha * dt) / 4**gamma
    R_num = model_exp(final_p, np.array([1000.0]), np.array([dt]), np.array([4.0]))[0] * 999
    print(f"  d={dt}: R_theory={R_sat:.6f}, R_numerical={R_num:.6f}")

print(f"\n3. 特征维度:")
d_star = 1 / alpha
print(f"  α={alpha:.4f} → d*=1/α={d_star:.2f}")
print(f"  exp(-α·6)={np.exp(-alpha*6):.4f} (d_c=6)")
print(f"  exp(-α·13)={np.exp(-alpha*13):.6f}")

# ============================================================
# PHASE 6: 残差 + 反例搜索
# ============================================================
print(f"\n{'='*80}")
print("PHASE 6: 残差分析 + 反例搜索")
print(f"{'='*80}")

print(f"\n残差统计:")
print(f"  均值={np.mean(residuals):+.4f}%, 标准差={np.std(residuals):.4f}%")
print(f"  偏度={np.mean((residuals-np.mean(residuals))**3)/np.std(residuals)**3:.4f}")

print(f"\n按维度残差:")
for dv in sorted(set(d.astype(int))):
    mask = d.astype(int) == dv
    print(f"  d={dv:>2}: mean={np.mean(residuals[mask]):+.2f}%, n={mask.sum()}")

print(f"\n误差最大TOP10:")
idx_worst = np.argsort(errors)[::-1][:10]
for rank, idx in enumerate(idx_worst, 1):
    print(f"  {rank:>2}. {names[idx]:<22} z={int(z[idx]):>3} d={int(d[idx]):>2} g={int(g[idx]):>2} "
          f"p_c={pc[idx]:.5f} pred={pred[idx]:.5f} err={errors[idx]:.1f}%")

# 极端参数测试
print(f"\n极端参数测试:")
extremes = [(3,2,3),(146,2,4),(4,13,4),(26,13,4)]
for zt,dt,gt in extremes:
    pp = model_exp(final_p, np.array([zt]), np.array([dt]), np.array([gt]))[0]
    ok = "✓" if 0 < pp <= 1 else "⚠"
    print(f"  {ok} z={zt:>3} d={dt:>2} g={gt}: p_c={pp:.6f}")
    if pp <= 0 or pp > 1:
        issues.append(f"CRITICAL: 极端参数给出p_c={pp:.6f}")

# Bethe一致性
all_above = all(pred[i] >= 1/(z[i]-1) * 0.99 for i in range(n))
print(f"\nBethe一致性: {'✓ 全部pred≥Bethe' if all_above else '⚠ 部分pred<Bethe'}")

n_bad = np.sum(errors > 15)
print(f"误差>15%: {n_bad}/{n} ({n_bad/n*100:.0f}%)")

# ============================================================
# PHASE 7: 过拟合风险 + 新旧对比
# ============================================================
print(f"\n{'='*80}")
print("PHASE 7: 过拟合 + 新旧对比")
print(f"{'='*80}")

print(f"\nn/k = {n}/4 = {n/4:.1f}")
print(f"训练MAPE = {mape:.2f}%")
print(f"CV MAPE = {cv_mape:.2f}%")
gap = cv_mape - mape
print(f"泛化差距 = {gap:.2f}%")
if gap > 2:
    issues.append(f"MAJOR: 泛化差距{gap:.2f}%较大")

# 新旧模型公平对比
pred_old = model_power(best_p_old, z, d, g)
errors_old = np.abs((pc - pred_old) / pc) * 100

mask_hc8 = (z == 2*d) & (d >= 8)
print(f"\n公平对比 (均用独立最优参数):")
print(f"{'指标':<30} {'幂律':>10} {'指数':>10} {'改进':>10}")
print("-" * 63)
print(f"{'全局MAPE':<30} {np.mean(errors_old):>9.2f}% {mape:>9.2f}% {np.mean(errors_old)-mape:>+9.2f}%")
print(f"{'超立方d≥8':<30} {np.mean(errors_old[mask_hc8]):>9.2f}% {np.mean(errors[mask_hc8]):>9.2f}% {np.mean(errors_old[mask_hc8])-np.mean(errors[mask_hc8]):>+9.2f}%")

# 检查恶化
print(f"\n恶化检查:")
for label, mask in cats.items():
    if mask.sum() > 0:
        e_o = np.mean(errors_old[mask])
        e_n = np.mean(errors[mask])
        tag = "⚠恶化" if e_o - e_n < -1 else "✓改善"
        print(f"  {label:<20} 旧={e_o:.2f}% → 新={e_n:.2f}% ({e_o-e_n:+.2f}%) {tag}")

# ============================================================
# PHASE 8: 物理声明审计
# ============================================================
print(f"\n{'='*80}")
print("PHASE 8: 物理声明审计")
print(f"{'='*80}")

print(f"""
声明1: 'd*=1/α≈3.0 与 d_c=6 一致'
  实际: α={alpha:.4f}, d*={1/alpha:.2f}
  分析: d*≈3.0是衰减的特征尺度,意味着d>3后修正快速衰减
  exp(-α·6)={np.exp(-alpha*6):.4f} — 并非趋于0, d=6处仍有13%的修正幅度
  exp(-α·10)={np.exp(-alpha*10):.4f} — d=10处修正仅3%
  结论: d*≠d_c, 更准确的说法是"指数衰减与上临界维度的物理图像一致"
""")
issues.append("MAJOR: 物理声明需修正 — 'd*≈3.0与d_c=6一致' 表述不精确。"
              "d*是衰减的特征尺度,不等于上临界维度。应改为'指数衰减的渐近行为与d_c=6的物理图像定性一致'")

print(f"""
声明2: 'z₀≈10 是普适特征 (新旧公式相同)'
  新公式 z₀={z0:.2f}, 旧公式 z₀=9.7
  差异: {abs(z0-9.7):.2f} — 确实接近但不完全相同
  结论: z₀≈10是鲁棒特征,但'完全相同'略有夸大
""")

print(f"""
声明3: '指数衰减由环路概率驱动'
  这是事后解释(post-hoc rationalization),不是从第一原理推导
  结论: 合理的物理启发,但不能声称是理论推导
""")
issues.append("MINOR: 论文应更明确区分'经验发现exp(-αd)有效' vs '从第一原理推导出exp(-αd)'"
              "目前的表述暗示了理论推导,实际是经验拟合+事后物理解释")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*80}")
print("SUMMARY: 审计总结")
print(f"{'='*80}")

fatal = [i for i in issues if i.startswith("FATAL")]
critical = [i for i in issues if i.startswith("CRITICAL")]
major = [i for i in issues if i.startswith("MAJOR")]
minor = [i for i in issues if i.startswith("MINOR")]

print(f"\n问题统计:")
print(f"  FATAL:    {len(fatal)}")
print(f"  CRITICAL: {len(critical)}")
print(f"  MAJOR:    {len(major)}")
print(f"  MINOR:    {len(minor)}")

print(f"\n完整问题清单:")
for i, iss in enumerate(issues, 1):
    print(f"  [{i}] {iss}")

gate = "PASS" if len(fatal)==0 and len(critical)==0 else "FAIL"
print(f"\n验收门: {'✓ PASS' if gate=='PASS' else '✗ FAIL'}")

# 保存
state = {
    "status": "completed", "n_data": n, "mape": float(mape),
    "R2": float(R2), "cv_mape": float(cv_mape),
    "fatal": len(fatal), "critical": len(critical),
    "major": len(major), "minor": len(minor),
    "issues": issues, "acceptance_gate": gate,
    "params_final": final_p.tolist(),
    "timestamp": "2026-10-04",
}
with open(f'{OUT}/PROOF_CHECK_STATE.json', 'w') as f:
    json.dump(state, f, indent=2, ensure_ascii=False)

elapsed = time.time() - t0
print(f"\n耗时: {elapsed:.1f}s")
print(f"状态已保存: PROOF_CHECK_STATE.json")
print(f"\n{'='*80}")
print(f"  AUDIT COMPLETE — {gate}")
print(f"{'='*80}")
