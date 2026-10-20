from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
Critical PRL Assessment
========================
Rigorous evaluation of whether the percolation threshold formula meets PRL standards.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

print("=" * 80)
print("【CRITICAL PRL ASSESSMENT】")
print("=" * 80)

# Load data
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values
n = len(pc)

print(f"\nDataset: {n} lattices")
print(f"Dimensions: d = {int(d.min())} to {int(d.max())}")
print(f"Coordination: z = {int(z.min())} to {int(z.max())}")
print(f"Girth values: {sorted(set(g.astype(int)))}")

# Model
a, alpha, gamma, z0 = 12.3346, 0.3368, 0.8477, 9.6171

def model(z, d, g):
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

pred = model(z, d, g)
errors = np.abs((pc - pred) / pc) * 100
residuals = (pred - pc) / pc * 100

# ============================================================
# CRITICAL TEST 1: Precision Assessment
# ============================================================
print(f"\n{'='*80}")
print("【CRITICAL TEST 1: Precision Assessment】")
print(f"{'='*80}")

print(f"\nMAPE = {np.mean(errors):.2f}%")
print(f"Median error = {np.median(errors):.2f}%")
print(f"Max error = {np.max(errors):.2f}%")
print(f"95th percentile error = {np.percentile(errors, 95):.2f}%")
print(f"Lattices with error > 10%: {np.sum(errors > 10)}/{n} ({np.sum(errors > 10)/n*100:.1f}%)")
print(f"Lattices with error > 15%: {np.sum(errors > 15)}/{n} ({np.sum(errors > 15)/n*100:.1f}%)")
print(f"Lattices with error > 20%: {np.sum(errors > 20)}/{n} ({np.sum(errors > 20)/n*100:.1f}%)")

# PRL standard: typically need <5% error for precision claims
if np.mean(errors) < 5:
    print(f"\n✓ PASS: MAPE < 5% — meets precision standard")
elif np.mean(errors) < 10:
    print(f"\n⚠ MARGINAL: MAPE < 10% — acceptable but not exceptional")
else:
    print(f"\n✗ FAIL: MAPE > 10% — does not meet precision standard for PRL")

# ============================================================
# CRITICAL TEST 2: High-Dimensional Claim
# ============================================================
print(f"\n{'='*80}")
print("【CRITICAL TEST 2: High-Dimensional Claim Verification】")
print(f"{'='*80}")

mask_hc = z == 2*d
mask_high = mask_hc & (d >= 8)

if np.sum(mask_high) > 0:
    err_high = np.mean(errors[mask_high])
    print(f"\nd ≥ 8 hypercubic lattices: {np.sum(mask_high)}")
    print(f"MAPE = {err_high:.2f}%")

    for i in np.where(mask_high)[0]:
        print(f"  d={int(d[i])}, z={int(z[i])}: p_c={pc[i]:.6f}, pred={pred[i]:.6f}, error={errors[i]:.2f}%")

    if err_high < 5:
        print(f"\n✓ PASS: High-dimensional error < 5%")
    elif err_high < 10:
        print(f"\n⚠ MARGINAL: High-dimensional error < 10%")
    else:
        print(f"\n✗ FAIL: High-dimensional error > 10%")
else:
    print("\n✗ No high-dimensional hypercubic lattices in dataset")

# ============================================================
# CRITICAL TEST 3: Statistical Significance of Exponential vs Power-Law
# ============================================================
print(f"\n{'='*80}")
print("【CRITICAL TEST 3: Exponential vs Power-Law — Statistical Significance】")
print(f"{'='*80}")

# Power-law model
def model_power(z, d, g):
    a_p, beta_p, gamma_p, z0_p = 12.9, 1.03, 0.85, 9.7
    correction = (a_p / (d**beta_p * g**gamma_p)) * (1 - np.exp(-z / z0_p))
    return (1 + correction) / (z - 1)

pred_power = model_power(z, d, g)
errors_power = np.abs((pc - pred_power) / pc) * 100

ssr_exp = np.sum((pc - pred)**2)
ssr_power = np.sum((pc - pred_power)**2)

aic_exp = n * np.log(ssr_exp/n) + 2*4
aic_power = n * np.log(ssr_power/n) + 2*4

bic_exp = n * np.log(ssr_exp/n) + 4*np.log(n)
bic_power = n * np.log(ssr_power/n) + 4*np.log(n)

print(f"\nExponential model:")
print(f"  SSR = {ssr_exp:.6f}")
print(f"  AIC = {aic_exp:.2f}")
print(f"  BIC = {bic_exp:.2f}")
print(f"  MAPE = {np.mean(errors):.2f}%")

print(f"\nPower-law model:")
print(f"  SSR = {ssr_power:.6f}")
print(f"  AIC = {aic_power:.2f}")
print(f"  BIC = {bic_power:.2f}")
print(f"  MAPE = {np.mean(errors_power):.2f}%")

print(f"\nDifferences:")
print(f"  ΔAIC = {aic_power - aic_exp:.2f}")
print(f"  ΔBIC = {bic_power - bic_exp:.2f}")
print(f"  ΔMAPE = {np.mean(errors_power) - np.mean(errors):.2f}%")

# Statistical significance of AIC difference
# Rule of thumb: ΔAIC > 2 is positive evidence, > 10 is strong evidence
delta_aic = aic_power - aic_exp
if delta_aic > 10:
    print(f"\n✓ STRONG: ΔAIC > 10 — strong evidence for exponential model")
elif delta_aic > 2:
    print(f"\n⚠ POSITIVE: ΔAIC > 2 — positive evidence for exponential model")
else:
    print(f"\n✗ WEAK: ΔAIC < 2 — weak evidence, models not distinguishable")

# Paired t-test on errors
t_stat, p_value = stats.ttest_rel(errors_power, errors)
print(f"\nPaired t-test on errors:")
print(f"  t-statistic = {t_stat:.3f}")
print(f"  p-value = {p_value:.4f}")
if p_value < 0.01:
    print(f"  ✓ Statistically significant (p < 0.01)")
elif p_value < 0.05:
    print(f"  ⚠ Marginally significant (p < 0.05)")
else:
    print(f"  ✗ Not statistically significant (p > 0.05)")

# ============================================================
# CRITICAL TEST 4: Cross-Validation Robustness
# ============================================================
print(f"\n{'='*80}")
print("【CRITICAL TEST 4: Cross-Validation Robustness】")
print(f"{'='*80}")

np.random.seed(42)
cv_errors = []
for fold in range(10):
    # Split
    test_idx = np.random.choice(n, size=n//10, replace=False)
    train_idx = np.setdiff1d(np.arange(n), test_idx)

    # Fit on train
    def model_fit(params, z, d, g):
        a, alpha, gamma, z0 = params
        correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
        return (1 + correction) / (z - 1)

    best_ss = np.inf
    best_params = None
    for _ in range(50):
        p0 = [a*np.random.uniform(0.7,1.3), alpha*np.random.uniform(0.7,1.3),
              gamma*np.random.uniform(0.7,1.3), z0*np.random.uniform(0.7,1.3)]
        res = minimize(lambda p: np.sum((pc[train_idx] - model_fit(p, z[train_idx], d[train_idx], g[train_idx]))**2),
                      p0, method='Nelder-Mead',
                      options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_ss:
            best_ss = res.fun
            best_params = res.x

    # Evaluate on test
    pred_test = model_fit(best_params, z[test_idx], d[test_idx], g[test_idx])
    err_test = np.abs((pc[test_idx] - pred_test) / pc[test_idx]) * 100
    cv_errors.extend(err_test.tolist())

cv_mape = np.mean(cv_errors)
cv_std = np.std(cv_errors)

print(f"\n10-fold cross-validation:")
print(f"  CV MAPE = {cv_mape:.2f}%")
print(f"  CV std = {cv_std:.2f}%")
print(f"  95% CI = [{np.percentile(cv_errors, 2.5):.2f}%, {np.percentile(cv_errors, 97.5):.2f}%]")

if cv_mape < 10:
    print(f"\n✓ PASS: CV MAPE < 10% — model generalizes")
else:
    print(f"\n✗ FAIL: CV MAPE > 10% — overfitting concern")

# ============================================================
# CRITICAL TEST 5: Systematic Bias Analysis
# ============================================================
print(f"\n{'='*80}")
print("【CRITICAL TEST 5: Systematic Bias Analysis】")
print(f"{'='*80}")

print(f"\nResidual analysis by dimension:")
for d_val in sorted(set(d.astype(int))):
    mask = d.astype(int) == d_val
    mean_res = np.mean(residuals[mask])
    std_res = np.std(residuals[mask])
    print(f"  d={d_val}: mean residual = {mean_res:+.2f}%, std = {std_res:.2f}%, n={np.sum(mask)}")

# Check for systematic bias
print(f"\nSystematic bias check:")
bias_2d = np.mean(residuals[d == 2])
bias_3d = np.mean(residuals[d == 3])
bias_high = np.mean(residuals[(z == 2*d) & (d >= 8)])

print(f"  d=2 bias: {bias_2d:+.2f}%")
print(f"  d=3 bias: {bias_3d:+.2f}%")
print(f"  d≥8 bias: {bias_high:+.2f}%")

if abs(bias_3d) > 5:
    print(f"\n⚠ WARNING: d=3 systematic bias = {bias_3d:+.2f}% exceeds 5% threshold")
    print(f"  This suggests the model may not be universal at d=3")

# ============================================================
# CRITICAL TEST 6: Data Quality and Coverage
# ============================================================
print(f"\n{'='*80}")
print("【CRITICAL TEST 6: Data Quality and Coverage】")
print(f"{'='*80}")

print(f"\nDataset composition:")
print(f"  Total lattices: {n}")
print(f"  d=2: {np.sum(d == 2)} ({np.sum(d == 2)/n*100:.1f}%)")
print(f"  d=3: {np.sum(d == 3)} ({np.sum(d == 3)/n*100:.1f}%)")
print(f"  d≥4: {np.sum(d >= 4)} ({np.sum(d >= 4)/n*100:.1f}%)")

print(f"\nGirth coverage:")
for g_val in sorted(set(g.astype(int))):
    print(f"  g={g_val}: {np.sum(g == g_val)} lattices")

print(f"\n⚠ LIMITATION: Only 3 girth values (3, 4, 6) in dataset")
print(f"  Cannot validate formula for g=5, g=8, or other girth values")

# ============================================================
# CRITICAL TEST 7: Novelty Assessment
# ============================================================
print(f"\n{'='*80}")
print("【CRITICAL TEST 7: Novelty Assessment】")
print(f"{'='*80}")

print(f"\nWhat is claimed as novel:")
print(f"  1. Exponential dimensional decay (vs power-law)")
print(f"  2. Characteristic dimension d* ≈ 3.0")
print(f"  3. Universal saturation scale z₀ ≈ 10")

print(f"\nCritical questions:")
print(f"  1. Has exponential decay been proposed before?")
print(f"     → Need literature search to verify novelty")
print(f"  2. Is d* ≈ 3.0 physically meaningful or just a fitting parameter?")
print(f"     → Connection to d_c = 6 is qualitative, not derived")
print(f"  3. Is z₀ ≈ 10 universal or specific to this dataset?")
print(f"     → Limited by dataset composition")

# ============================================================
# CRITICAL TEST 8: Comparison to PRL Standards
# ============================================================
print(f"\n{'='*80}")
print("【CRITICAL TEST 8: PRL Standards Comparison】")
print(f"{'='*80}")

print(f"\nPRL requires:")
print(f"  ✓ Novel and significant results")
print(f"  ✓ Broad interest to physics community")
print(f"  ✓ Rigorous methodology")
print(f"  ✓ Clear advance over prior work")
print(f"  ✓ Strong evidence supporting claims")

print(f"\nAssessment:")
print(f"\n1. NOVELTY:")
print(f"   • Exponential vs power-law: Potentially novel if not done before")
print(f"   • 4-parameter formula: Incremental improvement over prior fits")
print(f"   • Physical interpretation: Qualitative, not first-principles")
print(f"   VERDICT: ⚠ MARGINAL — empirical finding, not theoretical breakthrough")

print(f"\n2. SIGNIFICANCE:")
print(f"   • High-dimensional improvement: 20% → 3% is substantial")
print(f"   • Practical utility: Enables rapid threshold estimation")
print(f"   • Physical insight: Limited — empirical correlation")
print(f"   VERDICT: ⚠ MODERATE — useful but not transformative")

print(f"\n3. RIGOR:")
print(f"   • Systematic model comparison: ✓ (12 variants tested)")
print(f"   • Cross-validation: ✓ (10-fold CV)")
print(f"   • Bootstrap stability: ✓ (100 iterations)")
print(f"   • Statistical tests: ✓ (paired t-test, AIC/BIC)")
print(f"   VERDICT: ✓ GOOD — methodology is sound")

print(f"\n4. EVIDENCE STRENGTH:")
print(f"   • Dataset size: 94 lattices (adequate)")
print(f"   • Dimension range: d=2-13 (good coverage)")
print(f"   • Girth range: g∈{3,4,6} (limited)")
print(f"   • Precision: MAPE 8.1% (acceptable but not exceptional)")
print(f"   • Systematic bias: d=3 bias -5.2% (concerning)")
print(f"   VERDICT: ⚠ MODERATE — evidence adequate but not compelling")

# ============================================================
# FINAL VERDICT
# ============================================================
print(f"\n{'='*80}")
print("【FINAL PRL ASSESSMENT】")
print(f"{'='*80}")

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRL READINESS VERDICT                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STATUS: ✗ NOT READY FOR PRL                                                │
│                                                                             │
│  REASONING:                                                                 │
│                                                                             │
│  1. PRECISION INSUFFICIENT:                                                 │
│     • MAPE = 8.1% is acceptable but not exceptional                         │
│     • 11% of lattices exceed 15% error                                      │
│     • PRL typically requires higher precision for empirical formulas        │
│                                                                             │
│  2. THEORETICAL DEPTH LACKING:                                              │
│     • Purely empirical — no first-principles derivation                     │
│     • Physical interpretation is qualitative, not quantitative              │
│     • "d* ≈ 3.0 consistent with d_c = 6" is hand-waving, not proof        │
│     • PRL values theoretical insight over curve-fitting                     │
│                                                                             │
│  3. SYSTEMATIC BIAS CONCERNING:                                             │
│     • d=3 bias of -5.2% suggests model not universal                        │
│     • sq-series residuals indicate data quality issues                      │
│     • Raises questions about generalizability                               │
│                                                                             │
│  4. LIMITED GIRTH COVERAGE:                                                 │
│     • Only 3 girth values tested (3, 4, 6)                                  │
│     • Cannot claim universality without broader validation                  │
│                                                                             │
│  5. NOVELTY QUESTIONABLE:                                                   │
│     • Exponential vs power-law may not be novel                             │
│     • Need literature search to verify this hasn't been done                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              WHAT WOULD MAKE IT PRL-LEVEL                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PATH 1: THEORETICAL DERIVATION                                             │
│     • Derive exponential form from first principles (cluster expansion,     │
│       lattice Green's functions, or renormalization group)                  │
│     • Show α = f(d_c) analytically, not just empirically                    │
│     • This would transform empirical fit into theoretical result            │
│                                                                             │
│  PATH 2: DRAMATICALLY IMPROVE PRECISION                                     │
│     • Achieve MAPE < 3% across all lattices                                 │
│     • Eliminate systematic biases (d=3, sq-series)                          │
│     • Extend to girth values beyond {3,4,6}                                 │
│                                                                             │
│  PATH 3: REFRAME AS DATA PAPER                                              │
│     • Submit to Physical Review E or Scientific Data instead                │
│     • Focus on dataset curation and benchmark value                         │
│     • Emphasize practical utility over theoretical depth                    │
│                                                                             │
│  PATH 4: PUBLISH IN SPECIALIZED JOURNAL                                     │
│     • Journal of Statistical Physics                                        │
│     • Physical Review E (Statistical Physics)                               │
│     • These venues value empirical formulas more than PRL                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# RECOMMENDATIONS
# ============================================================
print(f"\n{'='*80}")
print("【RECOMMENDATIONS】")
print(f"{'='*80}")

print(f"""
IMMEDIATE ACTIONS:

1. LITERATURE SEARCH (CRITICAL):
   • Search for prior work on exponential dimensional decay in percolation
   • Verify that exponential form hasn't been proposed before
   • If it has been done, the novelty claim is invalidated

2. ADDRESS SYSTEMATIC BIASES:
   • Investigate d=3 bias — is it model limitation or data quality?
   • Check sq-series Monte Carlo estimates for systematic errors
   • If data quality issue, exclude problematic lattices and refit

3. EXTEND GIRTH VALIDATION:
   • Simulate or collect data for g=5, g=8 lattices
   • Test whether formula generalizes beyond {3,4,6}

4. CONSIDER ALTERNATIVE VENUES:
   • Physical Review E: More appropriate for statistical physics
   • Journal of Statistical Physics: Values empirical formulas
   • Scientific Data: If reframing as dataset contribution

LONG-TERM STRATEGY:

If pursuing PRL:
   • Attempt first-principles derivation of exponential form
   • Connect α to d_c analytically (not just qualitatively)
   • Achieve MAPE < 5% with no systematic biases
   • Demonstrate predictive power on unseen lattice types

Alternative:
   • Publish in PRE/JSTAT as solid empirical result
   • Build reputation in percolation field
   • Return to PRL with stronger theoretical foundation later
""")

print(f"\n{'='*80}")
print("【Assessment Complete】")
print(f"{'='*80}")
