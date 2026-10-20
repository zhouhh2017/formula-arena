# PRL Readiness Assessment — Hypothesis Generation Report

> **Date**: 2026-10-04
> **Manuscript**: Exponential dimensional decay in site percolation thresholds (v4)
> **Assessment type**: Independent critical evaluation

---

## Executive Summary

**Verdict: ✗ NOT READY FOR PRL**

The work presents a solid empirical finding — a 4-parameter formula for site percolation thresholds — but falls short of PRL standards on multiple critical dimensions:

| Criterion | Status | Issue |
|-----------|--------|-------|
| Precision | ⚠ Marginal | MAPE 8.1%, 11% of lattices >15% error |
| Statistical significance | ✗ Fail | Exponential vs power-law: **p=0.25, ΔAIC=0.86** |
| Theoretical depth | ✗ Fail | Purely empirical, no first-principles derivation |
| Novelty | ⚠ Unverified | Exponential form may not be novel |
| Systematic bias | ⚠ Concerning | d=3 bias +5.17%, not universal |
| Data coverage | ⚠ Limited | Only 3 girth values (3,4,6) |

**Most damaging finding**: The core claim — exponential decay "uniformly outperforms" power-law — is **not statistically significant**. A paired t-test gives p=0.253, and ΔAIC=0.86 is below the threshold of 2 for even "positive evidence." The "factor of 7 improvement" at d≥8 is driven by only 6 data points.

---

## Critical Test Results

### Test 1: Precision

```
MAPE = 8.14%
Median error = 7.06%
Max error = 28.87%
95th percentile = 19.12%
Lattices >10% error: 29/94 (30.9%)
Lattices >15% error: 10/94 (10.6%)
Lattices >20% error: 5/94 (5.3%)
```

**Verdict**: ⚠ Marginal. PRL typically requires higher precision for empirical formulas.

### Test 2: High-Dimensional Claim

```
d≥8 hypercubic: 6 lattices
MAPE = 2.79%
```

| d | z | p_c (measured) | p_c (predicted) | Error |
|---|---|---------------|-----------------|-------|
| 8 | 16 | 0.07521 | 0.08058 | 7.13% |
| 9 | 18 | 0.06521 | 0.06797 | 4.24% |
| 10 | 20 | 0.05759 | 0.05868 | 1.88% |
| 11 | 22 | 0.05159 | 0.05163 | **0.08%** |
| 12 | 24 | 0.04673 | 0.04615 | 1.25% |
| 13 | 26 | 0.04272 | 0.04178 | 2.18% |

**Verdict**: ✓ Pass for this subset, but based on only 6 data points.

### Test 3: Exponential vs Power-Law — THE CRITICAL TEST

```
Exponential model:  AIC = −668.30,  MAPE = 8.14%
Power-law model:    AIC = −667.43,  MAPE = 8.73%

ΔAIC = 0.86  (threshold for "positive evidence" = 2)
ΔBIC = 0.86
ΔMAPE = 0.60%

Paired t-test: t = 1.150, p = 0.2531
```

**Verdict**: ✗ **FAIL**. The difference is NOT statistically significant. The two models are indistinguishable at conventional significance levels. The manuscript's claim that exponential "uniformly outperforms" power-law is **not supported by the data**.

### Test 4: Cross-Validation

```
10-fold CV MAPE = 8.83% ± 7.37%
95% CI = [0.68%, 27.86%]
```

**Verdict**: ✓ Pass. Model generalizes, but high variance suggests sensitivity to data composition.

### Test 5: Systematic Bias

```
d=2 bias:  +0.17%
d=3 bias:  +5.17%  ← exceeds 5% threshold
d=4 bias:  +1.65%
d=5 bias:  −0.30%
d=6 bias:  −3.75%
```

**Verdict**: ⚠ The d=3 systematic bias suggests the model is not universal.

### Test 6: Data Coverage

```
Total lattices: 94
d=2: 63 (67.0%)
d=3: 18 (19.1%)
d≥4: 13 (13.8%)

Girth values: g ∈ {3, 4, 6} only
```

**Verdict**: ⚠ Dataset is dominated by 2D (67%). Girth coverage is extremely limited.

---

## Competing Hypotheses

### Hypothesis 1: Exponential Form Captures Physical Crossover

**Claim**: The exponential decay e^{−αd} genuinely captures the approach to the upper critical dimension d_c = 6.

**Evidence for**:
- High-dimensional error is low (2.8% for d≥8)
- α = 0.337 → d* ≈ 3.0 → 2d* ≈ 6 ≈ d_c (qualitative consistency)
- Physically motivated by loop probability arguments

**Evidence against**:
- ΔAIC = 0.86, p = 0.253 — **not statistically distinguishable** from power-law
- The "consistency" with d_c = 6 is post-hoc pattern matching
- d* = 3.0 ≠ d_c = 6; the connection is hand-waving

**Verdict**: ⚠ Unproven. The exponential form fits well but is not demonstrably superior.

### Hypothesis 2: Exponential vs Power-Law Difference Is Statistical Noise

**Claim**: The two models are effectively equivalent given the dataset size and noise level.

**Evidence for**:
- Paired t-test: p = 0.253 (not significant)
- ΔAIC = 0.86 < 2 (below "positive evidence" threshold)
- ΔMAPE = 0.60% — tiny absolute difference
- 67% of data is d=2, where both models perform identically
- Only 6 high-dimensional data points drive the claimed advantage

**Evidence against**:
- High-dimensional subset shows 7× improvement (20% → 3%)
- Bootstrap stability is reasonable (CV = 14.8% for α)

**Verdict**: ✓ **Likely correct**. The data does not support distinguishing the two models. The "exponential outperforms" narrative is driven by small-n subgroup analysis.

### Hypothesis 3: The Formula Is a Useful Empirical Tool, Not a PRL Breakthrough

**Claim**: The 4-parameter formula is a practical contribution suitable for a specialized journal, but not a PRL-level advance.

**Evidence for**:
- Systematic model comparison (12 variants)
- Cross-validation and bootstrap analysis
- Practical utility for rapid threshold estimation
- Sound methodology

**Evidence against**:
- No theoretical derivation — purely empirical curve fitting
- MAPE 8.1% is acceptable but not exceptional
- Systematic biases remain unexplained
- Limited girth coverage

**Verdict**: ✓ **Correct**. This is solid statistical physics, appropriate for PRE or JSTAT, but not PRL material.

### Hypothesis 4: The Work Is Appropriate for Physical Review E

**Claim**: The manuscript should be submitted to PRE (Statistical Physics) rather than PRL.

**Evidence for**:
- PRE values systematic empirical work
- The audience is specifically interested in percolation/statistical physics
- Lower bar for theoretical novelty
- Methodological rigor is adequate for PRE

**Evidence against**:
- The "exponential vs power-law" claim may still be seen as insufficiently supported
- Systematic biases need to be addressed regardless of venue

**Verdict**: ✓ **Recommended path**. PRE is the appropriate venue for this work.

---

## Critical Issues Requiring Resolution

### Issue 1: Statistical Significance (BLOCKING)

The core claim — exponential outperforms power-law — is not statistically significant.

**Required action**: Either:
1. Acknowledge the models are statistically indistinguishable (weaken the claim)
2. Collect more high-dimensional data to increase statistical power
3. Find a different distinguishing test

### Issue 2: Theoretical Depth (BLOCKING for PRL)

No first-principles derivation of the exponential form.

**Required action for PRL**: Derive the exponential form from:
- Cluster expansion techniques
- Lattice Green's functions
- Renormalization group arguments
- Random walk theory on lattices

### Issue 3: Systematic Bias at d=3

The d=3 bias of +5.17% is concerning.

**Required action**: Investigate whether this is:
- Model limitation (exponential form is wrong at d=3)
- Data quality issue (Monte Carlo estimates are biased)
- Missing physics (d=3 is special for some reason)

### Issue 4: Limited Girth Coverage

Only g ∈ {3, 4, 6} tested.

**Required action**: Simulate or collect data for g = 5, 8 to test generalizability.

---

## Recommended Paths Forward

### Path A: Publish in PRE Now (Recommended)

**Action**: Submit to Physical Review E (Statistical Physics)

**Steps**:
1. Weaken the "exponential outperforms" claim — acknowledge statistical indistinguishability
2. Reframe as "we find that both exponential and power-law forms provide comparable fits"
3. Address d=3 bias — investigate or acknowledge as limitation
4. Emphasize practical utility and systematic model comparison
5. Submit to PRE

**Timeline**: 2-4 weeks for revisions, 3-6 months for review

**Expected outcome**: Published paper, solid contribution to statistical physics

### Path B: Attempt Theoretical Derivation (High Risk, High Reward)

**Action**: Derive the exponential form from first principles, then submit to PRL

**Steps**:
1. Study cluster expansion for percolation thresholds
2. Calculate loop corrections as function of dimension
3. Show that exponential decay emerges from theory
4. Connect α to d_c analytically
5. If successful, submit to PRL as theoretical + empirical result

**Timeline**: 6-12 months (may fail)

**Expected outcome**: If successful, strong PRL paper. If failed, still have PRE paper.

### Path C: Extend Dataset and Improve Precision

**Action**: Collect more data, especially high-dimensional and diverse girth values

**Steps**:
1. Run Monte Carlo simulations for g = 5, 8 lattices
2. Extend to d = 14, 15, 16 hypercubic lattices
3. Re-fit models with larger dataset
4. Test whether exponential becomes statistically distinguishable
5. If so, submit to PRL with stronger evidence

**Timeline**: 3-6 months for simulations + analysis

**Expected outcome**: Stronger evidence base, but may still not achieve PRL-level theoretical depth

---

## Claim-Evidence Map

| Claim | Evidence | Status |
|-------|----------|--------|
| Exponential outperforms power-law | ΔAIC = 0.86, p = 0.253 | **NOT SUPPORTED** |
| MAPE = 8.1% across 94 lattices | Direct calculation | **SUPPORTED** |
| d≥8 error = 2.8% | 6 data points | SUPPORTED but small-n |
| d* ≈ 3.0 consistent with d_c = 6 | Qualitative pattern match | **NOT DERIVED** |
| z₀ ≈ 10 is universal | Same parameter fits d=2 and d=13 | PARTIALLY SUPPORTED |
| Model generalizes | 10-fold CV MAPE = 8.83% | SUPPORTED |
| Formula is useful | Rapid estimation capability | SUPPORTED |
| Model is universal | d=3 bias = +5.17% | **NOT SUPPORTED** |

---

## Final Recommendations

### For PRL Submission:

**Do not submit in current form.** The statistical evidence for the core claim is insufficient, and the theoretical depth is lacking.

**If pursuing PRL**:
1. Derive exponential form from first principles (6-12 months)
2. OR collect much more high-dimensional data to achieve statistical significance
3. Address systematic biases
4. Extend girth coverage

### For Publication:

**Submit to Physical Review E** after:
1. Weakening the "exponential outperforms" claim
2. Acknowledging statistical indistinguishability from power-law
3. Addressing or acknowledging d=3 bias
4. Reframing as systematic empirical contribution

**Alternative venues**:
- Journal of Statistical Physics
- Physical Review E (Statistical Physics)
- Scientific Data (if reframing as dataset paper)

---

## Conclusion

This work represents solid empirical research in statistical physics. The methodology is sound, the dataset is substantial, and the formula has practical utility. However, it does **not** meet PRL standards because:

1. The core claim (exponential > power-law) is not statistically significant
2. There is no theoretical derivation — it is curve fitting
3. Systematic biases suggest the model is not universal
4. Data coverage is limited (especially girth values)

The appropriate venue is Physical Review E, not PRL. To reach PRL level, the work would need either a first-principles derivation of the exponential form or dramatically stronger statistical evidence from a much larger dataset.

---

**Assessment completed**: 2026-10-04
**Assessment method**: Statistical testing, model comparison, PRL standards evaluation
**Confidence level**: High — based on direct statistical analysis of the data
