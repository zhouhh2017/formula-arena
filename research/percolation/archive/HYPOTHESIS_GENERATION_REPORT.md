# PRL Readiness Assessment: Hypothesis Generation Report

**Date**: 2026-10-04  
**Project**: Percolation Threshold Formula  
**Assessment Type**: Critical evaluation for PRL submission readiness

---

## Executive Summary

This report presents a rigorous critical assessment of whether the percolation threshold formula work meets Physical Review Letters (PRL) standards. Through systematic hypothesis generation and statistical testing, we have identified **critical weaknesses** that prevent PRL acceptance in the current form.

**Core Finding**: The exponential vs power-law model comparison—central to the manuscript's novelty claim—is **not statistically significant** (p=0.253, ΔAIC=0.86). The two models are indistinguishable given the dataset size and composition.

**Recommendation**: **Do NOT submit to PRL**. Target Physical Review E (Statistical Physics) or Journal of Statistical Physics instead.

---

## Generated Hypotheses

### H1: Exponential Form Captures Physical Crossover

**Claim**: The exponential decay e^{-αd} genuinely captures the approach to the upper critical dimension d_c = 6, providing a physical explanation for the dimensional correction.

**Evidence For**:
- High-dimensional error is low (2.8% for d≥8)
- Characteristic dimension d* = 1/α ≈ 3.0, suggesting 2d* ≈ 6 ≈ d_c
- Physically motivated by loop probability arguments

**Evidence Against**:
- **Paired t-test: p = 0.253** (not significant at α=0.05)
- **ΔAIC = 0.86** (below threshold of 2 for "positive evidence")
- ΔMAPE = 0.60% (tiny absolute difference)
- The "consistency" with d_c = 6 is post-hoc pattern matching, not derivation
- d* = 3.0 ≠ d_c = 6; the connection is hand-waving

**Verdict**: **UNPROVEN**  
The exponential form fits well, but the data does not support distinguishing it from power-law decay. The claimed physical insight is not statistically validated.

---

### H2: Exponential vs Power-Law Difference Is Statistical Noise

**Claim**: The two functional forms are effectively equivalent given the dataset size, noise level, and composition.

**Evidence For**:
- Paired t-test: **p = 0.253** (fail to reject null hypothesis)
- ΔAIC = 0.86 < 2 (below "positive evidence" threshold)
- ΔMAPE = 0.60% (practically negligible)
- 67% of data is d=2, where both models perform identically
- Only 6 high-dimensional data points drive the claimed advantage
- Bootstrap CV for α is 14.8%, indicating substantial uncertainty

**Evidence Against**:
- High-dimensional subset shows 7× improvement (20% → 3%)
- Exponential model has better AIC/BIC (though not significantly)

**Verdict**: **LIKELY CORRECT**  
The data does not support the claim that exponential "uniformly outperforms" power-law. The "factor of 7 improvement" is a small-n subgroup result, not a robust finding.

---

### H3: Formula Is Useful Empirical Tool, Not PRL Breakthrough

**Claim**: The 4-parameter formula is a practical contribution suitable for a specialized statistical physics journal, but does not constitute a PRL-level theoretical advance.

**Evidence For**:
- Systematic model comparison (12 variants tested)
- Cross-validation and bootstrap analysis demonstrate rigor
- Practical utility for rapid threshold estimation
- Sound methodology and transparent limitations discussion
- MAPE 8.1% is acceptable for empirical formula

**Evidence Against**:
- No theoretical derivation—purely empirical curve fitting
- Systematic biases remain (d=3 bias +5.17%)
- Limited girth coverage (only g ∈ {3, 4, 6})
- 30.9% of lattices have error >10%

**Verdict**: **CORRECT**  
This is solid statistical physics work, appropriate for PRE or JSTAT. The methodology is sound, but the contribution is empirical, not theoretical.

---

### H4: Work Is Appropriate for Physical Review E

**Claim**: The manuscript should be submitted to PRE (Statistical Physics) rather than PRL, as PRE values systematic empirical work and has lower theoretical novelty requirements.

**Evidence For**:
- PRE audience is specifically interested in percolation/statistical physics
- Systematic model comparison methodology is valued in PRE
- Lower bar for theoretical novelty
- Can frame as "comprehensive empirical study" rather than "theoretical breakthrough"
- Reviewers will appreciate honest discussion of limitations

**Evidence Against**:
- Even for PRE, the statistical insignificance of the core claim needs to be addressed
- Systematic biases should be investigated or acknowledged

**Verdict**: **RECOMMENDED PATH**  
PRE is the appropriate venue. The work can be published as a solid empirical contribution after addressing statistical significance issues.

---

## Critical Test Results

### Test 1: Precision Assessment
```
MAPE = 8.14%
Median error = 7.06%
95th percentile = 19.12%
Lattices >10% error: 29/94 (30.9%)
Lattices >15% error: 10/94 (10.6%)
```
**Verdict**: MARGINAL — acceptable but not exceptional for PRL

### Test 2: High-Dimensional Claim
```
d≥8 hypercubic: 6 lattices
MAPE = 2.79%
Range: 0.08% (d=11) to 7.13% (d=8)
```
**Verdict**: PASS for subset, but small-n (N=6)

### Test 3: Statistical Significance (CRITICAL)
```
Exponential MAPE: 8.14%
Power-law MAPE:    8.73%
ΔAIC = 0.86
ΔBIC = 0.86
Paired t-test: t = 1.150, p = 0.253
```
**Verdict**: **FAIL** — Not statistically significant

### Test 4: Cross-Validation
```
10-fold CV MAPE = 8.83% ± 7.37%
95% CI = [0.68%, 27.86%]
```
**Verdict**: PASS — Model generalizes, but high variance

### Test 5: Systematic Bias
```
d=2 bias:  +0.17%
d=3 bias:  +5.17%  ← exceeds 5% threshold
d≥8 bias: +1.65%
```
**Verdict**: WARNING — d=3 systematic bias suggests model not universal

### Test 6: Data Coverage
```
Total: 94 lattices
d=2: 63 (67.0%)
d=3: 18 (19.1%)
d≥4: 13 (13.8%)
Girth: only g ∈ {3, 4, 6}
```
**Verdict**: LIMITED — Dataset dominated by 2D, girth coverage insufficient

---

## PRL Standards Comparison

PRL requires **all** of the following:

| Criterion | Required | This Work | Status |
|-----------|----------|-----------|--------|
| Novel and significant | ✓ | Incremental empirical improvement | ⚠ MARGINAL |
| Broad interest | ✓ | Limited to statistical physics | ⚠ MARGINAL |
| Theoretical insight | ✓ | None (purely empirical) | ✗ FAIL |
| Rigorous methodology | ✓ | Sound (CV, bootstrap, model comparison) | ✓ PASS |
| Strong evidence | ✓ | Core claim not statistically significant | ✗ FAIL |

**Overall**: Does NOT meet PRL standards

---

## Recommended Paths Forward

### Path A: Publish in PRE Now (RECOMMENDED)

**Actions**:
1. **Weaken the core claim**: Acknowledge that exponential and power-law models are statistically indistinguishable
2. **Reframe the narrative**: "We find that both exponential and power-law forms provide comparable fits, with exponential showing slightly better high-dimensional performance"
3. **Address d=3 bias**: Investigate or acknowledge as limitation
4. **Emphasize strengths**: Systematic model comparison, practical utility, comprehensive dataset
5. **Submit to Physical Review E** (Statistical Physics section)

**Timeline**: 2-4 weeks for revisions, 3-6 months for review

**Success Probability**: HIGH (80-90%)

---

### Path B: Theoretical Derivation (HIGH RISK)

**Actions**:
1. Derive exponential form from first principles:
   - Cluster expansion techniques
   - Lattice Green's functions
   - Renormalization group arguments
   - Random walk theory on lattices
2. Show α = f(d_c) analytically, not just empirically
3. Predict parameter values from theory
4. Submit to PRL as theoretical + empirical result

**Timeline**: 6-12 months (may fail)

**Success Probability**: LOW-MODERATE (20-40%)

**If successful**: Strong PRL paper  
**If failed**: Still have PRE paper from Path A

---

### Path C: Extend Dataset (MODERATE RISK)

**Actions**:
1. Run Monte Carlo simulations for g = 5, 8 lattices
2. Extend to d = 14, 15, 16 hypercubic lattices
3. Re-fit models with larger dataset (N > 150)
4. Test whether exponential becomes statistically distinguishable
5. If significant, submit to PRL with stronger evidence

**Timeline**: 3-6 months for simulations + analysis

**Success Probability**: MODERATE (40-60%)

**Risk**: May still not achieve statistical significance even with more data

---

## Claim-Evidence Audit

| Claim in Manuscript | Evidence Provided | Status |
|---------------------|-------------------|--------|
| "Exponential uniformly outperforms power-law" | ΔAIC=0.86, p=0.253 | **NOT SUPPORTED** |
| "Factor of 7 improvement at d≥8" | 6 data points, small-n | **MISLEADING** |
| "MAPE = 8.1% across 94 lattices" | Direct calculation | ✓ SUPPORTED |
| "d* ≈ 3.0 consistent with d_c = 6" | Qualitative pattern match | **NOT DERIVED** |
| "z₀ ≈ 10 is universal" | Same parameter fits d=2 and d=13 | PARTIALLY SUPPORTED |
| "Model generalizes" | 10-fold CV MAPE = 8.83% | ✓ SUPPORTED |
| "Formula is practically useful" | Rapid estimation capability | ✓ SUPPORTED |
| "Model is universal" | d=3 bias = +5.17% | **NOT SUPPORTED** |

**Critical Issues**: 2 of 8 claims are NOT SUPPORTED or MISLEADING

---

## Visualizations

Two schematic figures have been generated to illustrate the assessment:

1. **PRL_assessment_schematic.png**: Competing hypotheses framework and decision tree
2. **statistical_comparison.png**: Statistical comparison of exponential vs power-law models

These figures clearly show:
- The statistical insignificance of the core claim
- The decision framework for venue selection
- The paired error comparison revealing model indistinguishability

---

## Conclusion

**This work does NOT meet PRL standards in its current form.**

The primary reasons:
1. **Core claim not statistically significant** (p=0.253, ΔAIC=0.86)
2. **No theoretical derivation** — purely empirical curve fitting
3. **Systematic biases** suggest model is not universal
4. **Limited data coverage** (especially girth values)

**Recommended action**: Submit to Physical Review E after:
- Acknowledging statistical indistinguishability of exponential and power-law
- Addressing or discussing d=3 systematic bias
- Reframing as systematic empirical contribution

**Alternative**: If theoretical derivation is achievable (Path B), this could become PRL material in 6-12 months, but this is high-risk.

---

## Appendix: Statistical Methods

### Paired t-test
- **Null hypothesis**: No difference between exponential and power-law errors
- **Alternative**: Exponential errors are smaller
- **Result**: t = 1.150, p = 0.253 (one-tailed)
- **Interpretation**: Fail to reject null hypothesis; models are statistically indistinguishable

### AIC/BIC Comparison
- **ΔAIC threshold**: >2 for "positive evidence", >10 for "strong evidence"
- **Observed ΔAIC**: 0.86
- **Interpretation**: No positive evidence for exponential over power-law

### Effect Size
- **ΔMAPE**: 0.60% (8.73% → 8.14%)
- **Interpretation**: Small absolute difference, may not be practically significant

---

**Report generated**: 2026-10-04  
**Assessment confidence**: HIGH (based on direct statistical analysis)  
**Next steps**: Revise manuscript for PRE submission
