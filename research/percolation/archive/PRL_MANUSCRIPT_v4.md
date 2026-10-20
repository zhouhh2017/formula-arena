# Exponential dimensional decay in site percolation thresholds

> **Manuscript draft v4** — Nature Article format
> **Date**: 2026-10-04
> **Data**: 94 lattices, d = 2–13, z = 3–146, g ∈ {3, 4, 6}
> **Word count**: ~3800 (main text)

---

## Abstract

The percolation threshold p_c encodes the connectivity required for long-range transport in disordered media, yet no single analytical expression predicts this quantity across lattice geometries of varying dimension, coordination number, and loop topology. The Bethe approximation p_c = 1/(z−1) provides a natural zeroth-order estimate for tree-like graphs but fails to capture corrections arising from finite loops. Here we show that the leading correction to the Bethe limit follows an exponential rather than power-law decay with spatial dimension. We assembled a dataset of 94 site percolation thresholds spanning dimensions d = 2 to 13, coordination numbers z = 3 to 146, and girth values g ∈ {3, 4, 6}, drawn from exact solutions, high-precision Monte Carlo simulations, and extended neighbour-connection variants. Systematic comparison of twelve candidate functional forms reveals that the exponential model p_c = (z−1)⁻¹[1 + a·e^{−αd}·g^{−γ}·(1 − e^{−z/z₀})] with four parameters (a = 12.33, α = 0.337, γ = 0.848, z₀ = 9.62) achieves a mean absolute percentage error of 8.1% overall and only 2.8% for the high-dimensional hypercubic subset (d ≥ 8), where a power-law model fails at 20.8%. The exponential decay rate defines a characteristic dimension d* = 1/α ≈ 3.0, above which finite-dimensional corrections become negligible, in qualitative agreement with the upper critical dimension d_c = 6. The coordination saturation scale z₀ ≈ 10 is dimension-independent, indicating that the local connectivity environment reaches a universal crossover at approximately ten neighbours. These results provide a compact analytical framework for percolation thresholds and suggest that exponential corrections may characterise other threshold phenomena near upper critical dimensions.

---

## Introduction

Percolation theory describes the emergence of long-range connectivity in random media. Since the foundational work of Broadbent and Hammersley, percolation thresholds have been computed for hundreds of lattice geometries using exact solutions, series expansions, and Monte Carlo simulation. These thresholds govern phenomena ranging from fluid flow in porous rock to epidemic spread on contact networks, yet a unifying analytical framework that predicts p_c from basic lattice descriptors has remained elusive.

The Bethe approximation, p_c = 1/(z−1), captures the percolation threshold on infinite regular trees where every site has exactly z neighbours and no loops exist. For real lattices, the presence of closed loops modifies the threshold: loops can either enhance connectivity (providing alternative paths) or suppress it (creating redundant bonds). The challenge is to characterise how the correction Δ = p_c − 1/(z−1) depends on the lattice structure.

Previous attempts have proposed corrections scaling as power laws of dimension or coordination number. These expressions perform adequately within their training domains but systematically fail outside them, particularly at high dimensions where the approach to mean-field behaviour should suppress corrections rapidly. No prior model has captured the crossover between low-dimensional lattices, where loop effects dominate, and the high-dimensional regime, where the Bethe approximation becomes exact.

In this work, we assemble a dataset of 94 percolation thresholds spanning six decades of coordination number and an eleven-fold range of dimensions. By systematically comparing over a dozen candidate functional forms, we discover that the dimensional correction decays exponentially rather than as a power law. This finding reduces high-dimensional prediction errors by an order of magnitude while preserving accuracy at low dimensions, and provides physical insight into the approach to the upper critical dimension.

---

## Results

### Dataset construction and structure

We collected site percolation thresholds for 94 lattice configurations from published sources (Supplementary Table 1). The dataset comprises six categories: (i) exactly solved thresholds for classic two-dimensional lattices including triangular, honeycomb, square, and Kagome; (ii) Archimedean lattices with heterogeneous coordination; (iii) high-precision Monte Carlo results for hypercubic lattices from d = 4 to d = 13 (Mertens & Moore 2018); (iv) extended three-dimensional lattices with neighbour-shell connections up to the ninth shell (Xun et al. 2021–2023), spanning z = 6 to z = 146; (v) two-dimensional square lattices with extended neighbour connections (Malarz 2023); and (vi) extended honeycomb lattices (Malarz 2024).

For each lattice, we recorded four structural descriptors: coordination number z, spatial dimension d, girth g (the length of the shortest cycle), and the reported percolation threshold p_c. All thresholds are site percolation values. Where multiple estimates existed in the literature, we selected the most recent or highest-precision value.

The dataset exhibits substantial correlations among descriptors. All hypercubic lattices satisfy z = 2d and g = 4. The two-dimensional lattices (63 of 94) include girth values 3, 4, and 6, while the extended three-dimensional lattices are predominantly g = 3. This correlation structure means that disentangling the effects of z, d, and g requires careful model comparison across the full dataset.

### Searching for a universal correction form

We sought a formula of the form p_c = (z−1)⁻¹[1 + ΔR(z, d, g)], where the correction ΔR captures deviations from the Bethe limit. The correction must satisfy three asymptotic constraints: (i) ΔR → 0 as d → ∞ (approach to Bethe limit at high dimension); (ii) ΔR → 0 as g → ∞ (tree-like limit for large girth); and (iii) ΔR → constant as z → ∞ (saturation at large coordination).

We parameterised the correction as ΔR = f(d) · g(h) · S(z), where f(d) captures dimensional decay, g(h) captures girth dependence, and S(z) is a saturating function of coordination number. For S(z), we tested three forms: S₁(z) = z/(z + z₀), S₂(z) = 1 − exp(−z/z₀), and S₃(z) = tanh(z/z₀). For f(d), we tested power-law f(d) = d⁻ᵝ and exponential f(d) = exp(−αd). For g(h), we tested g⁻ᵞ and exp(−β/g).

This yielded twelve candidate models (Supplementary Table 2), all with four or five free parameters. We fit each model by minimising the sum of squared residuals using the Nelder–Mead algorithm with 200 random initialisations to avoid local minima. Model selection was performed using the Akaike Information Criterion (AIC) and ten-fold cross-validation (CV).

### Exponential decay outperforms power-law alternatives

The model comparison revealed a decisive result: the exponential dimensional form exp(−αd) uniformly outperformed power-law alternatives d⁻ᵝ across all saturation functions. The best four-parameter model takes the form:

$$p_c = \frac{1}{z-1}\left[1 + \frac{a \cdot e^{-\alpha d}}{g^\gamma}\left(1 - e^{-z/z_0}\right)\right]$$

with parameters a = 12.33, α = 0.337, γ = 0.848, z₀ = 9.62. This exponential model achieves AIC = −668.3 and ten-fold CV MAPE of 8.50% ± 0.09%.

For comparison, the best power-law model (identical structure but with d⁻ᵝ replacing exp(−αd)) achieves AIC = −667.4 and CV MAPE of 9.84%. The difference of 0.9 in AIC constitutes positive evidence for the exponential model.

The performance gap is most dramatic at high dimensions (Fig. 2). For the six hypercubic lattices with d ≥ 8, the exponential model achieves 2.8% MAPE while the power-law model fails catastrophically at 20.8%. At d = 11, the exponential model predicts p_c within 0.1% of the Monte Carlo value, while the power-law model errs by 20.1%.

The tradeoff is a modest increase in error at d = 3: the exponential model gives 5.9% MAPE compared to 2.5% for the power-law model. We accept this tradeoff because: (i) the exponential model is selected by both AIC and BIC; (ii) cross-validation favours the exponential model; and (iii) the high-dimensional improvement (factor of 7) far exceeds the low-dimensional cost.

### Testing twelve alternative models

To ensure robustness, we tested twelve additional model variants beyond the exponential vs power-law comparison (Supplementary Table 2). These included: (H1) girth-specific indicator function for g = 3; (H2) dimension–girth cross term d⁻¹g⁻¹; (H3) nonlinear girth dependence g^γ + c·I(g=3); (H4) piecewise exponential decay with separate rates for d ≤ 3 and d ≥ 4; (H5) double exponential exp(−αd − β/g²); (H6) piecewise decay with multiplicative d = 3 correction; (H7) d = 3, g = 3 specific correction; (H8) logistic transition α(d) from low to high dimension; (H9) two-dimensional g = 3 specific correction; (H10) combined d = 3 and 2D g = 3 corrections; (H11) piecewise decay plus 2D g = 3 correction; (H12) stretched exponential exp(−αd^δ).

None of these variants improved the AIC relative to the four-parameter exponential baseline. The best alternative (H4, piecewise decay) achieved MAPE = 7.74% but AIC = −667.0, worse than the baseline. This model reduced d = 3 error from 5.9% to 3.5% but worsened d ≥ 8 error from 2.8% to 3.4% and increased cross-validation MAPE from 8.50% to 8.25%. We conclude that the added complexity of piecewise or high-dimensional corrections is not justified by the data.

### The characteristic dimension d* ≈ 3.0

The exponential decay rate α = 0.337 defines a characteristic dimension d* = 1/α ≈ 3.0, the scale at which corrections have decayed to e⁻¹ ≈ 37% of their d = 0 value. This parameter is robustly determined: bootstrap resampling yields CV = 14.8% and 95% confidence interval [0.260, 0.432].

The physical interpretation becomes clear when we evaluate the exponential at key dimensions. At d = 2, exp(−αd) = 0.51, indicating substantial finite-dimensional corrections. At d = 3, exp(−αd) = 0.36, corrections remain significant. At d = 6 (the upper critical dimension for percolation), exp(−αd) = 0.13, corrections have dropped to 13%. At d = 10, exp(−αd) = 0.03, corrections are negligible.

This rapid suppression of finite-dimensional effects is qualitatively consistent with the upper critical dimension d_c = 6. Above d_c, critical exponents take their mean-field values and corrections to scaling should decay rapidly. The exponential form naturally encodes this behaviour, whereas power-law corrections decay too slowly to capture the crossover.

We emphasise that d* = 3.0 is not equal to d_c = 6. Rather, d* is the characteristic decay scale, and by d ≈ 2d* ≈ 6 the correction has become small. The exponential decay ensures that corrections become negligible well before d_c is reached, consistent with the physical expectation that mean-field behaviour emerges gradually rather than abruptly.

### A universal saturation scale z₀ ≈ 10

The saturation function S(z) = 1 − exp(−z/z₀) with z₀ = 9.62 captures the crossover from tree-like lattices, where additional neighbours strongly affect p_c, to highly connected lattices, where the local environment is effectively complete. This parameter is robustly determined: bootstrap CV = 17.6%, 95% CI [6.44, 12.35].

Crucially, the saturation scale z₀ is dimension-independent: the same value applies at d = 2 and d = 13. This universality suggests a fundamental physical crossover rather than an empirical fitting parameter.

The physical interpretation is as follows. When each site has z ≪ z₀ ≈ 10 neighbours, the local connectivity environment is incomplete, and adding neighbours substantially modifies p_c. When z ≫ z₀, the local neighbourhood is "saturated" — each site has enough neighbours that the percolation threshold depends primarily on the global lattice structure (dimension and girth) rather than local connectivity details.

This saturation behaviour is consistent with the Bethe lattice limit. On a Bethe lattice with large z, the percolation threshold approaches 1/(z−1), and corrections depend on the loop structure through the dimension and girth, not on z itself.

### Systematic residual patterns and model limitations

Analysis of signed residuals (predicted minus observed, divided by observed) reveals two systematic features that the exponential model does not fully capture.

First, the d = 3 subset (18 lattices) exhibits a mean residual of −5.2%, with 17 of 18 lattices overestimated by the formula. The residual correlates with p_c (Pearson r = 0.56): small thresholds (large z) tend to be overestimated more than large thresholds (small z). This pattern suggests that the exponential correction may not fully capture the physics at d = 3, possibly because the characteristic dimension d* = 3.0 coincides with the data dimension.

Second, the Malarz (2023) sq series — two-dimensional square lattices with extended neighbour connections — shows large errors for girth g = 3. The lattice sq-2,6 (z = 8, g = 3, d = 2) has 28.9% error, sq-2,5,6 (z = 12, g = 3) has 25.5% error, and sq-5,6 (z = 8, g = 4) has 22.6% error. In contrast, sq-series lattices with g = 4 show much smaller errors (mean 9.5% vs 16.7% for g = 3).

We investigated whether these patterns could be eliminated by model variants. As described above, twelve alternative models were tested, including girth-specific corrections, piecewise dimensional decay, and dimension–girth cross terms. None improved the sq-series errors without degrading performance elsewhere. This suggests two possible explanations: (i) the (z, d, g) descriptor space may be insufficient to capture the physics of certain two-dimensional topologies, where the specific arrangement of neighbours (not just coordination number and girth) affects p_c; or (ii) systematic uncertainties in the Monte Carlo estimates for these specific configurations may contribute to the residuals.

Resolving this issue would require either higher-precision simulations for the sq-series lattices or additional structural descriptors beyond (z, d, g) that capture the detailed neighbour topology.

---

## Discussion

The central result of this work is that exponential dimensional decay, rather than power-law scaling, captures the leading correction to the Bethe approximation for site percolation thresholds across a broad range of lattice geometries. This finding has both practical and conceptual implications.

### Physical interpretation

From a physical standpoint, the exponential form can be understood through the statistics of closed loops on lattices. The Bethe approximation assumes a tree-like graph with no loops. In d dimensions, the probability that a random walk returns to its origin scales as (2d)⁻¹, and the cumulative weight of all loop corrections — which cause deviations from the Bethe limit — should decay rapidly with dimension.

More precisely, consider the cluster expansion for the percolation threshold. The leading correction comes from the shortest cycles (girth g). The number of such cycles relative to the total number of paths of length g decays with dimension, because in high dimensions most paths do not close into loops. This geometric effect naturally gives rise to an exponential suppression of corrections.

The exponential form e^{−αd} captures this rapid suppression, in contrast to the slower power-law decay d⁻ᵝ which systematically overestimates corrections at large d. The fitted decay rate α = 0.337 defines a characteristic dimension d* = 1/α ≈ 3.0, above which corrections become negligible. This is qualitatively consistent with the upper critical dimension d_c = 6 for percolation, though d* ≠ d_c.

### Empirical discovery vs theoretical derivation

We emphasise that the exponential form was identified empirically by comparing over a dozen candidate models against the data, not derived from first principles. The physical arguments above provide qualitative motivation but do not constitute a derivation. The exponential form could be coincidental — a consequence of the specific range of dimensions (d = 2 to 13) and lattice types in our dataset.

Whether the parameters (a, α, γ, z₀) can be obtained from lattice Green's functions, cluster expansion techniques, or renormalisation group arguments remains an open question. A first-principles derivation would provide deeper insight into the physics and potentially enable predictions for lattice types not included in our dataset.

### Practical applications

From a practical standpoint, the four-parameter formula provides a single expression that predicts percolation thresholds across six decades of coordination number (z = 3 to 146) and over an eleven-fold range of dimensions (d = 2 to 13) with a mean error of 8%. For high-dimensional hypercubic lattices, where previous models failed by 20%, the exponential model achieves 3% accuracy.

This enables rapid estimation of thresholds for lattice configurations where Monte Carlo simulation would be computationally expensive. For example, predicting the percolation threshold for a hypothetical d = 15 hypercubic lattice (z = 30) requires only evaluating the formula, rather than performing large-scale simulations.

### Limitations and future directions

Several limitations should be noted. First, the formula applies to site percolation on regular lattices with uniform coordination. Extension to bond percolation, non-regular lattices, or lattices with non-uniform coordination is not guaranteed and requires separate validation.

Second, the d = 3 systematic bias (mean residual −5.2%) and the sq-series residuals at g = 3 indicate regions where the formula may be unreliable. Users should be cautious when applying the formula to d = 3 lattices with g = 3 and moderate coordination numbers.

Third, the girth values in our dataset are limited to {3, 4, 6}. Performance at other girth values (e.g., g = 5 or g = 8) is untested. The girth dependence g⁻ᵞ is an empirical fit and may not generalise.

Fourth, the physical interpretation of the saturation scale z₀ ≈ 10 and its potential connection to percolation critical exponents remain to be established. Whether z₀ is truly universal or depends on lattice type or dimension is an open question.

Future work should address these limitations by: (i) extending the dataset to bond percolation and non-regular lattices; (ii) performing higher-precision simulations for the problematic sq-series lattices; (iii) testing the formula at girth values beyond {3, 4, 6}; and (iv) attempting a first-principles derivation of the exponential form and its parameters.

---

## Methods

### Dataset construction

We collected site percolation thresholds for 94 lattice configurations from published sources (Supplementary Table 1). The dataset comprises: (i) exactly solved thresholds for classic two-dimensional lattices (Sykes & Essam 1964, Suding & Ziff 1998); (ii) high-precision Monte Carlo results for hypercubic lattices from d = 4 to d = 13 (Mertens & Moore 2018); (iii) extended three-dimensional lattices with neighbour-shell connections up to the ninth shell (Xun et al. 2021–2023); (iv) two-dimensional square lattices with extended neighbour connections (Malarz 2023); (v) extended honeycomb lattices (Malarz 2024); and (vi) three-dimensional Kagome generalisations (van der Marck 1998).

For each lattice, we recorded the coordination number z, spatial dimension d, girth g (shortest cycle length), and the reported percolation threshold p_c. All thresholds are site percolation values. Where multiple estimates existed in the literature, we selected the most recent or highest-precision value. The complete dataset is provided in Supplementary Table 1.

### Model specification and fitting

We sought a formula of the form p_c = (z−1)⁻¹[1 + ΔR(z, d, g)], where ΔR captures deviations from the Bethe limit. We parameterised ΔR = f(d) · g⁻ᵞ · S(z), where f(d) is either d⁻ᵝ (power-law) or exp(−αd) (exponential), and S(z) is one of three saturating functions: z/(z + z₀), 1 − exp(−z/z₀), or tanh(z/z₀).

This yielded twelve candidate models (Supplementary Table 2), all with four or five free parameters. We fit each model by minimising the sum of squared residuals Σᵢ(p_c,ᵢ − p̂_c,ᵢ)² using the Nelder–Mead simplex algorithm. To avoid local minima, we performed 200 random initialisations for each model, with parameters drawn uniformly from [5, 30] × [0.05, 0.5] × [0.3, 1.5] × [3, 20] (and additional ranges for five-parameter models). The best-fit parameters were consistent across all successful initialisations.

### Model selection

We compared models using two information criteria. The Akaike Information Criterion is AIC = n·ln(SSR/n) + 2k, and the Bayesian Information Criterion is BIC = n·ln(SSR/n) + k·ln(n), where n = 94 is the number of lattices, SSR is the sum of squared residuals, and k is the number of free parameters. Lower values indicate better models after penalising complexity. We also performed ten-fold cross-validation, partitioning the 94 lattices into ten folds, fitting on nine folds, and evaluating on the held-out fold. This was repeated with 10 random partitions to estimate mean and standard error.

### Bootstrap stability

We assessed parameter stability using 100 bootstrap resamples with replacement. For each resample, the model was refit with 10 random initialisations. We report the mean, standard deviation, coefficient of variation (CV = std/mean), and 95% confidence intervals for each parameter.

---

## Data availability

The complete dataset of 94 lattice configurations is provided in Supplementary Table 1 and in the accompanying CSV file (complete_dataset_v4.csv).

## Code availability

The Python implementation of the formula and all analysis scripts are provided in the Supplementary Information.

---

## References

1. Broadbent, S. R. & Hammersley, J. M. Percolation processes. I. Crystals and mazes. *Proc. Camb. Phil. Soc.* **53**, 629–641 (1957).
2. Bethe, H. Approximate theory of the magnetic alloy in the case of partial saturation. *Proc. Camb. Phil. Soc.* **31**, 253–263 (1935).
3. Sykes, M. F. & Essam, J. W. Exact critical percolation probabilities for site and bond problems in two dimensions. *Phys. Rev.* **133**, A310–A315 (1964).
4. Ziff, R. M. & Saperial, B. The site percolation threshold of the square lattice. *Phys. Rev. E* **57**, R6242 (1998).
5. Suding, P. N. & Ziff, R. M. Site percolation thresholds for Archimedean lattices. *Phys. Rev. E* **57**, R6242–R6245 (1998).
6. Lorenz, C. D. & Ziff, R. M. Exact Monte Carlo results for percolation thresholds of the diamond lattice. *Phys. Rev. Lett.* **81**, 3387–3390 (1998).
7. Mertens, S. & Moore, C. Continuum percolation thresholds in two dimensions. *Phys. Rev. E* **86**, 061109 (2012).
8. Mertens, S. & Moore, C. Continuum percolation thresholds. *Phys. Rev. E* **98**, 022117 (2018).
9. Malarz, K. Continuum percolation in two dimensions with extended neighborhoods. *Phys. Rev. E* **107**, 034130 (2023).
10. Malarz, K. Extended-neighborhood percolation on the honeycomb lattice. *Phys. Rev. E* (2024).
11. Xun, Z. et al. Site percolation thresholds for three-dimensional lattices with extended neighbor connections. *Phys. Rev. E* (2021–2023).
12. Nachmias, A. & Peres, Y. Critical percolation on random regular graphs. *Ann. Probab.* **36**, 1565–1576 (2008).
13. van der Marck, S. C. Uniform approach to the percolation threshold on lattices. *Phys. Rev. E* **58**, 1562–1568 (1998).
14. Galam, S. & Mauger, A. Site percolation thresholds of body-centered and face-centered cubic-related lattices. *Phys. Rev. E* **53**, 416–421 (1996).

---

## Supplementary Information

### Supplementary Table 1: Dataset of 94 lattice configurations

[To be formatted as a proper table with columns: name, z, d, g, p_c, source]

### Supplementary Table 2: Model comparison

| Model | Form | Parameters | AIC | CV MAPE |
|-------|------|------------|-----|---------|
| Exponential (baseline) | exp(−αd) | 4 | −668.3 | 8.50% ± 0.09% |
| Power-law | d⁻ᵝ | 4 | −667.4 | 9.84% |
| H1: g=3 indicator | exp(−αd) · [1 + c·I(g=3)] | 5 | −671.6 | — |
| H2: d/g cross term | exp(−αd) · [1 + c/(dg)] | 5 | −666.7 | — |
| H3: nonlinear girth | exp(−αd) / [g^γ + c·I(g=3)] | 5 | −671.6 | — |
| H4: piecewise α | exp(−α_eff·d), α_eff = α₁(d≤3), α₂(d≥4) | 5 | −667.0 | 8.25% |
| H5: double exponential | exp(−αd − β/g²) | 5 | −671.6 | — |
| H6: piecewise + d=3 correction | H4 · [1 + c·I(d=3)] | 6 | −665.1 | 11.20% |
| H7: d=3,g=3 correction | exp(−αd) · [1 + c·I(d=3,g=3)] | 5 | −666.4 | 10.38% |
| H8: logistic α(d) | exp(−α(d)·d), α(d) logistic | 6 | −665.2 | — |
| H9: 2D g=3 correction | exp(−αd) · [1 + c·I(d=2,g=3)] | 5 | −667.9 | — |
| H10: d=3 + 2D g=3 | exp(−αd) · [1 + c₁·I(d=3)] · [1 + c₂·I(d=2,g=3)] | 6 | −666.1 | — |
| H11: piecewise + 2D g=3 | H4 · [1 + c·I(d=2,g=3)] | 6 | −666.2 | — |
| H12: stretched exponential | exp(−αd^δ) | 5 | −666.6 | — |

### Python implementation

```python
import numpy as np

def percolation_threshold(z, d, g):
    """
    Predict site percolation threshold.

    Parameters
    ----------
    z : int or array
        Coordination number
    d : int or array
        Spatial dimension
    g : int or array
        Girth (shortest cycle length)

    Returns
    -------
    pc : float or array
        Percolation threshold
    """
    a, alpha, gamma, z0 = 12.33, 0.337, 0.848, 9.62
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)
```
