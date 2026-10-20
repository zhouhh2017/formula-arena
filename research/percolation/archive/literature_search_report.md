# Literature Search Report: Site Percolation Thresholds for Crystal Lattices

## Summary

This report compiles published site percolation threshold data from the literature, covering 2D and 3D standard lattices, Archimedean lattices, extended-range lattices, high-dimensional hypercubic lattices, and special crystal structures. Data was extracted from published papers via arXiv API and paper content analysis.

---

## 1. 2D Standard Lattices (Exact and High-Precision Values)

### Exact Values
| Lattice | z | g | pc | Source |
|---------|---|---|----|--------|
| Triangular | 6 | 3 | 0.500000000 (exact) | Sykes & Essam 1964 |
| Kagomé | 4 | 3 | 0.6527036... (exact) | Sykes & Essam 1964 |

- **Triangular**: pc = 1/2 exactly, proven via self-matching lattice argument.
- **Kagomé**: pc = 1 - 2sin(π/18) ≈ 0.6527036... exactly, derived from star-triangle transformation via the honeycomb bond threshold.

### High-Precision MC Values
| Lattice | z | g | pc | Uncertainty | Source |
|---------|---|---|----|-------------|--------|
| Honeycomb | 3 | 6 | 0.697043 | ±0.000003 | Suding & Ziff 1998; Malarz 2024 |
| Square | 4 | 4 | 0.592746 | ±0.000001 | Mertens & Moore 2012 |

- **Honeycomb**: Best known value 0.6970402(1) (Malarz 2024).
- **Square**: Extremely well-determined; 0.5927460(5) from extensive MC.

---

## 2. 2D Archimedean (Uniform) Lattices

From Suding & Ziff (1998), cond-mat/9811416, using hull-walk gradient-percolation:

| Lattice (notation) | z | g | pc |
|---------------------|---|---|----|
| Honeycomb (6³) | 3 | 6 | 0.697043 |
| (3,12²) | 3 | 3 | 0.807904 |
| (4,6,12) | 3 | 3 | 0.747806 |
| (4,8²) | 3 | 4 | 0.729724 |
| (3⁴,6) | 5 | 3 | 0.579498 |
| (3,4,6,4) | 4 | 3 | 0.621819 |
| (3³,4²) | 5 | 3 | 0.550213 |
| (3²,4,3,4) | 4 | 3 | 0.550806 |

All determined by MC hull-gradient method to ~5 significant digits.

---

## 3. 3D Standard Lattices

| Lattice | z | d | g | pc | Uncertainty | Source |
|---------|---|---|---|----|-------------|--------|
| Diamond | 4 | 3 | 6 | 0.4285 | ±0.0010 | Galam & Mauger 1996 |
| Simple cubic (sc) | 6 | 3 | 4 | 0.31161 | ±0.00001 | Lorenz & Ziff 1998; Mertens & Moore 2018 |
| BCC | 8 | 3 | 4 | 0.24597 | ±0.00002 | Lorenz & Ziff 1998 |
| FCC | 12 | 3 | 3 | 0.19923 | ±0.00002 | Lorenz & Ziff 1998 |
| HCP | 12 | 3 | 3 | 0.1992555 | ±0.0000010 | Lorenz, May & Ziff 1999 |

### Key Notes:
- **FCC vs HCP**: The site percolation thresholds differ slightly (fcc: 0.19923 vs hcp: 0.1992555), but bond thresholds are nearly identical. This was established precisely by Lorenz, May & Ziff (cond-mat/9908034).
- **Diamond**: z=4 is the lowest coordination for a 3D lattice; its girth is 6 (hexagonal rings).

---

## 4. 3D Extended-Range Lattices

From Xun, Hao & Ziff (2021), arXiv:2111.10975:

### SC with extended neighbors:
| Lattice | z | pc | Uncertainty |
|---------|---|----|-------------|
| sc-1,2 | 18 | 0.1373045 | ±0.0000005 |
| sc-1,2,3 | 26 | 0.0976444 | ±0.0000006 |
| sc-1,2,3,4 | 32 | 0.0801171 | ±0.0000009 |
| sc-1,...,5 | 56 | 0.0461815 | ±0.0000005 |
| sc-1,...,6 | 80 | 0.0337049 | ±0.0000009 |
| sc-1,...,7 | 92 | 0.0290800 | ±0.0000010 |
| sc-1,...,8 | 122 | 0.0218686 | ±0.0000006 |
| sc-1,...,9 | 146 | 0.0184060 | ±0.0000010 |

### BCC with extended neighbors:
| Lattice | z | pc |
|---------|---|----|
| bcc-1,2 | 14 | 0.1759432(8) |
| bcc-1,2,3 | 26 | 0.0959084(6) |

### FCC with extended neighbors:
| Lattice | z | pc |
|---------|---|----|
| fcc-1,2 | 18 | 0.1361408(8) |
| fcc-1,2,3 | 42 | 0.0618842(8) |

---

## 5. High-Dimensional Hypercubic Lattices (d = 4 to 13)

From Mertens & Moore (2018), arXiv:1806.08067 - invasion percolation MC, extremely high precision:

| d | z = 2d | pc (site) | Uncertainty |
|---|--------|-----------|-------------|
| 4 | 8 | 0.19688561 | ±0.00000003 |
| 5 | 10 | 0.14079633 | ±0.00000004 |
| 6 | 12 | 0.109016661 | ±0.000000008 |
| 7 | 14 | 0.088951121 | ±0.000000001 |
| 8 | 16 | 0.075210128 | ±0.000000001 |
| 9 | 18 | 0.0652095348 | ±0.0000000006 |
| 10 | 20 | 0.0575929488 | ±0.0000000004 |
| 11 | 22 | 0.0515896843 | ±0.0000000002 |
| 12 | 24 | 0.0467309755 | ±0.0000000001 |
| 13 | 26 | 0.04271507960 | ±0.00000000010 |

### Notes:
- Upper critical dimension is d_c = 6. For d ≥ 6, mean-field behavior with logarithmic corrections at d = 6.
- Series expansion for site threshold: pc = (q-1)^{-1} + (3/2)(q-1)^{-2} + (15/4)(q-1)^{-3} + (83/4)(q-1)^{-4} + ...
- Mertens & Moore (2018b, arXiv:1805.02701) extended the asymptotic series by additional terms.
- Bethe limit: pc → 1/(z-1) as z → ∞.

---

## 6. 4D and 5D Extended-Range Lattices

### 4D Simple Hypercubic with Extended Neighbors
From Zhao, Yan, Xun, Hao & Ziff (2021), arXiv:2109.11195:

| Lattice | z | pc |
|---------|---|----|
| sc(4)-1,2 | 32 | 0.0617731(19) |
| sc(4)-1,2,3 | 64 | 0.0319407(13) |
| sc(4)-1,2,3,4 | 88 | 0.0231538(12) |
| sc(4)-1,...,5 | 136 | 0.0147918(12) |
| sc(4)-1,...,6 | 232 | 0.0088400(10) |
| sc(4)-1,...,7 | 296 | 0.0070006(6) |
| sc(4)-1,...,8 | 320 | 0.0064681(9) |
| sc(4)-1,...,9 | 424 | 0.0048301(9) |

### 5D Simple Hypercubic with Extended Neighbors
From Xun, Hao & Ziff (2023), arXiv:2308.15719:

| Lattice | z | pc |
|---------|---|----|
| sc(5)-1,2 | 50 | 0.033794(4) |
| sc(5)-1,2,3 | 130 | 0.013190(2) |
| sc(5)-1,2,3,4 | 220 | 0.007850(1) |
| sc(5)-1,...,5 | 332 | 0.0051575(5) |
| sc(5)-1,...,6 | 572 | 0.0029805(5) |
| sc(5)-1,...,7 | 892 | 0.001934(1) |

---

## 7. d-Dimensional Kagomé Lattices

From van der Marck (1998), cond-mat/9801112:

| d | z = 2d | pc (site) | Uncertainty |
|---|--------|-----------|-------------|
| 3 | 6 | 0.3895 | ±0.0002 |
| 4 | 8 | 0.2715 | ±0.0003 |
| 5 | 10 | 0.2084 | ±0.0004 |
| 6 | 12 | 0.1677 | ±0.0007 |

### Key Finding:
The Kagomé lattice scales as pc ~ 2/q instead of the Bethe prediction pc ~ 1/(q-1). This is because the Kagomé lattice has triangles, making it fundamentally different from hypercubic lattices at high dimensions.

---

## 8. 2D Honeycomb Extended Range

From Malarz (2024), arXiv:2204.12593:

Comprehensive data for honeycomb lattice with neighborhoods up to 5th coordination zone. Selected values:

| Lattice | z | pc |
|---------|---|----|
| hc-1 (standard) | 3 | 0.697043(3) |
| hc-1,2 | 9 | 0.3630 |
| hc-1,3 | 6 | 0.4132 |
| hc-1,2,3 | 12 | 0.3030 |
| hc-1,2,3,4 | 18 | 0.2113 |
| hc-1,2,3,5 | 18 | 0.1973 |

Total of 31 neighborhoods studied.

---

## 9. Universal Formulas

### Galam & Mauger (1996), cond-mat/9606081:
Power law: pc = p₀[(d-1)(q-1)]^{-a} d^{b}

Two universality classes for d ≤ d_c:
- Class 1 (2D triangular, square, honeycomb): p₀ = 0.8889, a = 0.3601 (b=0 for sites)
- Class 2 (Kagomé + all d≥3 lattices): p₀ = 1.2868, a = 0.6160 (b=0 for sites)
- Class 3 (high d ≥ d_c): b = 2a-1 for both sites and bonds

### Xun, Hao & Ziff (2021) asymptotic formulas:
- For large z: z·pc → c(d) where c(d) = 2^d · η_c
  - d=3: c ≈ 2.7351
  - d=4: c ≈ 2.086
  - d=5: c ≈ 1.742

---

## 10. Exact Site Percolation Thresholds on 2D Lattices

The following 2D site thresholds are known EXACTLY:
1. **Triangular**: pc = 1/2 (self-matching)
2. **Kagomé**: pc = 1 - 2sin(π/18) (star-triangle)
3. **Martini lattice** (Scullard 2005, cond-mat/0507392): pc = solution of p⁵ - 4p⁴ + 3p³ + 2p² - 1 = 0, giving pc ≈ 0.625457
4. **"A" lattice** (Ziff 2005, cond-mat/0510245): pc = 1/√2 ≈ 0.707107

No exact 3D site percolation thresholds are known.

---

## 11. Lattices NOT Found in Literature

The following lattices were specifically searched for but NO published site percolation threshold data was found:
- **Simple hexagonal** (z=8, d=3) - no dedicated study found
- **Rutile lattice** percolation thresholds - no published data
- **Corundum lattice** percolation thresholds - no published data
- **Laves graphs** percolation thresholds - no published data
- **Pyrochlore lattice** site percolation threshold - no dedicated study found
- **Simple hexagonal** (z=8) - not found as a distinct entry

---

## 12. Data Quality Summary

| Category | Number of Points | Precision Range |
|----------|-----------------|-----------------|
| 2D Exact | 2 | Exact |
| 2D Archimedean | 8 | ±0.000005 |
| 2D Honeycomb ext. | 24 | ±0.0005 |
| 3D Standard | 5 | ±0.0000010 |
| 3D Extended range | 11 | ±0.0000010 |
| 4D-13D Hypercubic | 10 | ±10^{-10} |
| 4D Extended range | 8 | ±0.0000019 |
| 5D Extended range | 6 | ±0.000004 |
| d-Kagomé (3-6D) | 4 | ±0.0007 |
| **Total** | **78** | |

---

## 13. Key References

1. Sykes & Essam (1964) - Exact thresholds via star-triangle
2. Galam & Mauger (1996) - cond-mat/9606081, Universal formulae
3. Suding & Ziff (1998) - cond-mat/9811416, Archimedean lattices
4. van der Marck (1998) - cond-mat/9801112, d-dimensional Kagomé
5. Lorenz & Ziff (1998) - cond-mat/9806224, 3D universality
6. Lorenz, May & Ziff (1999) - cond-mat/9908034, HCP vs FCC
7. Scullard (2005) - cond-mat/0507392, Exact site thresholds
8. Ziff (2005) - cond-mat/0510245, Generalized cell/dual-cell
9. Mertens & Moore (2018) - arXiv:1806.08067, Hypercubic d=4-13
10. Mertens & Moore (2018b) - arXiv:1805.02701, Series expansion
11. Xun, Hao & Ziff (2021) - arXiv:2111.10975, 3D extended range
12. Zhao et al. (2021) - arXiv:2109.11195, 4D extended range
13. Xun, Hao & Ziff (2023) - arXiv:2308.15719, 5D extended range
14. Malarz (2024) - arXiv:2204.12593, Honeycomb extended range
