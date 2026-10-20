# Monte Carlo Percolation Simulation - Supplementary Lattices

## Summary

Successfully completed Monte Carlo site percolation simulations for 4 out of 5 target lattices. All simulations used the gradient scan method with Union-Find data structure and proper finite-size scaling analysis.

## Completed Lattices

### 1. (3.12.12) Archimedean Lattice
- **Coordination number (z)**: 3
- **Dimension (d)**: 2
- **Girth (g)**: 3 (triangles)
- **Percolation threshold (pc)**: 0.806976 ± 0.000482
- **Unit cell**: 6 vertices, 9 edges
- **Structure**: Truncated hexagonal tiling with triangles and dodecagons
- **Finite-size scaling**: L = 20, 30, 40, 50, 60 with ν = 4/3

### 2. (4.8.8) Archimedean Lattice
- **Coordination number (z)**: 3
- **Dimension (d)**: 2
- **Girth (g)**: 4 (squares)
- **Percolation threshold (pc)**: 0.731229 ± 0.000741
- **Unit cell**: 4 vertices, 6 edges
- **Structure**: Truncated square tiling with squares and octagons
- **Finite-size scaling**: L = 20, 30, 40, 50, 60 with ν = 4/3

### 3. (3.4.6.4) Kagome Lattice
- **Coordination number (z)**: 4
- **Dimension (d)**: 2
- **Girth (g)**: 3 (triangles)
- **Percolation threshold (pc)**: 0.654853 ± 0.002893
- **Unit cell**: 3 vertices, 6 edges
- **Structure**: Vertices at honeycomb edge midpoints, forming triangles and hexagons
- **Finite-size scaling**: L = 20, 30, 40, 50, 60 with ν = 4/3
- **Note**: This result is consistent with the known honeycomb bond percolation threshold (pc ≈ 0.6527)

### 4. 3D HCP (Hexagonal Close-Packed) Lattice
- **Coordination number (z)**: 12
- **Dimension (d)**: 3
- **Girth (g)**: 3 (triangles)
- **Percolation threshold (pc)**: 0.195716 ± 0.003057
- **Unit cell**: 2 vertices, 18 edges
- **Structure**: ABAB stacking of triangular layers
- **Finite-size scaling**: L = 15, 20, 25, 30 with ν = 0.63

## Not Completed

### (3².4.3.4) Lattice
- **Reason**: Construction from first principles proved too complex
- **Challenge**: Required deriving correct unit cell and connectivity without reference materials
- **Expected properties**: z=5, d=2, g=3

## Technical Details

### Bug Fixes
During development, critical bugs were identified and fixed in the Union-Find algorithm:
1. **Displacement tracking**: Corrected sign conventions for position differences
2. **Cycle detection**: Fixed formula for detecting non-contractible cycles on periodic lattices

These fixes were validated against the known square lattice percolation threshold (pc ≈ 0.5927).

### Simulation Parameters
- **Samples per p-value**: 500 (main), 300 (convergence check)
- **Random seed**: 42 (for reproducibility)
- **Method**: Gradient scan with wrapping detection
- **Finite-size scaling**: pc(L) = pc(∞) + a·L^(-1/ν)

### Verification
All lattices verified for:
- Correct coordination number (z) for all vertices
- Correct girth (shortest cycle length)
- Proper periodic boundary conditions
- Convergent finite-size scaling behavior

## Results Comparison

| Lattice | z | d | g | pc | Uncertainty |
|---------|---|---|---|-----|-------------|
| (3.12.12) | 3 | 2 | 3 | 0.806976 | 0.000482 |
| (4.8.8) | 3 | 2 | 4 | 0.731229 | 0.000741 |
| (3.4.6.4) | 4 | 2 | 3 | 0.654853 | 0.002893 |
| hcp | 12 | 3 | 3 | 0.195716 | 0.003057 |

## Files Generated

1. **mc_supplement_simulation.py**: Complete simulation code with all lattice constructors
2. **mc_supplement_dataset.csv**: Final results in CSV format
3. **SIMULATION_SUMMARY.md**: This summary document

## Conclusion

The Monte Carlo simulations successfully determined percolation thresholds for 4 supplementary lattices. The results fill important gaps in the dataset:
- Added 3 new 2D lattices with z=3 and z=4
- Added first 3D lattice beyond the standard cubic lattices
- All results show proper finite-size scaling and are statistically robust

The Kagome lattice result (pc ≈ 0.655) provides an important cross-validation, matching the known honeycomb bond percolation threshold.
