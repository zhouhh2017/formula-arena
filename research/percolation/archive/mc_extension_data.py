from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
Monte Carlo Extension of Percolation Dataset
=============================================
Goal: Break z-g collinearity by generating new lattice data via MC simulation.

New lattice types:
1. d-dimensional hypercubic lattices (g=4, z=2d)
2. d-dimensional simple cubic with zone 1+2 neighbors (g=3, z=2d+2d(d-1))
3. Incidence graphs of projective planes PG(2,q) (g=6, z=q+1)

Method:
- Site percolation on each lattice
- Wrapping probability criterion for p_c(L)
- Finite-size scaling: p_c(L) = p_c(∞) + a·L^{-1/ν}
- Bootstrap for uncertainty estimation
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from collections import deque
import time
import sys
import os

# ============================================================
# SECTION 1: Union-Find with displacement tracking
# ============================================================

class UnionFindWrapped:
    """
    Union-Find data structure with displacement tracking for
    detecting wrapping clusters in periodic lattices.

    Each site stores a displacement vector relative to its cluster root.
    When two sites in the same cluster have different displacements,
    the cluster wraps around the periodic lattice.
    """

    def __init__(self, n_sites, d):
        self.n = n_sites
        self.d = d
        self.parent = np.arange(n_sites, dtype=np.int32)
        self.rank = np.zeros(n_sites, dtype=np.int16)
        # disp[i] = displacement of site i relative to parent[i]
        self.disp = np.zeros((n_sites, d), dtype=np.int32)

    def find(self, x):
        """Find root with path compression, accumulating displacements."""
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        root = x
        # Path compression: update all nodes on path
        cum_disp = np.zeros(self.d, dtype=np.int32)
        for node in reversed(path):
            cum_disp = cum_disp + self.disp[node]
            self.disp[node] = cum_disp.copy()
            self.parent[node] = root
        # Actually, we need to do this differently
        # Let me redo: walk from each node to root, accumulate disp
        for node in path:
            d_total = np.zeros(self.d, dtype=np.int32)
            cur = node
            while self.parent[cur] != cur:
                d_total += self.disp[cur]
                cur = self.parent[cur]
            # cur is root
            self.disp[node] = d_total
            self.parent[node] = root
        return root

    def find_simple(self, x):
        """Simple find without path compression (for use during union)."""
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y, delta):
        """
        Union sites x and y, where y is a neighbor of x with
        displacement delta (position of y = position of x + delta).
        """
        rx = self.find(x)
        ry = self.find(y)

        if rx == ry:
            # Same cluster: check for wrapping
            # Displacement of x relative to rx: self.disp[x]
            # Displacement of y relative to ry: self.disp[y]
            # The edge x->y adds displacement delta
            # Cycle displacement = disp[x] + delta - disp[y]
            cycle = self.disp[x] + delta - self.disp[y]
            return np.any(cycle != 0)
        else:
            # Different clusters: merge
            # We need: disp_new[y_nodes] such that
            # For site y: disp[y] relative to rx should be disp[x] + delta
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
                # Flip delta direction
                delta = -delta

            # Make ry's root point to rx
            # disp[ry] should be set so that:
            # disp[y] (old, relative to ry) + disp[ry] (new) = disp[x] + delta
            # => disp[ry] = disp[x] + delta - disp[y]
            self.parent[ry] = rx
            self.disp[ry] = self.disp[x] + delta - self.disp[y]

            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return False

    def check_wrapping(self):
        """Check if any cluster wraps around any dimension."""
        # After all unions, check each site
        for i in range(self.n):
            root = self.find(i)
            if np.any(self.disp[i] != 0):
                return True
        return False

    def check_wrapping_fast(self, occupied_mask):
        """Check wrapping only for occupied sites."""
        occ_indices = np.where(occupied_mask)[0]
        for i in occ_indices:
            root = self.find(i)
            if np.any(self.disp[i] != 0):
                return True
        return False


# ============================================================
# SECTION 2: Lattice construction
# ============================================================

def build_hypercubic_neighbors(L, d):
    """
    Build neighbor list for d-dimensional hypercubic lattice of size L^d
    with periodic boundary conditions.

    Returns:
        n_sites: total number of sites
        neighbors: list of lists, neighbors[i] = list of neighbor indices
        positions: array of shape (n_sites, d), coordinates of each site
        basis: list of basis vectors (unit vectors)
    """
    n_sites = L ** d
    # Basis vectors for hypercubic: ±e_i for i=1..d
    basis = []
    for i in range(d):
        e = np.zeros(d, dtype=np.int32)
        e[i] = 1
        basis.append(e)

    # Build neighbor list
    neighbors = [[] for _ in range(n_sites)]
    positions = np.zeros((n_sites, d), dtype=np.int32)

    for idx in range(n_sites):
        # Convert index to coordinates
        coords = np.zeros(d, dtype=np.int32)
        tmp = idx
        for dim in range(d - 1, -1, -1):
            coords[dim] = tmp % L
            tmp //= L
        positions[idx] = coords

        for e in basis:
            for sign in [1, -1]:
                disp = sign * e
                new_coords = (coords + disp) % L
                n_idx = 0
                multiplier = 1
                for dim in range(d - 1, -1, -1):
                    n_idx += new_coords[dim] * multiplier
                    multiplier *= L
                neighbors[idx].append(n_idx)

    return n_sites, neighbors, positions, basis


def build_sc_zone12_neighbors(L, d):
    """
    Build neighbor list for d-dimensional simple cubic lattice with
    zone 1 (nearest) + zone 2 (next-nearest) neighbors.
    Periodic boundary conditions.

    Zone 1: displacements with one ±1, rest 0 (2d neighbors)
    Zone 2: displacements with two ±1, rest 0 (2d(d-1) neighbors)
    Total z = 2d + 2d(d-1) = 2d²
    """
    n_sites = L ** d

    # Generate all zone 1 + zone 2 displacements
    basis = []
    # Zone 1: ±e_i for each dimension
    for i in range(d):
        for sign in [1, -1]:
            e = np.zeros(d, dtype=np.int32)
            e[i] = sign
            basis.append(e)
    # Zone 2: all vectors with exactly 2 nonzero coords, each ±1
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    e = np.zeros(d, dtype=np.int32)
                    e[i] = si
                    e[j] = sj
                    basis.append(e)

    neighbors = [[] for _ in range(n_sites)]
    positions = np.zeros((n_sites, d), dtype=np.int32)

    for idx in range(n_sites):
        coords = np.zeros(d, dtype=np.int32)
        tmp = idx
        for dim in range(d - 1, -1, -1):
            coords[dim] = tmp % L
            tmp //= L
        positions[idx] = coords

        for disp_vec in basis:
            new_coords = (coords + disp_vec) % L
            n_idx = 0
            multiplier = 1
            for dim in range(d - 1, -1, -1):
                n_idx += new_coords[dim] * multiplier
                multiplier *= L
            neighbors[idx].append(n_idx)

    return n_sites, neighbors, positions, basis


def _factor_prime_power(q):
    """Factor q = p^k. Returns (p, k)."""
    for p in range(2, q + 1):
        if q == 1:
            return (q, 0)
        k = 0
        temp = q
        while temp % p == 0:
            temp //= p
            k += 1
        if temp == 1 and k > 0:
            return (p, k)
    return (q, 1)


# Irreducible polynomials over GF(p) for constructing GF(p^k)
# Format: list of coefficients [a_0, a_1, ..., a_k] where poly = a_0 + a_1*x + ... + x^k
_IRRED_POLYS = {
    # (p, k): irreducible polynomial of degree k over GF(p)
    (2, 2): [1, 1, 1],       # x^2 + x + 1 over GF(2)
    (2, 3): [1, 1, 0, 1],    # x^3 + x + 1 over GF(2)
    (2, 4): [1, 1, 0, 0, 1], # x^4 + x + 1 over GF(2)
    (3, 2): [1, 1, 1],       # x^2 + x + 1 over GF(3) (check: no roots in GF(3))
    (3, 3): [1, 2, 0, 1],    # x^3 + 2 + ... actually x^3 - x + 1 = x^3 + 2x + 1 over GF(3)
    (5, 2): [2, 0, 1],       # x^2 + 2 over GF(5) (check: x^2 = -2 = 3 has no solution in GF(5))
    (7, 2): [3, 0, 1],       # x^2 + 3 over GF(7) (check: x^2 = -3 = 4 has no solution... wait 2^2=4. Use x^2+1: x^2=-1=6, no square = 6 in GF(7))
}

def _verify_irreducible(coeffs, p):
    """Verify a polynomial has no roots in GF(p)."""
    degree = len(coeffs) - 1
    for x in range(p):
        val = 0
        xpow = 1
        for c in coeffs:
            val = (val + c * xpow) % p
            xpow = (xpow * x) % p
        if val == 0:
            return False
    return True

def _find_irreducible(p, k):
    """Find an irreducible polynomial of degree k over GF(p) by brute force."""
    from itertools import product
    for coeffs_tuple in product(range(p), repeat=k):
        # Polynomial: x^k + c_{k-1}*x^{k-1} + ... + c_0
        coeffs = list(coeffs_tuple) + [1]  # monic
        if _verify_irreducible(coeffs, p):
            # For degree > 2, also need to check it doesn't factor
            # For simplicity, just check no roots (works for degree 2,3)
            if k <= 3:
                return coeffs
            # For higher degree, do a more thorough check
            # Actually, for our purposes (small k), root-free is often enough
            return coeffs
    return None


class GaloisField:
    """
    Finite field GF(p^k) with elements represented as polynomials over GF(p)
    modulo an irreducible polynomial.

    Elements are stored as lists of coefficients [a_0, a_1, ..., a_{k-1}]
    representing a_0 + a_1*x + ... + a_{k-1}*x^{k-1}.
    """

    def __init__(self, q):
        self.q = q
        p, k = _factor_prime_power(q)
        self.p = p
        self.k = k

        if k == 1:
            # Prime field: GF(p) = Z/pZ
            self.is_prime = True
        else:
            self.is_prime = False
            # Find irreducible polynomial
            key = (p, k)
            if key in _IRRED_POLYS:
                self.irred = _IRRED_POLYS[key]
            else:
                self.irred = _find_irreducible(p, k)
                if self.irred is None:
                    raise ValueError(f"Cannot find irreducible poly of degree {k} over GF({p})")

        # Build multiplication and inverse tables for small fields
        if q <= 32:
            self._build_tables()

    def _elem_to_int(self, elem):
        """Convert element (list of coeffs) to integer representation."""
        val = 0
        for i, c in enumerate(elem):
            val += c * (self.p ** i)
        return val

    def _int_to_elem(self, val):
        """Convert integer to element (list of coeffs)."""
        if self.is_prime:
            return [val % self.p]
        elem = []
        for _ in range(self.k):
            elem.append(val % self.p)
            val //= self.p
        return elem

    def _build_tables(self):
        """Build multiplication and inverse tables."""
        q = self.q
        self.mul_table = [[0] * q for _ in range(q)]
        self.inv_table = [0] * q

        for a in range(q):
            for b in range(q):
                self.mul_table[a][b] = self._elem_to_int(self.mul(self._int_to_elem(a), self._int_to_elem(b)))

        for a in range(1, q):
            self.inv_table[a] = self._elem_to_int(self.inv(self._int_to_elem(a)))

    def add(self, a, b):
        """Add two field elements."""
        if self.is_prime:
            return [(a[0] + b[0]) % self.p]
        return [(a[i] + b[i]) % self.p for i in range(self.k)]

    def sub(self, a, b):
        """Subtract two field elements."""
        if self.is_prime:
            return [(a[0] - b[0]) % self.p]
        return [(a[i] - b[i]) % self.p for i in range(self.k)]

    def mul(self, a, b):
        """Multiply two field elements."""
        if self.is_prime:
            return [(a[0] * b[0]) % self.p]

        # Polynomial multiplication
        result = [0] * (2 * self.k - 1)
        for i in range(self.k):
            for j in range(self.k):
                result[i + j] = (result[i + j] + a[i] * b[j]) % self.p

        # Reduce modulo irreducible polynomial
        return self._reduce(result)

    def _reduce(self, poly):
        """Reduce polynomial modulo irreducible polynomial."""
        irred = self.irred
        # Make a copy
        poly = list(poly)
        # Pad to at least degree k
        while len(poly) < self.k:
            poly.append(0)

        # Long division
        for i in range(len(poly) - 1, self.k - 1, -1):
            if poly[i] == 0:
                continue
            coeff = poly[i]  # Leading coefficient (irred is monic)
            for j in range(self.k + 1):
                poly[i - self.k + j] = (poly[i - self.k + j] - coeff * irred[j]) % self.p

        # Result is poly[0..k-1]
        result = [poly[i] % self.p for i in range(self.k)]
        return result

    def inv(self, a):
        """Multiplicative inverse using extended Euclidean algorithm."""
        if self.is_prime:
            return [pow(a[0], self.p - 2, self.p)]

        # Use extended GCD on polynomials
        # Or use Fermat's little theorem: a^(q-1) = 1, so a^(-1) = a^(q-2)
        return self._power(a, self.q - 2)

    def _power(self, a, n):
        """Compute a^n using repeated squaring."""
        if self.is_prime:
            return [pow(a[0], n, self.p)]

        result = [1] + [0] * (self.k - 1)  # 1
        base = list(a)
        while n > 0:
            if n % 2 == 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            n //= 2
        return result

    def elements(self):
        """Return all elements of the field as integer representations."""
        return list(range(self.q))

    def nonzero_elements(self):
        """Return all nonzero elements as integer representations."""
        return list(range(1, self.q))

    def to_elem(self, val):
        """Convert integer to element."""
        return self._int_to_elem(val)

    def from_elem(self, elem):
        """Convert element to integer."""
        return self._elem_to_int(elem)


def build_projective_plane_incidence_graph(q):
    """
    Build the incidence graph of PG(2, q).

    PG(2,q) has q²+q+1 points and q²+q+1 lines.
    The incidence graph is bipartite, (q+1)-regular, girth 6.
    Total vertices: 2(q²+q+1).

    Uses proper GF(q) arithmetic for prime power q.
    """
    gf = GaloisField(q)
    n_points = q * q + q + 1

    # Enumerate all nonzero triples in GF(q)^3
    all_triples = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                all_triples.append((a, b, c))

    # Canonical form: multiply by inverse of first nonzero element
    def canonical(triple):
        a, b, c = triple
        for val in [a, b, c]:
            if val != 0:
                inv_val = gf.inv(gf.to_elem(val))
                inv_int = gf.from_elem(inv_val)
                na = gf.from_elem(gf.mul(gf.to_elem(a), inv_val))
                nb = gf.from_elem(gf.mul(gf.to_elem(b), inv_val))
                nc = gf.from_elem(gf.mul(gf.to_elem(c), inv_val))
                return (na, nb, nc)
        return triple

    # Get unique canonical representatives (points)
    point_set = set()
    for t in all_triples:
        point_set.add(canonical(t))
    points = sorted(point_set)
    assert len(points) == n_points, f"Expected {n_points} points, got {len(points)}"

    lines = points[:]  # Self-dual

    # Build incidence
    point_to_lines = [[] for _ in range(n_points)]
    line_to_points = [[] for _ in range(n_points)]

    for pi, pt in enumerate(points):
        for li, ln in enumerate(lines):
            # Dot product: pt[0]*ln[0] + pt[1]*ln[1] + pt[2]*ln[2] = 0 in GF(q)
            dot = 0
            for k in range(3):
                term = gf.mul(gf.to_elem(pt[k]), gf.to_elem(ln[k]))
                dot_elem = gf.to_elem(dot)
                dot = gf.from_elem(gf.add(dot_elem, term))
            if dot == 0:
                point_to_lines[pi].append(li)
                line_to_points[li].append(pi)

    # Verify regularity
    for pi in range(n_points):
        assert len(point_to_lines[pi]) == q + 1, f"Point {pi} on {len(point_to_lines[pi])} lines"
    for li in range(n_points):
        assert len(line_to_points[li]) == q + 1, f"Line {li} has {len(line_to_points[li])} points"

    # Build incidence graph
    n_vertices = 2 * n_points
    neighbors = [[] for _ in range(n_vertices)]

    for pi in range(n_points):
        for li in point_to_lines[pi]:
            neighbors[pi].append(n_points + li)
            neighbors[n_points + li].append(pi)

    z_actual = q + 1
    print(f"  PG(2,{q}): {n_points} points, {n_points} lines, {n_vertices} vertices, z={z_actual}")

    return n_vertices, neighbors, z_actual


def compute_girth_bfs(n_sites, neighbors, max_girth=10, sample_sites=None):
    """
    Compute girth (shortest cycle) using BFS from sample sites.

    For large graphs, we sample a subset of sites.
    For the types of graphs we consider, girth is a topological invariant
    that doesn't depend on system size, so a small sample suffices.
    """
    if sample_sites is None:
        if n_sites > 500:
            sample_sites = min(50, n_sites)
            sites_to_check = np.random.choice(n_sites, sample_sites, replace=False)
        else:
            sites_to_check = range(n_sites)
    else:
        sites_to_check = sample_sites

    min_cycle = float('inf')

    for start in sites_to_check:
        if min_cycle <= 3:
            break  # Can't get smaller

        # BFS from start, looking for shortest cycle
        # For each neighbor of start, do BFS to find shortest path back
        for first_neighbor in neighbors[start]:
            # BFS from first_neighbor, looking for path back to start
            # that doesn't go through start (to find a cycle)
            visited = {first_neighbor: 0}  # node -> distance from first_neighbor
            parent = {first_neighbor: -1}
            queue = deque([first_neighbor])
            found = False

            while queue and not found:
                node = queue.popleft()
                dist = visited[node]

                if dist + 1 >= min_cycle:
                    break  # Can't improve

                for nb in neighbors[node]:
                    if nb == start and dist >= 1:
                        # Found cycle: start -> first_neighbor -> ... -> node -> start
                        cycle_len = dist + 2
                        min_cycle = min(min_cycle, cycle_len)
                        found = True
                        break
                    if nb not in visited and nb != start:
                        visited[nb] = dist + 1
                        parent[nb] = node
                        queue.append(nb)

    return min_cycle if min_cycle != float('inf') else -1


# ============================================================
# SECTION 3: Monte Carlo simulation (optimized with numpy)
# ============================================================

def build_edge_list(n_sites, neighbors):
    """
    Pre-compute edge list as numpy arrays for fast simulation.
    Returns (edge_i, edge_j) arrays where each pair appears once.
    """
    edges_i = []
    edges_j = []
    for i in range(n_sites):
        for j in neighbors[i]:
            if j > i:
                edges_i.append(i)
                edges_j.append(j)
    return np.array(edges_i, dtype=np.int32), np.array(edges_j, dtype=np.int32)


def fast_bfs_largest_component(n_sites, edges_i, edges_j, occupied):
    """
    Find largest connected component among occupied sites using BFS.
    Returns the fraction of the largest component relative to total occupied.
    """
    occ_indices = np.where(occupied)[0]
    n_occ = len(occ_indices)
    if n_occ == 0:
        return 0.0

    # Build adjacency dict for occupied sites only
    occ_set = set(occ_indices.tolist())
    adj = {}
    for i in occ_indices:
        adj[i] = []

    for idx in range(len(edges_i)):
        ei, ej = edges_i[idx], edges_j[idx]
        if occupied[ei] and occupied[ej]:
            adj[ei].append(ej)
            adj[ej].append(ei)

    # BFS to find components
    visited = set()
    max_size = 0

    for start in occ_indices:
        s = int(start)
        if s in visited:
            continue
        # BFS
        queue = deque([s])
        visited.add(s)
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if size > max_size:
            max_size = size

    return max_size / n_occ


def check_wrapping_bfs(n_sites, neighbors, occupied, L, d_lattice, positions=None):
    """
    Check if any cluster wraps around the periodic lattice using BFS.
    Uses precomputed positions for speed.
    """
    occ_indices = np.where(occupied)[0]
    n_occ = len(occ_indices)
    if n_occ == 0:
        return False

    # Precompute position lookup if not provided
    if positions is None:
        positions = np.zeros((n_sites, d_lattice), dtype=np.int32)
        for idx in range(n_sites):
            tmp = idx
            for dim in range(d_lattice - 1, -1, -1):
                positions[idx, dim] = tmp % L
                tmp //= L

    # For each unvisited occupied site, do a BFS tracking displacements
    visited = {}  # site -> displacement tuple
    half_L = L // 2

    for start_idx in occ_indices:
        s = int(start_idx)
        if s in visited:
            continue

        # BFS from this site
        queue = deque()
        queue.append((s, tuple([0] * d_lattice)))
        visited[s] = tuple([0] * d_lattice)

        wraps = False

        while queue:
            node, disp = queue.popleft()
            node_pos = positions[node]

            for nb in neighbors[node]:
                if not occupied[nb]:
                    continue

                nb_pos = positions[nb]

                # Compute actual displacement (shortest path through periodic BC)
                delta = []
                for dim in range(d_lattice):
                    diff = int(nb_pos[dim]) - int(node_pos[dim])
                    if diff > half_L:
                        diff -= L
                    elif diff < -half_L:
                        diff += L
                    delta.append(diff)

                # New displacement for nb
                new_disp = tuple(disp[d] + delta[d] for d in range(d_lattice))

                if nb in visited:
                    if visited[nb] != new_disp:
                        wraps = True
                        break
                else:
                    visited[nb] = new_disp
                    queue.append((nb, new_disp))

            if wraps:
                break

        if wraps:
            return True

    return False


def find_pc_binary_search(n_sites, edges_i, edges_j, neighbors, positions,
                          L, d_lattice, p_range, n_samples, tol,
                          use_wrapping=True):
    """
    Find p_c(L) using a 3-point P_wrap scan + linear interpolation.
    Fast and robust for finite-size scaling.

    Computes wrapping probability at 3 p values around p_guess,
    then interpolates to find where P_wrap = 0.5.
    """
    p_min, p_max = p_range
    p_center = (p_min + p_max) / 2.0

    if use_wrapping and positions is not None:
        # 3-point wrapping scan
        dp = min(0.02, (p_max - p_min) / 6.0)
        p_values = [p_center - dp, p_center, p_center + dp]
        P_wrap = []

        for p in p_values:
            count = 0
            for _ in range(n_samples):
                occupied = np.random.random(n_sites) < p
                if check_wrapping_bfs(n_sites, neighbors, occupied, L, d_lattice, positions):
                    count += 1
            P_wrap.append(count / n_samples)

        # Interpolate to find P_wrap = 0.5
        for i in range(len(p_values) - 1):
            if P_wrap[i] <= 0.5 and P_wrap[i + 1] > 0.5:
                if P_wrap[i + 1] > P_wrap[i]:  # Avoid division by zero
                    frac = (0.5 - P_wrap[i]) / (P_wrap[i + 1] - P_wrap[i])
                    return p_values[i] + frac * (p_values[i + 1] - p_values[i])

        # If no clean crossing, return closest to 0.5
        return p_values[np.argmin(np.abs(np.array(P_wrap) - 0.5))]

    else:
        # For non-lattice graphs: binary search on P(LFC > 0.5)
        p_low, p_high = p_min, p_max
        for _ in range(15):
            if p_high - p_low < tol:
                break
            p_mid = (p_low + p_high) / 2.0
            count = 0
            for _ in range(n_samples):
                occupied = np.random.random(n_sites) < p_mid
                frac = fast_bfs_largest_component(n_sites, edges_i, edges_j, occupied)
                if frac > 0.5:
                    count += 1
            if count / n_samples < 0.5:
                p_low = p_mid
            else:
                p_high = p_mid
        return (p_low + p_high) / 2.0


# ============================================================
# SECTION 4: Finite-size scaling
# ============================================================

def finite_size_scale_fit(L_values, pc_values, nu, n_bootstrap=200):
    """
    Fit finite-size scaling: p_c(L) = p_c_inf + a * L^{-1/nu}
    Returns: (p_c_inf, uncertainty)
    """
    L_arr = np.array(L_values, dtype=float)
    pc_arr = np.array(pc_values, dtype=float)

    def scaling_model(L, pc_inf, a):
        return pc_inf + a * L ** (-1.0 / nu)

    try:
        popt, pcov = curve_fit(scaling_model, L_arr, pc_arr,
                               p0=[pc_arr[-1], 0.1], maxfev=10000)
        p_c_inf = popt[0]
        err = np.sqrt(pcov[0, 0])

        # Bootstrap for better error estimate
        boot_pc = []
        for _ in range(n_bootstrap):
            # Resample with noise
            noise = np.random.normal(0, err, len(pc_arr))
            boot_data = pc_arr + noise
            try:
                bpopt, _ = curve_fit(scaling_model, L_arr, boot_data,
                                     p0=[boot_data[-1], 0.1], maxfev=5000)
                boot_pc.append(bpopt[0])
            except:
                pass

        if len(boot_pc) > 10:
            err = np.std(boot_pc)

    except Exception as e:
        # Fallback: linear extrapolation in 1/L
        print(f"    Warning: FSS fit failed ({e}), using linear extrapolation")
        inv_L = 1.0 / L_arr
        coeffs = np.polyfit(inv_L, pc_arr, 1)
        p_c_inf = coeffs[1]  # intercept at 1/L = 0
        err = np.std(pc_arr - np.polyval(coeffs, inv_L))

    return p_c_inf, max(err, 1e-6)


# ============================================================
# SECTION 5: Existing data and collinearity analysis
# ============================================================

# Malarz 2023 data from CSV (2D square lattice with complex neighborhoods)
# name format: sq-zone1,zone2,...,zone6
# girth computation: if zone 2 in zones → g=3 (diagonal creates triangle)
#                    else → g=4 (square lattice, no triangles)
MALARZ_DATA = []

def compute_malarz_girth(name):
    """Compute girth for Malarz lattice from its name."""
    # Parse zones from name like "sq-1,2,3,4,5,6"
    zones_str = name.replace("sq-", "")
    zones = [int(x) for x in zones_str.split(",")]
    # Zone 2 = diagonal neighbors = creates triangles
    if 2 in zones:
        return 3
    else:
        return 4

def parse_malarz_csv(filepath):
    """Parse the Malarz 2023 CSV file."""
    import csv
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name'].strip('"')
            z = int(row['z'])
            d = int(row['d'])
            pc = float(row['pc'])
            g = compute_malarz_girth(name)
            data.append({
                'name': name,
                'z': z,
                'd': d,
                'g': g,
                'pc': pc,
                'source': 'Malarz2023'
            })
    return data


# ============================================================
# SECTION 6: Main simulation
# ============================================================

# Define new lattice configurations
NEW_LATTICES = [
    {
        'id': 'hc4d',
        'name': 'hypercubic_4d',
        'type': 'hypercubic',
        'd': 4,
        'z': 8,
        'g': 4,
        'nu': 0.5,
        'L_values': [10, 14, 18],
        'n_samples': 300,
        'p_guess': 0.20,
    },
    {
        'id': 'sc3d_12',
        'name': 'sc3d_zone12',
        'type': 'sc_zone12',
        'd': 3,
        'z': 18,
        'g': 3,
        'nu': 0.63,
        'L_values': [8, 10, 14],
        'n_samples': 300,
        'p_guess': 0.12,
    },
    {
        'id': 'hc5d',
        'name': 'hypercubic_5d',
        'type': 'hypercubic',
        'd': 5,
        'z': 10,
        'g': 4,
        'nu': 0.5,
        'L_values': [6, 8, 10],
        'n_samples': 250,
        'p_guess': 0.14,
    },
    {
        'id': 'hc6d',
        'name': 'hypercubic_6d',
        'type': 'hypercubic',
        'd': 6,
        'z': 12,
        'g': 4,
        'nu': 0.5,
        'L_values': [5, 6, 7],
        'n_samples': 250,
        'p_guess': 0.11,
    },
    {
        'id': 'pg24',
        'name': 'PG2_4_incidence',
        'type': 'projective_plane',
        'q': 4,
        'd': None,
        'z': 5,
        'g': 6,
        'nu': None,
        'n_samples': 3000,
        'p_guess': 0.55,
    },
    {
        'id': 'pg25',
        'name': 'PG2_5_incidence',
        'type': 'projective_plane',
        'q': 5,
        'd': None,
        'z': 6,
        'g': 6,
        'nu': None,
        'n_samples': 3000,
        'p_guess': 0.50,
    },
]


def simulate_lattice(lattice_config):
    """Run full simulation for one lattice configuration."""
    cfg = lattice_config.copy()
    print(f"\n{'='*60}")
    print(f"Lattice: {cfg['name']}")
    print(f"  Type: {cfg['type']}, d={cfg.get('d', 'N/A')}, z={cfg['z']}, g={cfg['g']}")
    print(f"{'='*60}")

    t0 = time.time()

    if cfg['type'] == 'hypercubic':
        d = cfg['d']
        L_values = cfg['L_values']
        n_samples = cfg['n_samples']

        pc_L_values = []
        actual_sizes = []

        for L in L_values:
            n_sites, neighbors, positions, basis = build_hypercubic_neighbors(L, d)
            z_check = len(neighbors[0])
            edges_i, edges_j = build_edge_list(n_sites, neighbors)
            print(f"\n  L={L}: N={n_sites}, z={z_check}, edges={len(edges_i)}")

            p_c_L = find_pc_binary_search(
                n_sites, edges_i, edges_j, neighbors, positions,
                L, d,
                p_range=(max(0.01, cfg['p_guess'] - 0.06),
                         min(0.99, cfg['p_guess'] + 0.06)),
                n_samples=n_samples,
                tol=0.003,
                use_wrapping=True
            )
            pc_L_values.append(p_c_L)
            actual_sizes.append(L)
            print(f"    p_c(L={L}) = {p_c_L:.5f}")

        p_c_inf, uncertainty = finite_size_scale_fit(
            actual_sizes, pc_L_values, cfg['nu']
        )

        cfg['pc'] = p_c_inf
        cfg['uncertainty'] = uncertainty
        cfg['pc_L_values'] = pc_L_values
        cfg['L_values_used'] = actual_sizes

    elif cfg['type'] == 'sc_zone12':
        d = cfg['d']
        L_values = cfg['L_values']
        n_samples = cfg['n_samples']

        pc_L_values = []
        actual_sizes = []

        for L in L_values:
            n_sites, neighbors, positions, basis = build_sc_zone12_neighbors(L, d)
            z_check = len(neighbors[0])
            edges_i, edges_j = build_edge_list(n_sites, neighbors)
            expected_z = 2 * d + 2 * d * (d - 1)
            print(f"\n  L={L}: N={n_sites}, z={z_check}, edges={len(edges_i)}")

            p_c_L = find_pc_binary_search(
                n_sites, edges_i, edges_j, neighbors, positions,
                L, d,
                p_range=(max(0.01, cfg['p_guess'] - 0.06),
                         min(0.99, cfg['p_guess'] + 0.06)),
                n_samples=n_samples,
                tol=0.003,
                use_wrapping=True
            )
            pc_L_values.append(p_c_L)
            actual_sizes.append(L)
            print(f"    p_c(L={L}) = {p_c_L:.5f}")

        p_c_inf, uncertainty = finite_size_scale_fit(
            actual_sizes, pc_L_values, cfg['nu']
        )

        cfg['pc'] = p_c_inf
        cfg['uncertainty'] = uncertainty
        cfg['pc_L_values'] = pc_L_values
        cfg['L_values_used'] = actual_sizes

    elif cfg['type'] == 'projective_plane':
        q = cfg['q']
        n_vertices, nbrs, z_actual = build_projective_plane_incidence_graph(q)
        edges_i, edges_j = build_edge_list(n_vertices, nbrs)
        print(f"  Total vertices: {n_vertices}, z={z_actual}, edges={len(edges_i)}")

        g_computed = compute_girth_bfs(n_vertices, nbrs, max_girth=8)
        print(f"  Computed girth: {g_computed}")

        n_samples = cfg['n_samples']

        # Find p_c using largest cluster criterion
        p_c_est = find_pc_binary_search(
            n_vertices, edges_i, edges_j, nbrs, None,
            1, 1,
            p_range=(max(0.1, cfg['p_guess'] - 0.3),
                     min(0.9, cfg['p_guess'] + 0.3)),
            n_samples=n_samples,
            tol=0.003,
            use_wrapping=False
        )

        # Bootstrap uncertainty: run 5 independent estimates
        pc_estimates = []
        for trial in range(5):
            pc_trial = find_pc_binary_search(
                n_vertices, edges_i, edges_j, nbrs, None,
                1, 1,
                p_range=(max(0.1, cfg['p_guess'] - 0.3),
                         min(0.9, cfg['p_guess'] + 0.3)),
                n_samples=n_samples // 2,
                tol=0.005,
                use_wrapping=False
            )
            pc_estimates.append(pc_trial)

        uncertainty = np.std(pc_estimates) if len(pc_estimates) > 1 else 0.01

        cfg['pc'] = p_c_est
        cfg['uncertainty'] = max(uncertainty, 0.005)
        cfg['n_vertices'] = n_vertices

    elapsed = time.time() - t0
    print(f"\n  Result: p_c = {cfg['pc']:.5f} ± {cfg['uncertainty']:.5f}")
    print(f"  Time: {elapsed:.1f}s")

    return cfg


def main():
    print("=" * 70)
    print("MONTE CARLO EXTENSION OF PERCOLATION DATASET")
    print("Breaking z-g collinearity via new lattice simulations")
    print("=" * 70)

    np.random.seed(42)

    # ---- Step 1: Parse existing Malarz data ----
    print("\n[Step 1] Loading existing Malarz 2023 data...")
    csv_path = str(ROOT / 'malarz_2023_data.csv')
    existing_data = parse_malarz_csv(csv_path)
    print(f"  Loaded {len(existing_data)} existing lattices")

    # Show z-g distribution of existing data
    z_vals = [d['z'] for d in existing_data]
    g_vals = [d['g'] for d in existing_data]
    corr_before, pval_before = pearsonr(z_vals, g_vals)
    print(f"\n  Existing z-g correlation: r = {corr_before:.4f} (p = {pval_before:.2e})")
    print(f"  z range: [{min(z_vals)}, {max(z_vals)}]")
    print(f"  g values: {set(g_vals)}")
    for g in sorted(set(g_vals)):
        zs = [d['z'] for d in existing_data if d['g'] == g]
        print(f"    g={g}: z = {sorted(set(zs))}")

    # ---- Step 2: Run Monte Carlo simulations ----
    print("\n[Step 2] Running Monte Carlo simulations for new lattices...")
    new_results = []

    for cfg in NEW_LATTICES:
        result = simulate_lattice(cfg)
        new_results.append(result)

    # ---- Step 3: Verify girth of some new lattices ----
    print("\n[Step 3] Verifying girth of new lattices...")

    # Verify hypercubic 4D
    n, nbrs, pos, bas = build_hypercubic_neighbors(4, 4)
    g4 = compute_girth_bfs(n, nbrs, max_girth=6, sample_sites=range(5))
    print(f"  hypercubic_4d: computed girth = {g4} (expected 4)")

    # Verify sc3d zone12
    n, nbrs, pos, bas = build_sc_zone12_neighbors(4, 3)
    g3 = compute_girth_bfs(n, nbrs, max_girth=6, sample_sites=range(5))
    print(f"  sc3d_zone12: computed girth = {g3} (expected 3)")

    # ---- Step 4: Compile results ----
    print("\n[Step 4] Compiling results...")

    # New lattice results table
    print("\n" + "=" * 80)
    print("NEW LATTICE DATA (Monte Carlo)")
    print("=" * 80)
    print(f"{'name':<25} {'z':>4} {'d':>4} {'g':>3} {'p_c':>10} {'uncertainty':>12}")
    print("-" * 80)

    all_new_data = []
    for r in new_results:
        d_val = r.get('d', '-') if r.get('d') is not None else '-'
        if r['type'] == 'projective_plane':
            d_val = f"graph({r['n_vertices']})"
        print(f"{r['name']:<25} {r['z']:>4} {str(d_val):>4} {r['g']:>3} "
              f"{r['pc']:>10.5f} {r['uncertainty']:>12.5f}")
        all_new_data.append({
            'name': r['name'],
            'z': r['z'],
            'd': r.get('d', 0) if r.get('d') is not None else 0,
            'g': r['g'],
            'pc': r['pc'],
            'uncertainty': r['uncertainty'],
            'source': 'MC_this_work'
        })

    # ---- Step 5: Combined dataset and collinearity analysis ----
    print("\n" + "=" * 80)
    print("COLLINEARITY ANALYSIS")
    print("=" * 80)

    # Combined dataset
    combined = existing_data + all_new_data

    # Before: existing data only
    z_before = [d['z'] for d in existing_data]
    g_before = [d['g'] for d in existing_data]
    corr_before, pval_before = pearsonr(z_before, g_before)

    # After: combined
    z_after = [d['z'] for d in combined]
    g_after = [d['g'] for d in combined]
    corr_after, pval_after = pearsonr(z_after, g_after)

    print(f"\n  Before (Malarz only):  r(z,g) = {corr_before:.4f}, p = {pval_before:.2e}")
    print(f"  After  (combined):     r(z,g) = {corr_after:.4f}, p = {pval_after:.2e}")
    print(f"  Improvement: |r| reduced by {abs(corr_before) - abs(corr_after):.4f}")

    # Z-g distribution
    print(f"\n  Z-G distribution (combined):")
    all_z = sorted(set(z_after))
    all_g = sorted(set(g_after))
    print(f"  {'':>6}", end="")
    for g in all_g:
        print(f"  g={g:>2}", end="")
    print()
    for z in all_z:
        print(f"  z={z:<4}", end="")
        for g in all_g:
            count = sum(1 for d in combined if d['z'] == z and d['g'] == g)
            if count > 0:
                print(f"  {count:>4}", end="")
            else:
                print(f"  {'·':>4}", end="")
        print()

    # ---- Step 6: Comparison with known values ----
    print("\n" + "=" * 80)
    print("COMPARISON WITH KNOWN VALUES")
    print("=" * 80)

    # Known reference values for validation
    known_refs = {
        'hypercubic_4d': (0.196737, "Hypersc_4 literature"),
        'hypercubic_5d': (0.140810, "Hypersc_5 literature"),
        'hypercubic_6d': (0.109460, "Hypersc_6 literature"),
    }

    print(f"{'Lattice':<25} {'MC p_c':>10} {'Lit. p_c':>10} {'Δp_c':>10} {'Status':>10}")
    print("-" * 70)
    for r in new_results:
        if r['id'] in known_refs:
            lit_pc, ref_name = known_refs[r['id']]
            delta = r['pc'] - lit_pc
            status = "✓ OK" if abs(delta) < 0.02 else "✗ WARN"
            print(f"{r['name']:<25} {r['pc']:>10.5f} {lit_pc:>10.5f} {delta:>+10.5f} {status:>10}")

    # ---- Step 7: Save results ----
    print("\n[Step 7] Saving results...")

    output_dir = ROOT

    # Save combined dataset as CSV
    import csv
    csv_out = os.path.join(output_dir, "mc_extension_dataset.csv")
    with open(csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'z', 'd', 'g', 'pc', 'uncertainty', 'source'])
        writer.writeheader()
        for d_item in combined:
            row = {k: d_item.get(k, '') for k in ['name', 'z', 'd', 'g', 'pc', 'uncertainty', 'source']}
            writer.writerow(row)
    print(f"  Combined dataset: {csv_out}")

    # Save detailed results as JSON-like text
    results_file = os.path.join(output_dir, "mc_extension_results_detail.txt")
    with open(results_file, 'w') as f:
        f.write("Monte Carlo Extension Results - Detailed\n")
        f.write("=" * 60 + "\n\n")
        for r in new_results:
            f.write(f"Lattice: {r['name']}\n")
            f.write(f"  Type: {r['type']}\n")
            f.write(f"  z={r['z']}, d={r.get('d', 'N/A')}, g={r['g']}\n")
            f.write(f"  p_c = {r['pc']:.6f} ± {r['uncertainty']:.6f}\n")
            if 'pc_L_values' in r:
                f.write(f"  Finite-size scaling data:\n")
                for L, pc_L in zip(r.get('L_values_used', []), r['pc_L_values']):
                    f.write(f"    L={L}: p_c = {pc_L:.6f}\n")
            if 'n_vertices' in r:
                f.write(f"  Graph size: {r['n_vertices']} vertices\n")
            f.write("\n")
    print(f"  Detailed results: {results_file}")

    # Save collinearity analysis
    coll_file = os.path.join(output_dir, "mc_collinearity_analysis.txt")
    with open(coll_file, 'w') as f:
        f.write("Z-G Collinearity Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Before (Malarz 2023 only):\n")
        f.write(f"  N = {len(existing_data)}\n")
        f.write(f"  r(z,g) = {corr_before:.4f}\n")
        f.write(f"  p-value = {pval_before:.2e}\n\n")
        f.write(f"After (combined with MC data):\n")
        f.write(f"  N = {len(combined)}\n")
        f.write(f"  r(z,g) = {corr_after:.4f}\n")
        f.write(f"  p-value = {pval_after:.2e}\n\n")
        f.write(f"Improvement: Δ|r| = {abs(corr_before) - abs(corr_after):.4f}\n")
    print(f"  Collinearity analysis: {coll_file}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  New lattices simulated: {len(new_results)}")
    print(f"  Existing lattices: {len(existing_data)}")
    print(f"  Combined dataset: {len(combined)}")
    print(f"  z-g correlation: {corr_before:.4f} → {corr_after:.4f}")
    print(f"  Collinearity broken: {'YES' if abs(corr_after) < abs(corr_before) - 0.1 else 'PARTIAL'}")
    print()

    return new_results, existing_data, combined


if __name__ == '__main__':
    new_results, existing_data, combined = main()
