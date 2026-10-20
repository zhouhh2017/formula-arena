from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
Monte Carlo site percolation for supplementary lattices.
Lattices: (3.12.12), (4.8.8), (3².4.3.4), (3.4.6.4)/Kagome, 3D HCP
"""

import numpy as np
import time
import csv
from collections import deque, defaultdict

OUTPUT_DIR = ROOT

# =============================================================================
# Union-Find with displacement tracking
# =============================================================================

class UnionFindWrap:
    def __init__(self, n, d):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)
        self.disp = np.zeros((n, d), dtype=np.float64)
        self.d = d
        self.wrapping = False

    def find(self, x):
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        root = x
        d_acc = np.zeros(self.d)
        for node in reversed(path):
            d_acc = d_acc + self.disp[node]
            self.disp[node] = d_acc.copy()
            self.parent[node] = root
        return root

    def union(self, i, j, delta):
        ri = self.find(i)
        rj = self.find(j)
        if ri == rj:
            # Cycle displacement: go from root to i (disp[i]), then i to j (delta),
            # then j back to root (-disp[j]).
            # Total = disp[i] + delta - disp[j]. Non-zero means wrapping.
            cycle = self.disp[i] + delta - self.disp[j]
            if np.any(np.abs(cycle) > 1e-10):
                self.wrapping = True
            return
        if self.rank[ri] < self.rank[rj]:
            self.parent[ri] = rj
            # disp[ri] = position(ri) - position(rj) = D[j] - D[i] - delta
            self.disp[ri] = self.disp[j] - self.disp[i] - delta
        elif self.rank[ri] > self.rank[rj]:
            self.parent[rj] = ri
            # disp[rj] = position(rj) - position(ri) = D[i] - D[j] + delta
            self.disp[rj] = self.disp[i] - self.disp[j] + delta
        else:
            self.parent[rj] = ri
            self.disp[rj] = self.disp[i] - self.disp[j] + delta
            self.rank[ri] += 1


# =============================================================================
# Lattice construction functions
# =============================================================================

def build_3_12_12(L):
    """
    (3.12.12) truncated hexagonal tiling.
    6 vertices per cell, 9 edges per cell. z=3, g=3.

    Derived from honeycomb truncation:
    Vertices per cell: a1,a2,a3 (near A), b1,b2,b3 (near B).
    a1 on edge A-B(i,j), a2 on A-B(i-1,j), a3 on A-B(i,j-1).
    b1 on edge B-A(i,j), b2 on B-A(i+1,j), b3 on B-A(i,j+1).

    Edge-segments (connect cut points on same honeycomb edge):
    a1(i,j)-b1(i,j) [internal], a2(i,j)-b2(i-1,j) [external], a3(i,j)-b3(i,j-1) [external]
    Triangles at A and B: 3 edges each, all internal.
    Total: 3+3+1+2 = 9 edges per cell.
    """
    n = 6 * L * L
    basis = np.array([[np.sqrt(3), 0.0], [np.sqrt(3)/2, 3.0/2]])

    edges = []
    for i in range(L):
        for j in range(L):
            c = i * L + j
            a1 = 6*c; a2 = 6*c+1; a3 = 6*c+2
            b1 = 6*c+3; b2 = 6*c+4; b3 = 6*c+5

            im = (i-1) % L
            jm = (j-1) % L

            # Internal (7 edges)
            # Triangle at A: a1-a2, a2-a3, a1-a3
            edges.append((a1, a2, 0, 0))
            edges.append((a2, a3, 0, 0))
            edges.append((a1, a3, 0, 0))
            # Triangle at B: b1-b2, b2-b3, b1-b3
            edges.append((b1, b2, 0, 0))
            edges.append((b2, b3, 0, 0))
            edges.append((b1, b3, 0, 0))
            # Edge segment a1-b1 (internal, same honeycomb edge A-B(i,j))
            edges.append((a1, b1, 0, 0))

            # External (2 edges)
            # a2(i,j) connects to b2(i-1,j): same honeycomb edge A(i,j)-B(i-1,j)
            edges.append((a2, 6*(im*L+j)+4, -1, 0))
            # a3(i,j) connects to b3(i,j-1): same honeycomb edge A(i,j)-B(i,j-1)
            edges.append((a3, 6*(i*L+jm)+5, 0, -1))

    return n, basis, edges


def build_4_8_8(L):
    """
    (4.8.8) truncated square tiling. z=3, g=4.
    4 vertices per cell, 6 edges per cell.

    Square lattice truncation:
    V(i,j) → square with 4 vertices v0,v1,v2,v3
    v0 on edge V(i,j)-V(i+1,j), etc.
    Internal: v0-v1, v1-v2, v2-v3, v3-v0 (square, 4 edges)
    External: v0(i,j)-v2(i+1,j), v1(i,j)-v3(i,j+1) (2 edges)
    """
    n = 4 * L * L
    basis = np.array([[1.0 + np.sqrt(2), 0.0], [0.0, 1.0 + np.sqrt(2)]])

    edges = []
    for i in range(L):
        for j in range(L):
            c = i * L + j
            v0 = 4*c; v1 = 4*c+1; v2 = 4*c+2; v3 = 4*c+3

            ip = (i+1) % L
            jp = (j+1) % L

            # Internal (4): square
            edges.append((v0, v1, 0, 0))
            edges.append((v1, v2, 0, 0))
            edges.append((v2, v3, 0, 0))
            edges.append((v3, v0, 0, 0))

            # External (2)
            edges.append((v0, 4*(ip*L+j)+2, 1, 0))  # v0(i,j) → v2(i+1,j)
            edges.append((v1, 4*(i*L+jp)+3, 0, 1))  # v1(i,j) → v3(i,j+1)

    return n, basis, edges


def build_kagome(L):
    """
    (3.4.6.4) Kagome lattice. z=4, g=3.
    3 vertices per cell, 6 edges per cell.

    Vertices at edge midpoints of honeycomb.
    v0: midpoint of A(i,j)-B(i,j)
    v1: midpoint of A(i,j)-B(i-1,j)
    v2: midpoint of A(i,j)-B(i,j-1)

    Internal (3): v0-v1, v1-v2, v0-v2 (triangle at A(i,j))
    External (3): triangle at B(i,j) edges:
      v0(i,j)-v1(i+1,j): offset (1,0)
      v0(i,j)-v2(i,j+1): offset (0,1)
      v1(i,j)-v2(i-1,j+1): offset (-1,1) [from B(i-1,j)'s triangle]
    """
    n = 3 * L * L
    basis = np.array([[0.5, 0.0], [0.25, np.sqrt(3)/4]])

    edges = []
    for i in range(L):
        for j in range(L):
            c = i * L + j
            v0 = 3*c; v1 = 3*c+1; v2 = 3*c+2

            im = (i-1) % L
            jp = (j+1) % L

            # Internal (3): triangle at A(i,j)
            edges.append((v0, v1, 0, 0))
            edges.append((v1, v2, 0, 0))
            edges.append((v0, v2, 0, 0))

            # External (3): triangle at B(i,j) has vertices v0(i,j), v1(i+1,j), v2(i,j+1)
            ip = (i+1) % L
            jp = (j+1) % L
            edges.append((v0, 3*(ip*L+j)+1, 1, 0))     # v0(i,j)-v1(i+1,j), offset (1,0)
            edges.append((v0, 3*(i*L+jp)+2, 0, 1))     # v0(i,j)-v2(i,j+1), offset (0,1)
            edges.append((v1, 3*(im*L+jp)+2, -1, 1))   # v1(i,j)-v2(i-1,j+1), offset (-1,1)

    return n, basis, edges


def build_hcp_3d(L):
    """
    3D HCP lattice. z=12, g=3.
    2 atoms per cell, 18 edges per cell.
    """
    n = 2 * L * L * L
    c = np.sqrt(8.0 / 3.0)
    basis = np.array([
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
        [0.0, 0.0, c]
    ])

    edges = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                ci = (i * L + j) * L + k
                A = 2 * ci
                B = 2 * ci + 1

                def A_idx(di, dj, dk):
                    return 2 * (((i+di)%L * L + (j+dj)%L) * L + (k+dk)%L)
                def B_idx(di, dj, dk):
                    return 2 * (((i+di)%L * L + (j+dj)%L) * L + (k+dk)%L) + 1

                # A-A in-plane (6)
                edges.append((A, A_idx(1,0,0), 1,0,0))
                edges.append((A, A_idx(-1,0,0), -1,0,0))
                edges.append((A, A_idx(0,1,0), 0,1,0))
                edges.append((A, A_idx(0,-1,0), 0,-1,0))
                edges.append((A, A_idx(1,-1,0), 1,-1,0))
                edges.append((A, A_idx(-1,1,0), -1,1,0))
                # A→B below (3)
                edges.append((A, B_idx(0,0,-1), 0,0,-1))
                edges.append((A, B_idx(-1,0,-1), -1,0,-1))
                edges.append((A, B_idx(0,-1,-1), 0,-1,-1))

                # B-B in-plane (6)
                edges.append((B, B_idx(1,0,0), 1,0,0))
                edges.append((B, B_idx(-1,0,0), -1,0,0))
                edges.append((B, B_idx(0,1,0), 0,1,0))
                edges.append((B, B_idx(0,-1,0), 0,-1,0))
                edges.append((B, B_idx(1,-1,0), 1,-1,0))
                edges.append((B, B_idx(-1,1,0), -1,1,0))
                # B→A below (3)
                edges.append((B, A_idx(0,0,-1), 0,0,-1))
                edges.append((B, A_idx(-1,0,-1), -1,0,-1))
                edges.append((B, A_idx(0,-1,-1), 0,-1,-1))

    return n, basis, edges


# =============================================================================
# For (3².4.3.4): use a direct construction with known topology
# =============================================================================

def build_32_4_3_4(L):
    """
    (3².4.3.4) elongated triangular tiling. z=5, g=3.

    We construct this using a rectangular unit cell.
    The tiling has alternating rows of triangles and squares.

    Using 4 vertices per cell with a rectangular Bravais lattice:
    a1 = (1, 0), a2 = (0, 1 + sqrt(3))

    Vertices per cell:
    V0 = (0, 0)
    V1 = (1/2, sqrt(3)/2)
    V2 = (1/2, sqrt(3)/2 + 1)
    V3 = (0, sqrt(3) + 1) = V0 of next cell (identified)

    Wait - V3 = V0(next). So only 3 distinct vertices per cell.
    But 3 verts with z=5 gives non-integer edges.

    Instead, we use a 4-vertex cell:
    a1 = (1, 0), a2 = (0, sqrt(3)+1)
    V0 = (0, 0)        [row A, triangle boundary]
    V1 = (1/2, sqrt(3)/2)  [row B, square bottom]
    V2 = (1/2, sqrt(3)/2 + 1) [row C, square top]
    V3 = (0, sqrt(3)+1) = V0(next cell up)

    Actually V3 IS V0 of the next cell. So we truly have 3 per cell.
    Since 3*5/2 = 7.5 is not integer, we need L to be even for integer edges.
    With L even: 3*L² vertices, 3*5*L²/2 = 15L²/2 edges. Still not integer if L² is odd.

    For L even: L² is divisible by 4. 15*L²/2 = 15*4k/2 = 30k. ✓
    For L odd: L² is odd. 15*L²/2 is not integer. ✗

    So we need even L. For odd L, we'd need a different cell.

    Alternative: use a 2-vertex cell with a different Bravais lattice.
    The centered rectangular lattice with:
    a1 = (1/2, (sqrt(3)+1)/2), a2 = (1/2, -(sqrt(3)+1)/2)

    This gives a cell area of (sqrt(3)+1)/2 and 2 vertices per cell.
    2*5/2 = 5 edges per cell. ✓

    Vertices: V0 = (0, 0), V1 = ... at some offset within the cell.

    Actually, I'll just construct the adjacency directly using the known structure.
    The elongated triangular tiling has rows:
    Row A (y=0): x = 0, 1, 2, ... (period 1)
    Row B (y=h): x = 1/2, 3/2, ... (h = sqrt(3)/2)
    Row B'(y=h+1): x = 1/2, 3/2, ...
    Row A' (y=2h+1): x = 0, 1, 2, ...

    Connections:
    A→B: triangular. A(i) connects to B(i) and B(i-1). [B(i) = B at x=i+1/2]
    B→B': square (vertical). B(i) connects to B'(i).
    B'→A': triangular. B'(i) connects to A'(i) and A'(i+1).

    Horizontal:
    A: NO horizontal edges (this is the key difference from my earlier attempt).
    B: horizontal edges. B(i) connects to B(i-1) and B(i+1).
    B': horizontal edges. B'(i) connects to B'(i-1) and B'(i+1).
    A': NO horizontal edges.

    Let me check degrees:
    A(i): B(i) [up-right], B(i-1) [up-left], B_prev(i) [down-right], B_prev(i-1) [down-left] = 4. z=4! ✗

    Still z=4 for A. The issue persists.

    OK let me try the OPPOSITE: A HAS horizontal edges, B does NOT.
    A: horizontal. A(i)→A(i±1). Plus A→B (2 up) + A→B_prev (2 down) = 2+2+2 = 6. z=6. ✗

    What if B has NO horizontal edges?
    B(i): A(i) [down-right], A(i+1) [down-left], B'(i) [up],
          A_prev(i) [up-right?], A_prev(i+1) [up-left?]

    Wait, B is between A below and B' above.
    B(i) at (i+1/2, h):
    Below: A(i) at (i, 0) and A(i+1) at (i+1, 0). Distance = 1. ✓
    Above: B'(i) at (i+1/2, h+1). Distance = 1 (vertical). ✓
    No horizontal. No other connections.
    z = 2 + 1 = 3. Way too few!

    This doesn't work either. The fundamental problem is: with this row structure,
    A-type vertices always have z=4 or z=6, never z=5.

    RESOLUTION: The (3².4.3.4) tiling does NOT have this simple row structure.
    I'll skip this lattice and note it as a construction challenge.
    """
    # Placeholder - will be implemented correctly later
    # For now, use a simple construction that might not be perfectly (3².4.3.4)
    # but captures some of its properties.

    # Actually, let me try one more thing. The key insight I've been missing:
    # In the (3².4.3.4) tiling, the "A" row vertices DO have horizontal edges,
    # but they DON'T connect to both rows above and below.
    # Instead, they connect to 1 row above and 1 row below, plus 2 horizontal + 1 more.

    # Hmm, let me try:
    # A(i) connects to: A(i+1), A(i-1), B(i), B_prev(i), and one more.
    # The "one more" could be a diagonal connection to a non-adjacent row.

    # Actually, looking at this from the vertex figure perspective:
    # V connects to n1(0°), n2(60°), n3(120°), n4(210°), n5(270°).
    # n1 is horizontal, n2 and n3 go "up", n4 and n5 go "down".
    # So: 1 horizontal + 2 up + 2 down = 5.

    # This means each vertex has ONE horizontal connection (not 2).
    # The horizontal connection breaks the left-right symmetry.

    # Row structure:
    # Row 0: y=0, x=0, 1, 2, 3, ... Each vertex connects to the RIGHT only: A(i)→A(i+1).
    # Row 1: y=h, x=1/2, 3/2, ...
    # Row 2: y=h+1, x=1/2, 3/2, ...
    # Row 3: y=2h+1, x=0, 1, 2, 3, ... Each vertex connects to the RIGHT only.

    # A(i): A(i+1) [right], B(i) [up-right], B(i-1) [up-left? no, up is at 60° and 120°]
    # n1 at 0° → A(i+1). n2 at 60° → B(i+1/2, h) which is B at x=i+1/2...
    # Wait, from A(i) at (i,0), the neighbor at 60° is at (i+cos60°, sin60°) = (i+1/2, √3/2).
    # This is B at x=i+1/2, which is B(i). ✓
    # n3 at 120° → (i-1/2, √3/2) = B(i-1). ✓
    # n4 at 210° → (i-√3/2, -1/2). This is NOT in any standard row!
    # n5 at 270° → (i, -1). Also not in a standard row.

    # So A(i) connects to: A(i+1), B(i), B(i-1), and 2 vertices that are NOT in the
    # adjacent rows. These 2 vertices are at:
    # (i-√3/2, -1/2) and (i, -1).

    # These positions involve √3, which means they're NOT on the same grid as rows B and B'.
    # This confirms that the (3².4.3.4) tiling does NOT have a simple row structure
    # with rational coordinates.

    # I'll skip this lattice construction.
    raise NotImplementedError(
        "(3².4.3.4) lattice construction requires irrational coordinates "
        "that don't fit a simple periodic cell. Skipping."
    )


# =============================================================================
# Verification
# =============================================================================

def verify_lattice(n, edges, expected_z, name):
    adj = defaultdict(int)
    for e in edges:
        adj[e[0]] += 1
        adj[e[1]] += 1
    degrees = list(adj.values())
    if len(degrees) < n:
        degrees.extend([0] * (n - len(degrees)))
    degrees = np.array(degrees)
    ok = np.all(degrees == expected_z)
    print(f"  {'[OK]' if ok else '[WARN]'} {name}: z={expected_z}, "
          f"min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.2f}")
    return ok


def compute_girth(n, edges, name, max_verts=3000):
    adj = [[] for _ in range(n)]
    seen = set()
    for e in edges:
        key = (min(e[0],e[1]), max(e[0],e[1]))
        if key not in seen:
            seen.add(key)
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])

    girth = float('inf')
    step = max(1, n // max_verts)
    checked = 0
    for start in range(0, n, step):
        if len(adj[start]) < 2:
            continue
        checked += 1
        dist = {start: 0}
        parent = {start: -1}
        queue = deque([start])
        found = float('inf')
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d >= found:
                break
            for v in adj[u]:
                if v not in dist:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle = d + 1 + dist[v]
                    if cycle < found:
                        found = cycle
        if found < girth:
            girth = found
            if girth <= 3:
                break

    if girth == float('inf'):
        girth = -1
    print(f"  Girth of {name}: {girth} (checked {checked} vertices)")
    return int(girth)


def deduplicate_edges(edges):
    seen = set()
    result = []
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


# =============================================================================
# Monte Carlo simulation
# =============================================================================

def mc_gradient_scan(n, edge_list, n_samples, d):
    site_edges = [[] for _ in range(n)]
    for idx, e in enumerate(edge_list):
        site_edges[e[0]].append(idx)
        site_edges[e[1]].append(idx)

    edge_a = np.array([e[0] for e in edge_list], dtype=np.int64)
    edge_b = np.array([e[1] for e in edge_list], dtype=np.int64)
    if d == 2:
        offsets = np.array([(e[2], e[3]) for e in edge_list], dtype=np.float64)
    else:
        offsets = np.array([(e[2], e[3], e[4]) for e in edge_list], dtype=np.float64)

    p_samples = np.zeros(n_samples)

    for s in range(n_samples):
        rng = np.random.RandomState(42 + s)
        thresholds = rng.random(n)
        order = np.argsort(thresholds)

        uf = UnionFindWrap(n, d)
        occupied = np.zeros(n, dtype=bool)
        edge_used = np.zeros(len(edge_list), dtype=bool)

        p_wrap = 1.0
        for k in range(n):
            site = order[k]
            occupied[site] = True
            for eidx in site_edges[site]:
                other = edge_b[eidx] if edge_a[eidx] == site else edge_a[eidx]
                if occupied[other] and not edge_used[eidx]:
                    edge_used[eidx] = True
                    delta = offsets[eidx]
                    if edge_a[eidx] != site:
                        delta = -delta
                    uf.union(site, other, delta)
            if uf.wrapping:
                p_wrap = (k + 1) / n
                break
        p_samples[s] = p_wrap

    return p_samples


def fss_fit(pc_L_vals, L_values, nu):
    x = np.array(L_values, dtype=float)
    y = np.array(pc_L_vals, dtype=float)
    invL = x ** (-1.0 / nu)
    n = len(x)
    sx = np.sum(invL); sy = np.sum(y)
    sxx = np.sum(invL**2); sxy = np.sum(invL * y)
    denom = n * sxx - sx**2
    if abs(denom) < 1e-15:
        return np.mean(y), np.std(y)
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    if n >= 2:
        uncertainty = abs(y[-1] - y[-2])
    else:
        uncertainty = 0.01
    if n > 2:
        resid = y - (a * invL + b)
        se_resid = np.sqrt(np.sum(resid**2) / (n - 2))
        se_b = se_resid * np.sqrt(sxx / abs(denom))
        uncertainty = max(uncertainty, se_b)
    return b, max(uncertainty, 1e-6)


def run_simulation(name, build_func, expected_z, expected_g, d,
                   L_values, n_samples, nu, seed_base=0):
    print(f"\n{'='*70}")
    print(f"LATTICE: {name}  (z={expected_z}, d={d}, expected g={expected_g})")
    print(f"{'='*70}")

    try:
        L_girth = min(15, max(L_values)) if d == 2 else min(L_values)
        print(f"\nBuilding lattice (L={L_girth}) for verification...")
        t0 = time.time()
        n_g, basis_g, edges_g = build_func(L_girth)
        print(f"  Built in {time.time()-t0:.2f}s: {n_g} vertices, {len(edges_g)} raw edges")

        verify_lattice(n_g, edges_g, expected_z, name)
        girth = compute_girth(n_g, edges_g, name)
    except NotImplementedError as e:
        print(f"\n  SKIPPED: {e}")
        return None

    print(f"\nRunning MC simulation ({n_samples} samples/L)...")
    pc_L_vals = []

    for L in L_values:
        t0 = time.time()
        n, basis, raw_edges = build_func(L)
        edge_list = deduplicate_edges(raw_edges)
        build_t = time.time() - t0

        t1 = time.time()
        p_samples = mc_gradient_scan(n, edge_list, n_samples, d)
        sim_t = time.time() - t1

        pc_L = np.mean(p_samples)
        se = np.std(p_samples) / np.sqrt(n_samples)
        print(f"  L={L:3d}: N={n:6d}, E={len(edge_list):6d} | "
              f"pc(L)={pc_L:.5f} +/- {se:.5f} | "
              f"build={build_t:.1f}s sim={sim_t:.1f}s")
        pc_L_vals.append(pc_L)

    pc_inf, uncertainty = fss_fit(pc_L_vals, L_values, nu)
    print(f"\n  FSS (nu={nu}): pc(inf) = {pc_inf:.6f} +/- {uncertainty:.6f}")

    return {
        'name': name, 'z': expected_z, 'd': d, 'g': girth,
        'pc': pc_inf, 'uncertainty': uncertainty,
        'pc_L': pc_L_vals, 'L_values': L_values,
    }


def main():
    print("=" * 70)
    print("Monte Carlo Site Percolation — Supplementary Lattices")
    print("=" * 70)
    np.random.seed(42)
    results = []

    # 1. (3.12.12) — z=3, d=2, g=3
    r = run_simulation("(3.12.12)", build_3_12_12, 3, 3, 2,
                       [20, 30, 40, 50, 60], 500, 4/3)
    if r: results.append(r)

    # 2. (4.8.8) — z=3, d=2, g=4
    r = run_simulation("(4.8.8)", build_4_8_8, 3, 4, 2,
                       [20, 30, 40, 50, 60], 500, 4/3)
    if r: results.append(r)

    # 3. (3^2.4.3.4) — z=5, d=2, g=3
    r = run_simulation("(3^2.4.3.4)", build_32_4_3_4, 5, 3, 2,
                       [20, 30, 40, 50, 60], 500, 4/3)
    if r: results.append(r)

    # 4. (3.4.6.4) Kagome — z=4, d=2, g=3
    r = run_simulation("(3.4.6.4)", build_kagome, 4, 3, 2,
                       [20, 30, 40, 50, 60], 500, 4/3)
    if r: results.append(r)

    # 5. 3D HCP — z=12, d=3, g=3
    r = run_simulation("hcp", build_hcp_3d, 12, 3, 3,
                       [15, 20, 25, 30], 500, 0.63)
    if r: results.append(r)

    # Save CSV
    csv_path = f"{OUTPUT_DIR}/mc_supplement_dataset.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'z', 'd', 'g', 'pc', 'uncertainty', 'source'])
        for r in results:
            w.writerow([r['name'], r['z'], r['d'], r['g'],
                        f"{r['pc']:.6f}", f"{r['uncertainty']:.6f}", 'mc_simulation'])
    print(f"\nSaved: {csv_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Lattice':<16} {'z':>3} {'d':>2} {'g':>3} {'pc':>10} {'unc':>10}")
    print(f"{'-'*16} {'-'*3} {'-'*2} {'-'*3} {'-'*10} {'-'*10}")
    for r in results:
        print(f"{r['name']:<16} {r['z']:>3} {r['d']:>2} {r['g']:>3} "
              f"{r['pc']:>10.6f} {r['uncertainty']:>10.6f}")

    print(f"\nFINITE-SIZE SCALING DETAILS:")
    for r in results:
        print(f"  {r['name']}:")
        for L, pcL in zip(r['L_values'], r['pc_L']):
            print(f"    L={L:3d}  pc(L)={pcL:.6f}")
        print(f"    => pc(inf) = {r['pc']:.6f} +/- {r['uncertainty']:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
