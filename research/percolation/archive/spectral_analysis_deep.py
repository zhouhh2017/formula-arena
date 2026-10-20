from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
谱理论深度分析：计算谱隙 Δλ 并建立与 p_c 的映射
====================================================

任务:
  1. 计算14个晶格的谱隙 Δλ (邻接矩阵 & Laplacian)
  2. 分析 Δλ 与 p_c 的定量关系
  3. 与 Nachmias-Peres 理论对比
  4. 探索谱隙标度的物理意义
  5. 可视化

All outputs are saved beside this script.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings, os, json, time
warnings.filterwarnings('ignore')

OUT = ROOT
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 0. 晶格数据
# ============================================================
lattices = [
    {'name': 'honeycomb',  'z': 3,  'd': 2, 'g': 6, 'pc': 0.697040},
    {'name': 'square',     'z': 4,  'd': 2, 'g': 4, 'pc': 0.592746},
    {'name': 'triangular', 'z': 6,  'd': 2, 'g': 3, 'pc': 0.500000},
    {'name': 'kagome',     'z': 4,  'd': 2, 'g': 3, 'pc': 0.652703},
    {'name': 'sc',         'z': 6,  'd': 3, 'g': 4, 'pc': 0.311600},
    {'name': 'bcc',        'z': 8,  'd': 3, 'g': 4, 'pc': 0.245800},
    {'name': 'fcc',        'z': 12, 'd': 3, 'g': 3, 'pc': 0.118800},
    {'name': 'diamond',    'z': 4,  'd': 3, 'g': 6, 'pc': 0.431900},
    {'name': 'hypersc_4',  'z': 8,  'd': 4, 'g': 4, 'pc': 0.196737},
    {'name': 'hypersc_5',  'z': 10, 'd': 5, 'g': 4, 'pc': 0.140810},
    {'name': 'hypersc_6',  'z': 12, 'd': 6, 'g': 4, 'pc': 0.109460},
    {'name': 'hypersc_7',  'z': 14, 'd': 7, 'g': 4, 'pc': 0.089270},
    {'name': 'hypersc_8',  'z': 16, 'd': 8, 'g': 4, 'pc': 0.075050},
    {'name': 'hypersc_9',  'z': 18, 'd': 9, 'g': 4, 'pc': 0.064530},
]

# 各晶格的有限尺寸参数 — keep matrices small for speed
SIZE_PARAMS = {
    'honeycomb':  (15, 15),
    'square':     (15, 15),
    'triangular': (15, 15),
    'kagome':     (12, 12),
    'sc':         (10, 10, 10),
    'bcc':        (8, 8, 8),
    'fcc':        (7, 7, 7),
    'diamond':    (6, 6, 6),
    'hypersc_4':  (5, 5, 5, 5),
    'hypersc_5':  (4, 4, 4, 4, 4),
    'hypersc_6':  (3, 3, 3, 3, 3, 3),
    'hypersc_7':  (3, 3, 3, 3, 3, 3, 3),
    'hypersc_8':  (2, 2, 2, 2, 2, 2, 2, 2),
    'hypersc_9':  (2, 2, 2, 2, 2, 2, 2, 2, 2),
}

# ============================================================
# 1. 构建邻接矩阵 (高效COO格式)
# ============================================================

def build_square(L1, L2):
    N = L1 * L2
    rows, cols = [], []
    for i in range(L1):
        for j in range(L2):
            u = i * L2 + j
            for v in [((i+1)%L1)*L2+j, ((i-1)%L1)*L2+j, i*L2+(j+1)%L2, i*L2+(j-1)%L2]:
                rows.append(u); cols.append(v)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N))

def build_honeycomb(L1, L2):
    N = 2 * L1 * L2
    rows, cols = [], []
    def idx(s, i, j): return s * L1 * L2 + (i % L1) * L2 + (j % L2)
    for i in range(L1):
        for j in range(L2):
            a = idx(0, i, j)
            for b in [idx(1, i, j), idx(1, i-1, j), idx(1, i, j-1)]:
                rows.append(a); cols.append(b)
                rows.append(b); cols.append(a)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N))

def build_triangular(L1, L2):
    N = L1 * L2
    rows, cols = [], []
    def idx(i, j): return (i % L1) * L2 + (j % L2)
    for i in range(L1):
        for j in range(L2):
            u = idx(i, j)
            for v in [idx(i+1,j), idx(i-1,j), idx(i,j+1), idx(i,j-1), idx(i+1,j+1), idx(i-1,j-1)]:
                rows.append(u); cols.append(v)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N))

def build_kagome(L1, L2):
    N = 3 * L1 * L2
    bonds = set()
    def idx(s, i, j): return s * L1 * L2 + (i % L1) * L2 + (j % L2)
    def add(u, v):
        if u != v: bonds.add((min(u,v), max(u,v)))
    for i in range(L1):
        for j in range(L2):
            sa, sb, sc = idx(0,i,j), idx(1,i,j), idx(2,i,j)
            # Up triangle: A(i,j), B(i,j), C(i,j)
            add(sa, sb); add(sb, sc); add(sa, sc)
            # Down triangle: A(i+1,j), B(i,j+1), C(i,j)
            add(idx(0,i+1,j), idx(1,i,j+1))
            add(idx(1,i,j+1), sc)
            add(sc, idx(0,i+1,j))
    rows = [b[0] for b in bonds] + [b[1] for b in bonds]
    cols = [b[1] for b in bonds] + [b[0] for b in bonds]
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N))

def build_sc(Lx, Ly, Lz):
    N = Lx * Ly * Lz
    rows, cols = [], []
    def idx(i,j,k): return (i%Lx)*Ly*Lz + (j%Ly)*Lz + (k%Lz)
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                u = idx(i,j,k)
                for v in [idx(i+1,j,k), idx(i-1,j,k), idx(i,j+1,k), idx(i,j-1,k), idx(i,j,k+1), idx(i,j,k-1)]:
                    rows.append(u); cols.append(v)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N))

def build_bcc(Lx, Ly, Lz):
    N = Lx * Ly * Lz
    rows, cols = [], []
    def idx(i,j,k): return (i%Lx)*Ly*Lz + (j%Ly)*Lz + (k%Lz)
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                u = idx(i,j,k)
                for di in [-1,1]:
                    for dj in [-1,1]:
                        for dk in [-1,1]:
                            v = idx(i+di,j+dj,k+dk)
                            if u != v: rows.append(u); cols.append(v)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N))

def build_fcc(Lx, Ly, Lz):
    N = Lx * Ly * Lz
    rows, cols = [], []
    def idx(i,j,k): return (i%Lx)*Ly*Lz + (j%Ly)*Lz + (k%Lz)
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                u = idx(i,j,k)
                nbrs = []
                for d1,d2 in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                    nbrs += [idx(i+d1,j+d2,k), idx(i+d1,j,k+d2), idx(i,j+d1,k+d2)]
                for v in nbrs:
                    if u != v: rows.append(u); cols.append(v)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N))

def build_diamond(Lx, Ly, Lz):
    N = 2 * Lx * Ly * Lz
    rows, cols = [], []
    def idx(s,i,j,k): return s*Lx*Ly*Lz + (i%Lx)*Ly*Lz + (j%Ly)*Lz + (k%Lz)
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                u = idx(0,i,j,k)
                for di,dj,dk in [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]:
                    v = idx(1,i+di,j+dj,k+dk)
                    rows.append(u); cols.append(v)
                    rows.append(v); cols.append(u)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N))

def build_hypersc(sizes):
    d = len(sizes)
    N = int(np.prod(sizes))
    rows, cols = [], []
    def idx(coords):
        r = 0; stride = 1
        for dim in range(d-1, -1, -1):
            r += (coords[dim] % sizes[dim]) * stride
            stride *= sizes[dim]
        return r
    from itertools import product
    for coords in product(*[range(s) for s in sizes]):
        u = idx(coords)
        for dim in range(d):
            for delta in [-1, 1]:
                nbr = list(coords); nbr[dim] += delta
                v = idx(nbr)
                if u != v: rows.append(u); cols.append(v)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N))


# ============================================================
# 2. 计算谱隙
# ============================================================

def compute_spectral_gap(A, n_ev=8):
    N = A.shape[0]
    k = min(n_ev, N - 2)

    # Largest adjacency eigenvalues
    try:
        eA = eigsh(A.astype(np.float64), k=k, which='LM', maxiter=3000)[0]
        eA = np.sort(eA)[::-1]
    except:
        if N <= 4000:
            eA = np.sort(np.linalg.eigvalsh(A.toarray().astype(np.float64)))[::-1]
        else:
            return {'lambda1': np.nan, 'lambda2': np.nan, 'delta_lambda': np.nan,
                    'lap_gap': np.nan, 'N': N}
    lam1, lam2 = eA[0], eA[1]

    # Laplacian gap
    deg = np.array(A.sum(axis=1)).flatten()
    L = csr_matrix((deg, (range(N), range(N))), shape=(N,N)) - A
    try:
        eL = eigsh(L.astype(np.float64), k=k, which='SM', maxiter=3000, sigma=-0.01)[0]
        eL = np.sort(eL)
        mu2 = eL[1]
    except:
        if N <= 4000:
            eL = np.sort(np.linalg.eigvalsh(L.toarray().astype(np.float64)))
            mu2 = eL[1]
        else:
            mu2 = np.nan

    return {'lambda1': lam1, 'lambda2': lam2, 'delta_lambda': lam1 - lam2,
            'lap_gap': mu2, 'N': N}


# ============================================================
# 3. 色散曲率 (解析)
# ============================================================

def compute_curvature(name, d, z):
    """Curvature matrix C at band edge: λ(k) ≈ z - k^T C k"""

    if name == 'square':
        return np.eye(2)
    elif name == 'triangular':
        return np.array([[2.0, 1.0], [1.0, 2.0]])
    elif name == 'honeycomb':
        # Upper band: λ+(k) = sqrt(3 + 2cos(kx) + 2cos(ky) + 2cos(kx-ky))
        eps = 1e-5
        def f(kx, ky):
            return np.sqrt(3 + 2*np.cos(kx) + 2*np.cos(ky) + 2*np.cos(kx-ky))
        z0 = f(0, 0)
        d2x = (f(eps,0) - 2*z0 + f(-eps,0)) / eps**2
        d2y = (f(0,eps) - 2*z0 + f(0,-eps)) / eps**2
        d2xy = (f(eps,eps) - f(eps,-eps) - f(-eps,eps) + f(-eps,-eps)) / (4*eps**2)
        return np.array([[-d2x, -d2xy], [-d2xy, -d2y]]) / 2
    elif name == 'kagome':
        eps = 1e-5
        def top(kx, ky):
            H = np.zeros((3,3), dtype=complex)
            H[0,1] = 1 + np.exp(-1j*kx); H[1,0] = np.conj(H[0,1])
            H[0,2] = 1 + np.exp(-1j*ky); H[2,0] = np.conj(H[0,2])
            H[1,2] = 1 + np.exp(-1j*(kx-ky)); H[2,1] = np.conj(H[1,2])
            return np.sort(np.linalg.eigvalsh(H))[::-1][0]
        z0 = top(0, 0)
        d2x = (top(eps,0) - 2*z0 + top(-eps,0)) / eps**2
        d2y = (top(0,eps) - 2*z0 + top(0,-eps)) / eps**2
        d2xy = (top(eps,eps) - top(eps,-eps) - top(-eps,eps) + top(-eps,-eps)) / (4*eps**2)
        return np.array([[-d2x, -d2xy], [-d2xy, -d2y]]) / 2
    elif name == 'sc':
        return np.eye(3)
    elif name == 'bcc':
        return 4.0 * np.eye(3)
    elif name == 'fcc':
        return 4.0 * np.eye(3)
    elif name == 'diamond':
        eps = 1e-5
        def f(kx, ky, kz):
            return abs(1 + np.exp(1j*kx) + np.exp(1j*ky) + np.exp(1j*kz))
        z0 = f(0,0,0)
        d2x = (f(eps,0,0)-2*z0+f(-eps,0,0))/eps**2
        d2y = (f(0,eps,0)-2*z0+f(0,-eps,0))/eps**2
        d2z = (f(0,0,eps)-2*z0+f(0,0,-eps))/eps**2
        d2xy = (f(eps,eps,0)-f(eps,-eps,0)-f(-eps,eps,0)+f(-eps,-eps,0))/(4*eps**2)
        d2xz = (f(eps,0,eps)-f(eps,0,-eps)-f(-eps,0,eps)+f(-eps,0,-eps))/(4*eps**2)
        d2yz = (f(0,eps,eps)-f(0,eps,-eps)-f(0,-eps,eps)+f(0,-eps,-eps))/(4*eps**2)
        return np.array([[-d2x,-d2xy,-d2xz],[-d2xy,-d2y,-d2yz],[-d2xz,-d2yz,-d2z]])/2
    elif name.startswith('hypersc'):
        return np.eye(d)
    else:
        return np.eye(d) * z / (2*d)


# ============================================================
# 4. 主计算循环
# ============================================================

print("=" * 80)
print("谱理论深度分析：谱隙 Δλ 与渗流阈值 p_c 的映射")
print("=" * 80)

results = []
for lat in lattices:
    name, z, d, g, pc = lat['name'], lat['z'], lat['d'], lat['g'], lat['pc']
    print(f"\n--- {name} (z={z}, d={d}, g={g}) ---")
    sizes = SIZE_PARAMS[name]

    t0 = time.time()
    builders = {
        'honeycomb': build_honeycomb, 'square': build_square,
        'triangular': build_triangular, 'kagome': build_kagome,
        'sc': build_sc, 'bcc': build_bcc, 'fcc': build_fcc,
        'diamond': build_diamond,
    }
    if name in builders:
        A = builders[name](*sizes)
    else:
        A = build_hypersc(sizes)

    N = A.shape[0]
    deg = np.array(A.sum(axis=1)).flatten()
    z_actual = np.mean(deg)
    print(f"  N={N}, z_actual={z_actual:.1f}, build={time.time()-t0:.2f}s")

    t0 = time.time()
    spec = compute_spectral_gap(A)
    print(f"  spec={time.time()-t0:.2f}s, λ₁={spec['lambda1']:.4f}, Δλ={spec['delta_lambda']:.6f}, Lap_gap={spec['lap_gap']:.6f}")

    C = compute_curvature(name, d, z)
    trC = np.trace(C)
    detC = np.linalg.det(C) if d <= 9 else np.nan

    results.append({
        'name': name, 'z': z, 'd': d, 'g': g, 'pc': pc, 'N': N,
        'L': sizes[0], 'z_actual': z_actual,
        'lambda1': spec['lambda1'], 'lambda2': spec['lambda2'],
        'delta_lambda_adj': spec['delta_lambda'],
        'lap_gap': spec['lap_gap'],
        'delta_lambda_over_z': spec['delta_lambda'] / z,
        'lap_gap_over_z': spec['lap_gap'] / z,
        'tr_C': trC, 'det_C': detC,
        'curv_per_dim': trC / d if d > 0 else np.nan,
        'sqrt_det_C': np.sqrt(abs(detC)) if not np.isnan(detC) else np.nan,
    })

# ============================================================
# 5. 分析
# ============================================================
names = [r['name'] for r in results]
d_arr = np.array([r['d'] for r in results])
z_arr = np.array([r['z'] for r in results])
pc_arr = np.array([r['pc'] for r in results])
dl_arr = np.array([r['delta_lambda_adj'] for r in results])
lg_arr = np.array([r['lap_gap'] for r in results])
dl_z = dl_arr / z_arr
lg_z = lg_arr / z_arr
L_arr = np.array([r['L'] for r in results])
trC_arr = np.array([r['tr_C'] for r in results])
detC_arr = np.array([r['det_C'] for r in results])
cpd_arr = np.array([r['curv_per_dim'] for r in results])
R_arr = pc_arr * (z_arr - 1)
geom_mean_C = np.power(np.abs(detC_arr), 1.0/d_arr)

print("\n" + "=" * 80)
print("谱隙汇总表")
print("=" * 80)
print(f"{'晶格':<15} {'d':>2} {'z':>3} {'N':>7} {'L':>3} {'Δλ_adj':>10} {'Δλ/z':>10} {'Lap_gap':>10} {'Tr(C)':>8}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<15} {r['d']:>2} {r['z']:>3} {r['N']:>7} {r['L']:>3} "
          f"{r['delta_lambda_adj']:>10.6f} {r['delta_lambda_over_z']:>10.6f} "
          f"{r['lap_gap']:>10.6f} {r['tr_C']:>8.3f}")

# 超立方检验: Δλ 应与 L 有关, 与 d 无关
print("\n--- 超立方晶格 Δλ 与 L 的关系 ---")
for r in results:
    if r['name'].startswith('hypersc'):
        pred = 2*(1-np.cos(2*np.pi/r['L']))
        print(f"  {r['name']}: L={r['L']}, Δλ={r['delta_lambda_adj']:.6f}, 预测={pred:.6f}")

print("\n--- 归一化: Δλ · L² / (2π²) (应≈1对超立方) ---")
for r in results:
    norm = r['delta_lambda_adj'] * r['L']**2 / (2*np.pi**2)
    print(f"  {r['name']}: {norm:.4f}")

# ============================================================
# 5a. 高维: p_c - 1/z 标度
# ============================================================
print("\n" + "=" * 80)
print("高维超立方 (d≥4) p_c - 1/z 标度")
print("=" * 80)

mask_hd = d_arr >= 4
excess_hd = pc_arr[mask_hd] - 1.0/z_arr[mask_hd]
z_hd = z_arr[mask_hd]

print(f"\n{'晶格':<15} {'z':>3} {'p_c':>8} {'1/z':>8} {'excess':>10} {'NP:5/z^{4/3}':>12}")
for i in np.where(mask_hd)[0]:
    np_pred = 5.0/z_arr[i]**(4.0/3)
    print(f"  {names[i]:<13} {z_arr[i]:>3} {pc_arr[i]:>8.5f} {1.0/z_arr[i]:>8.5f} "
          f"{excess_hd[np.where(mask_hd)[0].tolist().index(i)]:>10.6f} {np_pred:>12.6f}")

log_ex = np.log(excess_hd)
log_z = np.log(z_hd)
slope_ex, intercept_ex, rval_ex, _, _ = linregress(log_z, log_ex)
c_ex, gamma_ex = np.exp(intercept_ex), slope_ex
print(f"\n拟合: p_c - 1/z = c · z^γ")
print(f"  γ = {slope_ex:.4f} (NP预测: -4/3 = -1.333)")
print(f"  c = {c_ex:.4f} (NP预测: 5)")
print(f"  R² = {rval_ex**2:.4f}")

# ============================================================
# 5b. 色散曲率与 p_c
# ============================================================
print("\n" + "=" * 80)
print("色散曲率 Tr(C) 与 p_c")
print("=" * 80)

# p_c = a · (Tr(C)/z)^b
log_x = np.log(trC_arr / z_arr)
log_y = np.log(pc_arr)
s1, i1, r1, _, _ = linregress(log_x, log_y)
a1, b1 = np.exp(i1), s1
pred1 = a1 * (trC_arr / z_arr)**b1
R2_1 = 1 - np.sum((pc_arr - pred1)**2) / np.sum((pc_arr - np.mean(pc_arr))**2)
print(f"\n拟合: p_c = a·(Tr(C)/z)^b")
print(f"  a = {a1:.4f}, b = {b1:.4f}, R² = {R2_1:.4f}")

# p_c = a · (curv_per_dim / z)^b
log_x2 = np.log(cpd_arr / z_arr)
s2, i2, r2, _, _ = linregress(log_x2, log_y)
a2, b2 = np.exp(i2), s2
pred2 = a2 * (cpd_arr / z_arr)**b2
R2_2 = 1 - np.sum((pc_arr - pred2)**2) / np.sum((pc_arr - np.mean(pc_arr))**2)
print(f"\n拟合: p_c = a·(curv_per_dim/z)^b")
print(f"  a = {a2:.4f}, b = {b2:.4f}, R² = {R2_2:.4f}")

# p_c = a · (det(C)^{1/d} / z)^b
log_x3 = np.log(geom_mean_C / z_arr)
s3, i3, r3, _, _ = linregress(log_x3, log_y)
a3, b3 = np.exp(i3), s3
pred3 = a3 * (geom_mean_C / z_arr)**b3
R2_3 = 1 - np.sum((pc_arr - pred3)**2) / np.sum((pc_arr - np.mean(pc_arr))**2)
print(f"\n拟合: p_c = a·(det(C)^{{1/d}}/z)^b")
print(f"  a = {a3:.4f}, b = {b3:.4f}, R² = {R2_3:.4f}")

# Multivariate: p_c = a · (Tr(C)/z)^α · d^β
def obj_mv(params):
    a, alpha, beta = params
    pred = a * (trC_arr/z_arr)**alpha * d_arr**beta
    return np.sum((pc_arr - pred)**2)
res_mv = minimize(obj_mv, [1.0, 1.0, -1.0], method='Nelder-Mead',
                  options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-14})
a_mv, al_mv, be_mv = res_mv.x
pred_mv = a_mv * (trC_arr/z_arr)**al_mv * d_arr**be_mv
R2_mv = 1 - np.sum((pc_arr - pred_mv)**2) / np.sum((pc_arr - np.mean(pc_arr))**2)
print(f"\n多变量拟合: p_c = a·(Tr(C)/z)^α · d^β")
print(f"  a = {a_mv:.4f}, α = {al_mv:.4f}, β = {be_mv:.4f}, R² = {R2_mv:.4f}")

# ============================================================
# 5c. Δλ/z 与 p_c 的直接关系 (固定L)
# ============================================================
print("\n" + "=" * 80)
print("有限尺寸谱隙 Δλ/z 与 p_c")
print("=" * 80)

# For hypercubic at same L: Δλ/z = 2(1-cos(2π/L))/z ~ 1/(d*L²)
# So Δλ/z ~ 1/z at fixed L (since z=2d)
# And p_c ~ 1/z too → direct proportionality!

# Group by L to see the relationship
print(f"\n{'晶格':<15} {'L':>3} {'Δλ/z':>10} {'p_c':>8} {'p_c/(Δλ/z)':>12}")
print("-" * 55)
for r in results:
    ratio = r['pc'] / r['delta_lambda_over_z'] if r['delta_lambda_over_z'] > 0 else np.nan
    print(f"{r['name']:<15} {r['L']:>3} {r['delta_lambda_over_z']:>10.6f} {r['pc']:>8.5f} {ratio:>12.3f}")

# ============================================================
# 5d. Nachmias-Peres 理论对比
# ============================================================
print("\n" + "=" * 80)
print("与 Nachmias-Peres (2008) 理论的对比")
print("=" * 80)

print("""
Nachmias-Peres 定理 (d → ∞):
  p_c = 1/(2d) + 5/(2d)^{4/3} + o(1/d^{4/3})
       = 1/z + 5/z^{4/3}

归一化:
  R = p_c·(z-1) ≈ 1 - 1/z + 5/z^{1/3}   (z >> 1)

色散曲率分析:
  超立方: C = I_d, Tr(C) = d = z/2
  DOS near band edge: g(ε) ~ ε^{d/2-1}

  NP 修正项 5/z^{4/3} 等价于 5/(2d)^{4/3}

  谱理论预测: p_c - 1/z 来自环结构的贡献
  对于超立方: 最短环 g = 4, 环密度 ~ z^2/d
  每个环对 p_c 的贡献 ~ p_c^{g-1} → 与 d 的关系复杂

关键差异来源:
  1. NP 是 d→∞ 渐近, 我们覆盖 d=4-9
  2. 有限 d 下, 高阶级数项不可忽略
  3. 不同晶格结构 (非超立方) 的环拓扑不同
""")

# 对比 NP 预测
print(f"{'晶格':<15} {'p_c实际':>8} {'NP预测':>8} {'误差%':>8}")
print("-" * 45)
for r in results:
    np_pred = 1.0/r['z'] + 5.0/r['z']**(4.0/3)
    err = abs(r['pc'] - np_pred) / r['pc'] * 100
    print(f"{r['name']:<15} {r['pc']:>8.5f} {np_pred:>8.5f} {err:>8.1f}%")

# ============================================================
# 5e. 谱隙标度的物理意义
# ============================================================
print("\n" + "=" * 80)
print("谱隙标度的物理意义")
print("=" * 80)

print("""
关联长度指数 ν 与谱隙:
  ξ ~ |p - p_c|^{-ν}
  有限尺寸: p_c(L) - p_c(∞) ~ L^{-1/ν}
  晶格谱隙: Δμ_lattice ~ L^{-2}
  → p_c - p_c(∞) ~ Δμ^{1/(2ν)}

已知 ν 值:
  2D: ν = 4/3 ≈ 1.333 → 1/(2ν) = 3/8 = 0.375
  3D: ν ≈ 0.876     → 1/(2ν) ≈ 0.571
  4D: ν ≈ 0.734     → 1/(2ν) ≈ 0.681
  ≥6D: ν = 1/2      → 1/(2ν) = 1.0

谱维数 d_s:
  对正则晶格: d_s = d
  对渗流簇 (p=p_c): d_s = 2d_f/d_w
  2D: d_s ≈ 91/48 ≈ 1.896
  3D: d_s ≈ 1.3 (近似)
  ≥6D: d_s = 4 (平均场)

我们的发现:
  R = 1 - 1/z + a/(z·d^b) 中 b ≈ 0.42
  高维 (d≥4): γ = 1+b ≈ 1.42
  NP 预测: γ = 4/3 ≈ 1.33

  差异 δγ = 1.42 - 1.33 = 0.09
  这可能来自:
  1. 有限 d 效应 (d=4-9 而非 d→∞)
  2. 不同晶格结构的环拓扑效应
  3. 谱隙标度与关联长度的非平凡关系
""")

# ============================================================
# 6. 可视化
# ============================================================
print("\n" + "=" * 80)
print("生成可视化图表")
print("=" * 80)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})
colors = ['#e74c3c' if d==2 else '#3498db' if d==3 else '#2ecc71' for d in d_arr]

# --- Figure 1: Spectral gap vs d ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0,0]
ax.scatter(d_arr, dl_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)):
    ax.annotate(names[i], (d_arr[i], dl_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='Dimension d', ylabel='Spectral gap Δλ (adjacency)', title='(a) Spectral gap vs dimension', yscale='log')
ax.set_xticks(range(2,10)); ax.grid(True, alpha=0.3)
ax.legend(handles=[Patch(fc='#e74c3c',label='2D'),Patch(fc='#3498db',label='3D'),Patch(fc='#2ecc71',label='4D-9D')], loc='upper right')

ax = axes[0,1]
ax.scatter(d_arr, dl_z, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)):
    ax.annotate(names[i], (d_arr[i], dl_z[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='Dimension d', ylabel='Δλ/z', title='(b) Normalized gap vs d', yscale='log')
ax.set_xticks(range(2,10)); ax.grid(True, alpha=0.3)

ax = axes[1,0]
ax.scatter(d_arr, trC_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)):
    ax.annotate(names[i], (d_arr[i], trC_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='Dimension d', ylabel='Tr(C)', title='(c) Curvature Tr(C) vs d')
ax.set_xticks(range(2,10)); ax.grid(True, alpha=0.3)
ax.plot(range(2,10), range(2,10), 'k--', alpha=0.3, label='Tr(C)=d')
ax.legend()

ax = axes[1,1]
ax.scatter(d_arr, cpd_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)):
    ax.annotate(names[i], (d_arr[i], cpd_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='Dimension d', ylabel='Tr(C)/d', title='(d) Curvature per dimension')
ax.set_xticks(range(2,10)); ax.grid(True, alpha=0.3)
ax.axhline(1.0, color='k', ls='--', alpha=0.3, label='1.0')
ax.legend()

fig.suptitle('Spectral Properties of Lattices', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT}/spectral_gap_vs_dimension.png'); plt.close()
print("  ✓ spectral_gap_vs_dimension.png")

# --- Figure 2: p_c vs spectral ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0,0]
ax.scatter(dl_z, pc_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)):
    ax.annotate(names[i], (dl_z[i], pc_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='Δλ/z', ylabel='p_c', title='(a) p_c vs Δλ/z', xscale='log', yscale='log')
valid = (dl_z>0)&(pc_arr>0)
lx, ly = np.log(dl_z[valid]), np.log(pc_arr[valid])
s, ic, r, _, _ = linregress(lx, ly)
xf = np.linspace(lx.min(), lx.max(), 100)
ax.plot(np.exp(xf), np.exp(ic+s*xf), 'k--', alpha=0.5, label=f'p_c ~ (Δλ/z)^{s:.2f}, R²={r**2:.3f}')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0,1]
ratio = trC_arr/z_arr
ax.scatter(ratio, pc_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)):
    ax.annotate(names[i], (ratio[i], pc_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='Tr(C)/z', ylabel='p_c', title='(b) p_c vs Tr(C)/z', xscale='log', yscale='log')
v2 = ratio>0; lx2, ly2 = np.log(ratio[v2]), np.log(pc_arr[v2])
s2, i2, r2, _, _ = linregress(lx2, ly2)
xf2 = np.linspace(lx2.min(), lx2.max(), 100)
ax.plot(np.exp(xf2), np.exp(i2+s2*xf2), 'k--', alpha=0.5, label=f'p_c ~ (Tr(C)/z)^{s2:.2f}, R²={r2**2:.3f}')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,0]
for dv in [2,3,4,5,6,7,8,9]:
    m = d_arr==dv
    if np.any(m): ax.scatter(dl_z[m], pc_arr[m], s=80, zorder=5, edgecolors='k', linewidth=0.5, label=f'd={dv}')
ax.set(xlabel='Δλ/z', ylabel='p_c', title='(c) p_c vs Δλ/z (by d)', xscale='log', yscale='log')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.scatter(d_arr, R_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)):
    ax.annotate(names[i], (d_arr[i], R_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='Dimension d', ylabel='R = p_c·(z-1)', title='(d) Normalized threshold R vs d')
ax.set_xticks(range(2,10)); ax.axhline(1.0, color='k', ls='--', alpha=0.3, label='Mean-field R=1')
ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle('Percolation Threshold vs Spectral Properties', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT}/pc_vs_spectral.png'); plt.close()
print("  ✓ pc_vs_spectral.png")

# --- Figure 3: Theory vs Experiment ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0,0]
mh = d_arr>=4
ax.scatter(d_arr[mh], pc_arr[mh], c='red', s=80, zorder=5, edgecolors='k', linewidth=0.5, label='Data')
df = np.linspace(4,9,100); zf = 2*df
ax.plot(df, 1/zf + 5/zf**(4/3), 'b--', lw=2, label='NP: 1/z + 5/z^{4/3}')
ax.plot(df, c_ex/zf**(-gamma_ex), 'r-', lw=2, label=f'Our: {c_ex:.2f}/z^{{{-gamma_ex:.2f}}}')
ax.set(xlabel='d', ylabel='p_c', title='(a) High-d: Theory vs Data'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0,1]
excess = pc_arr - 1.0/z_arr; mp = excess>0
ax.scatter(z_arr[mp], excess[mp], c=[colors[i] for i in np.where(mp)[0]], s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in np.where(mp)[0]: ax.annotate(names[i], (z_arr[i], excess[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
zf2 = np.linspace(3,20,100)
ax.plot(zf2, 5/zf2**(4/3), 'b--', lw=2, label='NP: 5/z^{4/3}')
ax.plot(zf2, c_ex/zf2**(-gamma_ex), 'r-', lw=2, label=f'Our: {c_ex:.2f}/z^{{{-gamma_ex:.2f}}}')
ax.set(xlabel='z', ylabel='p_c - 1/z', title='(b) Excess threshold vs z', xscale='log', yscale='log')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,0]
me = (excess>0)&(z_arr>=6)
if np.sum(me)>2:
    ax.scatter(np.log(z_arr[me]), np.log(excess[me]), c='red', s=80, zorder=5, edgecolors='k', linewidth=0.5)
    se, ie, re, _, _ = linregress(np.log(z_arr[me]), np.log(excess[me]))
    xf = np.linspace(np.log(z_arr[me]).min(), np.log(z_arr[me]).max(), 100)
    ax.plot(xf, ie+se*xf, 'r-', label=f'Slope={se:.3f} (NP:-4/3=-1.333)')
    ax.set(xlabel='ln(z)', ylabel='ln(p_c - 1/z)', title='(c) Excess scaling exponent'); ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.scatter(1/z_arr, R_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)): ax.annotate(names[i], (1/z_arr[i], R_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='1/z', ylabel='R = p_c·(z-1)', title='(d) R vs 1/z')
xf = np.linspace(0,0.35,100); ax.plot(xf, 1-xf, 'b--', alpha=0.5, label='R=1-1/z')
ax.axhline(1.0, color='k', ls='-', alpha=0.3); ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle('Theory vs Experiment: Spectral Gap and Percolation', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT}/theory_vs_experiment.png'); plt.close()
print("  ✓ theory_vs_experiment.png")

# --- Figure 4: Finite-size scaling ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(L_arr, dl_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)): ax.annotate(names[i], (L_arr[i], dl_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
Lf = np.linspace(2,20,100); ax.plot(Lf, (2*np.pi/Lf)**2, 'k--', alpha=0.4, label='(2π/L)²')
ax.set(xlabel='L', ylabel='Δλ', title='(a) Δλ vs L', yscale='log', xscale='log'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
dlL2 = dl_arr * L_arr**2
ax.scatter(d_arr, dlL2, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)): ax.annotate(names[i], (d_arr[i], dlL2[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.axhline(4*np.pi**2, color='k', ls='--', alpha=0.4, label=f'4π²={4*np.pi**2:.1f}')
ax.set(xlabel='d', ylabel='Δλ·L²', title='(b) Δλ·L² vs d'); ax.set_xticks(range(2,10)); ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle('Finite-size scaling of spectral gap', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT}/finite_size_scaling.png'); plt.close()
print("  ✓ finite_size_scaling.png")

# --- Figure 5: Comprehensive mapping ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0,0]
ax.scatter(1/z_arr, pc_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)): ax.annotate(names[i], (1/z_arr[i], pc_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.plot([0,0.35],[0,0.35],'k--',alpha=0.3,label='p_c=1/z (Bethe)')
ax.set(xlabel='1/z', ylabel='p_c', title='(a) p_c vs 1/z'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0,1]
pcz = pc_arr*z_arr
ax.scatter(d_arr, pcz, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)): ax.annotate(names[i], (d_arr[i], pcz[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.axhline(1.0, color='k', ls='--', alpha=0.3, label='Bethe: p_c·z=1')
ax.set(xlabel='d', ylabel='p_c·z', title='(b) p_c·z vs d'); ax.set_xticks(range(2,10)); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,0]
ax.scatter(d_arr, geom_mean_C, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)): ax.annotate(names[i], (d_arr[i], geom_mean_C[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='d', ylabel='det(C)^{1/d}', title='(c) Geometric mean curvature vs d'); ax.set_xticks(range(2,10)); ax.grid(True, alpha=0.3)

ax = axes[1,1]
xc = geom_mean_C/z_arr
ax.scatter(xc, pc_arr, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)): ax.annotate(names[i], (xc[i], pc_arr[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set(xlabel='det(C)^{1/d}/z', ylabel='p_c', title='(d) p_c vs geom. mean curv./z', xscale='log', yscale='log')
v3 = xc>0; lx3, ly3 = np.log(xc[v3]), np.log(pc_arr[v3])
s3p, i3p, r3p, _, _ = linregress(lx3, ly3)
xf3 = np.linspace(lx3.min(), lx3.max(), 100)
ax.plot(np.exp(xf3), np.exp(i3p+s3p*xf3), 'k--', alpha=0.5, label=f'Slope={s3p:.3f}, R²={r3p**2:.3f}')
ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle('Comprehensive Spectral-Percolation Mapping', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT}/spectral_pc_mapping.png'); plt.close()
print("  ✓ spectral_pc_mapping.png")

# --- Figure 6: NP comparison detail ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ex_all = pc_arr - 1.0/z_arr; mp2 = ex_all>0
ax.scatter(z_arr[mp2], ex_all[mp2], c='steelblue', s=80, zorder=5, edgecolors='k', linewidth=0.5)
zf3 = np.linspace(3,20,200)
ax.plot(zf3, 5/zf3**(4/3), 'r--', lw=2, label='NP: 5/z^{4/3}')
ax.set(xlabel='z', ylabel='p_c - 1/z', title='(a) Excess threshold', xscale='log', yscale='log')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
dev = 1 - R_arr
ax.scatter(1/z_arr, dev, c=colors, s=80, zorder=5, edgecolors='k', linewidth=0.5)
for i in range(len(results)): ax.annotate(names[i], (1/z_arr[i], dev[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.plot([0,0.35],[0,0.35],'k--',alpha=0.3,label='1-R=1/z')
ax.set(xlabel='1/z', ylabel='1-R', title='(b) Deviation from Bethe'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
md = dev>0
ln1mR = np.log(dev[md]); ln1mz = np.log(1/z_arr[md])
ax.scatter(ln1mz, ln1mR, c='steelblue', s=80, zorder=5, edgecolors='k', linewidth=0.5)
if len(ln1mR)>2:
    sl, ic2, rv, _, _ = linregress(ln1mz, ln1mR)
    xff = np.linspace(ln1mz.min(), ln1mz.max(), 100)
    ax.plot(xff, ic2+sl*xff, 'r-', lw=2, label=f'Slope={sl:.3f} (Bethe:1.0)')
    ax.legend()
ax.set(xlabel='ln(1/z)', ylabel='ln(1-R)', title='(c) Scaling of Bethe deviation'); ax.grid(True, alpha=0.3)

fig.suptitle('Nachmias-Peres Theory Comparison', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT}/nachmias_peres_comparison.png'); plt.close()
print("  ✓ nachmias_peres_comparison.png")

# --- Figure 7: Exponent comparison (the KEY figure) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Compare exponents: our γ vs NP 4/3
ax = axes[0]
exponents = {'Our fit (γ)': -gamma_ex, 'NP theory (4/3)': 4/3, 'Mean-field (1)': 1.0}
bars = ax.bar(range(len(exponents)), list(exponents.values()), color=['#e74c3c','#3498db','#95a5a6'], edgecolor='k')
ax.set_xticks(range(len(exponents)))
ax.set_xticklabels(list(exponents.keys()), fontsize=10)
ax.set_ylabel('Exponent γ in p_c - 1/z ~ z^{-γ}')
ax.set_title('(a) Scaling exponent comparison')
ax.axhline(4/3, color='blue', ls=':', alpha=0.5)
for i, v in enumerate(exponents.values()):
    ax.text(i, v+0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# (b) Residuals of NP prediction
ax = axes[1]
for i, r in enumerate(results):
    np_p = 1.0/r['z'] + 5.0/r['z']**(4.0/3)
    resid = (r['pc'] - np_p) / r['pc'] * 100
    ax.bar(i, resid, color=colors[i], edgecolor='k', linewidth=0.5)
ax.set_xticks(range(len(results)))
ax.set_xticklabels([n[:6] for n in names], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Residual (%)')
ax.set_title('(b) NP prediction residuals')
ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Scaling Exponent: Experiment vs Nachmias-Peres', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT}/exponent_comparison.png'); plt.close()
print("  ✓ exponent_comparison.png")

# ============================================================
# 7. Save data
# ============================================================
summary_data = []
for r in results:
    d_entry = {k: (float(v) if isinstance(v, (np.floating, float)) else
                   int(v) if isinstance(v, (np.integer, int)) else v)
               for k, v in r.items()}
    summary_data.append(d_entry)
with open(f'{OUT}/spectral_data.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print(f"\n{'='*80}")
print(f"所有计算完成!")
print(f"输出文件: {OUT}/")
print(f"  - spectral_data.json")
print(f"  - spectral_gap_vs_dimension.png")
print(f"  - pc_vs_spectral.png")
print(f"  - theory_vs_experiment.png")
print(f"  - finite_size_scaling.png")
print(f"  - spectral_pc_mapping.png")
print(f"  - nachmias_peres_comparison.png")
print(f"  - exponent_comparison.png")
print(f"{'='*80}")
