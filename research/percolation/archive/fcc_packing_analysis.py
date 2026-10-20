from pathlib import Path
ROOT = Path(__file__).resolve().parent
#!/usr/bin/env python3
"""
FCC 特殊性研究：引入堆积密度 φ 作为新物理量
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

# ============================================================
# 数据集 (包含堆积密度 φ)
# ============================================================
data = [
    {'name':'honeycomb',  'z':3,  'd':2,'g':6,'n_tri':0, 'pc':0.697040,'phi':0.30,'type':'standard'},
    {'name':'square',     'z':4,  'd':2,'g':4,'n_tri':0, 'pc':0.592746,'phi':0.52,'type':'standard'},
    {'name':'triangular', 'z':6,  'd':2,'g':3,'n_tri':6, 'pc':0.500000,'phi':0.91,'type':'standard'},
    {'name':'kagome',     'z':4,  'd':2,'g':3,'n_tri':2, 'pc':0.652703,'phi':0.46,'type':'standard'},
    {'name':'sc',         'z':6,  'd':3,'g':4,'n_tri':0, 'pc':0.311600,'phi':0.52,'type':'standard'},
    {'name':'bcc',        'z':8,  'd':3,'g':4,'n_tri':0, 'pc':0.245800,'phi':0.68,'type':'standard'},
    {'name':'fcc',        'z':12, 'd':3,'g':3,'n_tri':24,'pc':0.118800,'phi':0.74,'type':'standard'},
    {'name':'diamond',    'z':4,  'd':3,'g':6,'n_tri':0, 'pc':0.431900,'phi':0.34,'type':'standard'},
    {'name':'hypersc_4',  'z':8,  'd':4,'g':4,'n_tri':0, 'pc':0.196737,'phi':0.52,'type':'highD'},
    {'name':'hypersc_5',  'z':10, 'd':5,'g':4,'n_tri':0, 'pc':0.140810,'phi':0.52,'type':'highD'},
    {'name':'hypersc_6',  'z':12, 'd':6,'g':4,'n_tri':0, 'pc':0.109460,'phi':0.52,'type':'highD'},
    {'name':'hypersc_7',  'z':14, 'd':7,'g':4,'n_tri':0, 'pc':0.089270,'phi':0.52,'type':'highD'},
    {'name':'hypersc_8',  'z':16, 'd':8,'g':4,'n_tri':0, 'pc':0.075050,'phi':0.52,'type':'highD'},
    {'name':'hypersc_9',  'z':18, 'd':9,'g':4,'n_tri':0, 'pc':0.064530,'phi':0.52,'type':'highD'},
    {'name':'MC_sq',        'z':4, 'd':2,'g':4,'n_tri':0, 'pc':0.592746,'phi':0.52,'type':'MC'},
    {'name':'MC_sq+[1,2]',  'z':8, 'd':2,'g':3,'n_tri':4, 'pc':0.373700,'phi':0.52,'type':'MC'},
    {'name':'MC_sq+[1,4]',  'z':12,'d':2,'g':3,'n_tri':12,'pc':0.208500,'phi':0.52,'type':'MC'},
    {'name':'MC_sq+[1,3]',  'z':8, 'd':2,'g':4,'n_tri':0, 'pc':0.306500,'phi':0.52,'type':'MC'},
    {'name':'MC_sq+[1,2,3]','z':12,'d':2,'g':3,'n_tri':8, 'pc':0.264200,'phi':0.52,'type':'MC'},
]
df = pd.DataFrame(data)
df['zd'] = df['z'] * df['d']

names = df['name'].values
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
phi = df['phi'].values.astype(float)
pc = df['pc'].values.astype(float)
n = len(pc)
fcc_idx = 6  # fcc index
n_data = n

# ============================================================
# 拟合工具
# ============================================================

def fit(func, p0, z, d, g, phi, pc):
    """Simple Nelder-Mead fit"""
    def obj(p):
        try:
            pred = func(p, z, d, g, phi)
            if not np.all(np.isfinite(pred)):
                return 1e10
            return np.sum((pred - pc)**2)
        except:
            return 1e10

    # 3 random restarts
    best_r = None
    best_ss = np.inf
    for trial in range(3):
        p_init = np.array(p0, dtype=float)
        if trial > 0:
            p_init *= (1 + 0.2 * np.random.randn(len(p_init)))
        r = minimize(obj, p_init, method='Nelder-Mead',
                    options={'maxiter': 30000, 'xatol': 1e-12, 'fatol': 1e-12})
        if r.fun < best_ss and r.fun < 1e10:
            best_ss = r.fun
            best_r = r

    p = best_r.x
    pred = func(p, z, d, g, phi)
    ss_res = np.sum((pc - pred)**2)
    ss_tot = np.sum((pc - np.mean(pc))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pc - pred)**2))
    mae = np.mean(np.abs(pc - pred))
    max_err_rel = np.max(np.abs(pc - pred) / pc)
    aic = n_data * np.log(ss_res / n_data + 1e-30) + 2 * len(p)
    return {'params': p, 'R2': r2, 'RMSE': rmse, 'MAE': mae,
            'max_err_rel': max_err_rel, 'pred': pred, 'ss_res': ss_res,
            'aic': aic, 'k': len(p)}


# ============================================================
# 公式定义
# ============================================================

def F0(p, z, d, g, phi):
    """基准: p_c = A·z^{-α}·d^{-γ}·(1+B/g)"""
    return p[0] * z**(-p[1]) * d**(-p[2]) * (1 + p[3]/g)

def F1(p, z, d, g, phi):
    """φ幂律: p_c = A·z^{-α}·d^{-γ}·(1+B/g)·φ^β"""
    return p[0] * z**(-p[1]) * d**(-p[2]) * (1 + p[3]/g) * phi**p[4]

def F2(p, z, d, g, phi):
    """φ线性: p_c = A·z^{-α}·d^{-γ}·(1+B/g)·(1+C·(φ-0.52))"""
    return p[0] * z**(-p[1]) * d**(-p[2]) * (1 + p[3]/g) * (1 + p[4]*(phi - 0.52))

def F3(p, z, d, g, phi):
    """φ指数: p_c = A·z^{-α}·d^{-γ}·(1+B/g)·exp(λ·φ)"""
    return p[0] * z**(-p[1]) * d**(-p[2]) * (1 + p[3]/g) * np.exp(p[4]*phi)

def F4(p, z, d, g, phi):
    """zd合并: p_c = A·(zd)^{-α}·(1+B/g)·φ^β"""
    return p[0] * (z*d)**(-p[1]) * (1 + p[2]/g) * phi**p[3]

def F5(p, z, d, g, phi):
    """z-φ交互: p_c = A·z^{-α}·d^{-γ}·(1+B/g)·(1+δ·z·(φ-0.52))"""
    return p[0] * z**(-p[1]) * d**(-p[2]) * (1 + p[3]/g) * (1 + p[4]*z*(phi - 0.52))

def F6(p, z, d, g, phi):
    """自由体积: p_c = A·z^{-α}·d^{-γ}·(1+B/g)·((1-φ)/φ)^δ"""
    fv = (1 - phi) / phi
    return p[0] * z**(-p[1]) * d**(-p[2]) * (1 + p[3]/g) * fv**p[4]

# ============================================================
# 运行分析
# ============================================================

print("=" * 90)
print("  FCC 特殊性研究：引入堆积密度 φ 作为新物理量")
print("=" * 90)

# ---- Part 1: 相关性 ----
print("\n[1] 相关性分析")
print("-" * 50)

corr_pc_phi = np.corrcoef(pc, phi)[0, 1]
corr_pc_z = np.corrcoef(pc, z)[0, 1]
corr_pc_zd = np.corrcoef(pc, z*d)[0, 1]

std_mask = df['type'] == 'standard'
corr_z_phi = np.corrcoef(z[std_mask], phi[std_mask])[0, 1]

# 偏相关
coef_pz = np.polyfit(np.log(z), np.log(pc), 1)
resid_pz = np.log(pc) - np.polyval(coef_pz, np.log(z))
coef_phiz = np.polyfit(np.log(z), np.log(phi), 1)
resid_phiz = np.log(phi) - np.polyval(coef_phiz, np.log(z))
partial_corr = np.corrcoef(resid_pz, resid_phiz)[0, 1]

print(f"  corr(p_c, φ)   = {corr_pc_phi:.4f}")
print(f"  corr(p_c, z)   = {corr_pc_z:.4f}")
print(f"  corr(p_c, z·d) = {corr_pc_zd:.4f}")
print(f"  corr(z, φ) std = {corr_z_phi:.4f}")
print(f"  partial_corr(p_c, φ | z) = {partial_corr:.4f}")

print(f"\n  标准晶格:")
for i in range(len(names)):
    if std_mask.iloc[i]:
        print(f"    {names[i]:<12} φ={phi[i]:.2f}  z={int(z[i]):>3}  d={int(d[i])}  p_c={pc[i]:.5f}")

# ---- Part 2: 公式拟合 ----
print("\n[2] 公式拟合比较")
print("-" * 80)

formulas = [
    ('F0: baseline (no φ)',       F0, [2.4, 0.85, 0.75, 1.5]),
    ('F1: ×φ^β',                  F1, [2.4, 0.85, 0.75, 1.5, 0.3]),
    ('F2: ×(1+C(φ-0.52))',        F2, [2.4, 0.85, 0.75, 1.5, -0.5]),
    ('F3: ×exp(λφ)',              F3, [2.4, 0.85, 0.75, 1.5, -0.5]),
    ('F4: (zd)^{-α}(1+B/g)φ^β',  F4, [2.0, 0.85, 1.5, 0.3]),
    ('F5: ×(1+δz(φ-0.52))',      F5, [2.4, 0.85, 0.75, 1.5, -0.1]),
    ('F6: ×((1-φ)/φ)^δ',         F6, [1.0, 0.85, 0.75, 1.5, 0.5]),
]

results = {}
print(f"  {'Formula':<40} {'R²':>8} {'RMSE':>8} {'MaxErr':>7} {'FCC_err':>8} {'AIC':>8}")
print("  " + "-" * 76)

for fname, func, p0 in formulas:
    r = fit(func, p0, z, d, g, phi, pc)
    fcc_err = abs(pc[fcc_idx] - r['pred'][fcc_idx]) / pc[fcc_idx]
    r['fcc_err'] = fcc_err
    r['fname'] = fname
    results[fname] = r
    print(f"  {fname:<40} {r['R2']:>8.6f} {r['RMSE']:>8.6f} {r['max_err_rel']*100:>6.1f}% "
          f"{fcc_err*100:>7.1f}% {r['aic']:>8.1f}")

best_name = min(results, key=lambda k: results[k]['aic'])
best = results[best_name]
base = results['F0: baseline (no φ)']

print(f"\n  Best (AIC): {best_name}")
print(f"  Params: {best['params']}")
print(f"  FCC err: {best['fcc_err']:.1%}")
print(f"  R²: {best['R2']:.6f}")

# ---- Part 3: 逐点分析 ----
print(f"\n[3] 逐点分析")
print("-" * 90)
print(f"  {'Lattice':<18} {'p_c':>7} {'F0_pred':>7} {'F0_err':>7} {'Best_pred':>9} {'Best_err':>8} {'Δ':>7}")
print("  " + "-" * 72)

for i in range(n):
    f0e = abs(pc[i]-base['pred'][i])/pc[i]*100
    be_e = abs(pc[i]-best['pred'][i])/pc[i]*100
    imp = f0e - be_e
    mk = " <--FCC" if i == fcc_idx else ""
    print(f"  {names[i]:<18} {pc[i]:>7.4f} {base['pred'][i]:>7.4f} {f0e:>6.1f}% "
          f"{best['pred'][i]:>9.5f} {be_e:>7.1f}% {imp:>+6.1f}%{mk}")

mask_nf = np.ones(n, dtype=bool)
mask_nf[fcc_idx] = False
print(f"\n  MAE (excl FCC): F0={np.mean(np.abs(pc[mask_nf]-base['pred'][mask_nf])):.5f}"
      f"  Best={np.mean(np.abs(pc[mask_nf]-best['pred'][mask_nf])):.5f}")

# ---- Part 4: 排除FCC ----
print(f"\n[4] 排除 FCC 后的 φ 贡献")
print("-" * 50)
r_b_nf = fit(F0, [2.4, 0.85, 0.75, 1.5], z[mask_nf], d[mask_nf], g[mask_nf], phi[mask_nf], pc[mask_nf])
r_p_nf = fit(F1, [2.4, 0.85, 0.75, 1.5, 0.3], z[mask_nf], d[mask_nf], g[mask_nf], phi[mask_nf], pc[mask_nf])
print(f"  F0 (无φ): R²={r_b_nf['R2']:.6f}, RMSE={r_b_nf['RMSE']:.6f}")
print(f"  F1 (含φ): R²={r_p_nf['R2']:.6f}, RMSE={r_p_nf['RMSE']:.6f}")
print(f"  ΔR² = {r_p_nf['R2']-r_b_nf['R2']:+.6f}")

# ---- Part 5: β 扫描 ----
print(f"\n[5] β 参数扫描")
print("-" * 50)

beta_vals = np.linspace(-2.0, 2.0, 60)
r2_arr = []
fcc_err_arr = []

for b in beta_vals:
    def F1_fixed(p, z, d, g, phi, _b=b):
        return p[0] * z**(-p[1]) * d**(-p[2]) * (1 + p[3]/g) * phi**_b
    r_t = fit(F1_fixed, [2.4, 0.85, 0.75, 1.5], z, d, g, phi, pc)
    r2_arr.append(r_t['R2'])
    fcc_err_arr.append(abs(pc[fcc_idx]-r_t['pred'][fcc_idx])/pc[fcc_idx]*100)

r2_arr = np.array(r2_arr)
fcc_err_arr = np.array(fcc_err_arr)
best_b_idx = np.argmax(r2_arr)
best_beta = beta_vals[best_b_idx]

print(f"  β 范围: [{beta_vals[0]:.1f}, {beta_vals[-1]:.1f}]")
print(f"  最优 β = {best_beta:.3f}  →  R²={r2_arr[best_b_idx]:.6f}")
print(f"  FCC err at β*: {fcc_err_arr[best_b_idx]:.1f}%")

# ---- Part 6: 密堆积对比 ----
print(f"\n[6] 密堆积晶格对比")
print("-" * 50)
cp = [('FCC','closest-packed',0.74,12,3,0.1188),
      ('HCP','closest-packed',0.74,12,3,0.1188),
      ('triangular','2D-hex',0.91,6,2,0.5000),
      ('BCC','non-close',0.68,8,3,0.2458),
      ('SC','non-close',0.52,6,3,0.3116),
      ('square','non-close',0.52,4,2,0.5927),
      ('diamond','open',0.34,4,3,0.4319),
      ('honeycomb','open',0.30,3,2,0.6970),
      ('kagome','open',0.46,4,2,0.6527)]
print(f"  {'Lattice':<12} {'Type':<15} {'φ':>5} {'z':>3} {'d':>2} {'p_c':>8}")
for r in cp:
    print(f"  {r[0]:<12} {r[1]:<15} {r[2]:>5.2f} {r[3]:>3} {r[4]:>2} {r[5]:>8.4f}")

# ============================================================
# 可视化
# ============================================================
print(f"\n[7] 生成图表")
print("-" * 50)

fig = plt.figure(figsize=(20, 22))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
tc = {'standard':'#2196F3','highD':'#FF9800','MC':'#4CAF50'}

# (a) p_c vs φ
ax = fig.add_subplot(gs[0,0])
for t,c in tc.items():
    m = df['type']==t
    ax.scatter(df.loc[m,'phi'], df.loc[m,'pc'], c=c, label=t, s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
fm = df['name']=='fcc'
ax.scatter(df.loc[fm,'phi'], df.loc[fm,'pc'], c='red', label='FCC', s=150, marker='*', zorder=5, edgecolors='k')
xp = np.linspace(0.25, 0.95, 100)
ax.plot(xp, np.polyval(np.polyfit(phi, pc, 2), xp), 'k--', alpha=0.3)
ax.set_xlabel('Packing fraction φ'); ax.set_ylabel('p_c')
ax.set_title('(a) p_c vs φ', fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (b) p_c vs z·d colored by φ
ax = fig.add_subplot(gs[0,1])
sc = ax.scatter(df['zd'], df['pc'], c=df['phi'], cmap='viridis', s=60, edgecolors='k', linewidth=0.5)
plt.colorbar(sc, ax=ax, label='φ')
ax.annotate('FCC', (df.loc[fm,'zd'].values[0], df.loc[fm,'pc'].values[0]),
           textcoords="offset points", xytext=(10,5), fontsize=11, fontweight='bold', color='red')
ax.set_xlabel('z·d'); ax.set_ylabel('p_c')
ax.set_title('(b) p_c vs z·d (color=φ)', fontweight='bold')
ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, alpha=0.3, which='both')

# (c) Residuals before/after
ax = fig.add_subplot(gs[0,2])
rf0 = (pc - base['pred'])/pc*100
rbe = (pc - best['pred'])/pc*100
xi = np.arange(n); w = 0.35
ax.bar(xi-w/2, rf0, w, label='F0 (no φ)', color='#90CAF9', alpha=0.8)
ax.bar(xi+w/2, rbe, w, label='Best (with φ)', color='#EF9A9A', alpha=0.8)
ax.bar(xi[fcc_idx]-w/2, rf0[fcc_idx], w, color='blue', alpha=0.9)
ax.bar(xi[fcc_idx]+w/2, rbe[fcc_idx], w, color='red', alpha=0.9)
ax.set_xticks(xi)
ax.set_xticklabels([nm[:10] for nm in names], rotation=45, ha='right', fontsize=6.5)
ax.set_ylabel('Residual (%)'); ax.set_title('(c) Residuals: before vs after φ', fontweight='bold')
ax.axhline(0, color='k', lw=0.5); ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis='y')

# (d) Close-packed comparison
ax = fig.add_subplot(gs[1,0])
cphi = [r[2] for r in cp]; cpc = [r[5] for r in cp]
ccol = ['#F44336' if 'closest' in r[1] else ('#FF9800' if r[1]=='non-close' else '#66BB6A') for r in cp]
ax.scatter(cphi, cpc, c=ccol, s=100, edgecolors='k', linewidth=0.8, zorder=3)
for i,r in enumerate(cp):
    ax.annotate(r[0], (cphi[i],cpc[i]), textcoords="offset points", xytext=(5,5), fontsize=8)
ax.set_xlabel('φ'); ax.set_ylabel('p_c')
ax.set_title('(d) Close-packed lattice comparison', fontweight='bold'); ax.grid(True, alpha=0.3)

# (e) Model comparison bar chart
ax = fig.add_subplot(gs[1,1])
mns = [k.split(':')[0] for k in results]
r2v = [results[k]['R2'] for k in results]
fccv = [results[k]['fcc_err']*100 for k in results]
xm = np.arange(len(mns))
ax.bar(xm-0.2, r2v, 0.4, label='R²', color='#64B5F6', alpha=0.8)
axt = ax.twinx()
axt.bar(xm+0.2, fccv, 0.4, label='FCC err%', color='#EF5350', alpha=0.8)
ax.set_xticks(xm); ax.set_xticklabels(mns, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('R²', color='#1565C0'); axt.set_ylabel('FCC error (%)', color='#C62828')
ax.set_title('(e) Model comparison', fontweight='bold')
ax.set_ylim([0.95,1.0]); ax.legend(loc='lower right', fontsize=8); axt.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# (f) Residual distribution
ax = fig.add_subplot(gs[1,2])
ax.hist(rf0[mask_nf], bins=10, alpha=0.6, label='F0 excl FCC', color='#90CAF9', edgecolor='k', linewidth=0.5)
ax.hist(rbe[mask_nf], bins=10, alpha=0.6, label='Best excl FCC', color='#EF9A9A', edgecolor='k', linewidth=0.5)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('Residual (%)'); ax.set_ylabel('Count')
ax.set_title('(f) Residual distribution (excl. FCC)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# (g) Phase space (z, φ)
ax = fig.add_subplot(gs[2,0])
for t,c in tc.items():
    m = df['type']==t
    ax.scatter(df.loc[m,'z'], df.loc[m,'phi'], c=c, label=t, s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
ax.scatter(df.loc[fm,'z'], df.loc[fm,'phi'], c='red', label='FCC', s=200, marker='*', zorder=5, edgecolors='k', linewidth=1.5)
zr = np.linspace(3, 20, 40); pr = np.linspace(0.25, 0.95, 40)
Z, P = np.meshgrid(zr, pr)
A_b, al_b, ga_b, B_b = base['params']
PC_g = A_b * Z**(-al_b) * 3**(-ga_b) * (1 + B_b/4)
cs = ax.contour(Z, P, PC_g, levels=8, cmap='coolwarm', alpha=0.5)
ax.clabel(cs, inline=True, fontsize=7, fmt='%.3f')
ax.set_xlabel('z'); ax.set_ylabel('φ')
ax.set_title('(g) Phase space (z, φ) with p_c contours', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (h) Beta sensitivity
ax = fig.add_subplot(gs[2,1])
ax.plot(beta_vals, r2_arr, 'b-', lw=2, label='R²')
axt = ax.twinx()
axt.plot(beta_vals, fcc_err_arr, 'r-', lw=2, label='FCC err%')
ax.axvline(best_beta, color='green', ls='--', alpha=0.6, label=f'β*={best_beta:.2f}')
ax.set_xlabel('β (φ exponent)'); ax.set_ylabel('R²', color='blue'); axt.set_ylabel('FCC error (%)', color='red')
ax.set_title('(h) Sensitivity to φ exponent β', fontweight='bold')
ax.legend(loc='lower right', fontsize=9); axt.legend(loc='upper right', fontsize=9); ax.grid(True, alpha=0.2)

# (i) Per-lattice improvement
ax = fig.add_subplot(gs[2,2])
imp = np.abs(rf0) - np.abs(rbe)
ax.barh(range(n), imp, color='#81C784', alpha=0.8, edgecolor='k', linewidth=0.3)
ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('|Residual| reduction (%)')
ax.set_title('(i) Per-lattice improvement from φ', fontweight='bold')
ax.axvline(0, color='k', lw=0.8); ax.grid(True, alpha=0.2, axis='x')
ax.barh(fcc_idx, imp[fcc_idx], color='red', alpha=0.9)
ax.annotate(f'FCC: {imp[fcc_idx]:+.1f}%', xy=(imp[fcc_idx], fcc_idx),
           xytext=(5,0), textcoords='offset points', fontsize=9, fontweight='bold', color='red', va='center')

plt.savefig(f'{OUT}fcc_packing_figures.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  图表已保存: {OUT}fcc_packing_figures.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 90)
print("  核心结论")
print("=" * 90)

dr2 = best['R2'] - base['R2']
dfcc = base['fcc_err'] - best['fcc_err']

print(f"""
  1. corr(p_c, φ) = {corr_pc_phi:.4f}
  2. corr(z, φ) = {corr_z_phi:.4f} (标准晶格, z-φ 高度共线)
  3. partial_corr(p_c, φ | z) = {partial_corr:.4f}
  4. 基准公式 FCC 误差: {base['fcc_err']:.1%}
  5. 最优含φ公式 FCC 误差: {best['fcc_err']:.1%}
  6. 最优公式: {best_name}
  7. ΔR² = {dr2:+.6f}
  8. β 最优值: {best_beta:.3f}
  9. 排除FCC后 ΔR² = {r_p_nf['R2']-r_b_nf['R2']:+.6f}

  物理分析:
  - φ 与 z 高度共线 (corr={corr_z_phi:.2f}), 独立贡献受限
  - FCC 同时达到 z_max=12 和 φ_max=0.74, 双重极端值
  - 排除FCC后, φ 的额外解释力 {'微弱' if abs(r_p_nf['R2']-r_b_nf['R2']) < 0.001 else '有一定贡献'}
  - φ 更适合作为 FCC 特殊性的诊断量而非独立预测变量
""")

# Save summary
with open(f'{OUT}fcc_packing_summary.txt', 'w') as f:
    f.write("FCC Packing Fraction Analysis Summary\n")
    f.write("="*60 + "\n\n")
    f.write(f"corr(p_c, phi) = {corr_pc_phi:.4f}\n")
    f.write(f"corr(z, phi) = {corr_z_phi:.4f}\n")
    f.write(f"partial_corr(p_c, phi | z) = {partial_corr:.4f}\n\n")
    f.write(f"Baseline: R²={base['R2']:.6f}, FCC_err={base['fcc_err']:.1%}\n")
    f.write(f"Best phi model: {best_name}\n")
    f.write(f"  R²={best['R2']:.6f}, FCC_err={best['fcc_err']:.1%}\n")
    f.write(f"  params={best['params']}\n\n")
    f.write("All models:\n")
    for k,v in results.items():
        f.write(f"  {k:<40} R²={v['R2']:.6f} FCC_err={v['fcc_err']:.1%} AIC={v['aic']:.1f}\n")
    f.write(f"\nOptimal beta = {best_beta:.3f}\n")
    f.write(f"Excl FCC: F0 R²={r_b_nf['R2']:.6f}, F1 R²={r_p_nf['R2']:.6f}\n")
print(f"  摘要已保存: {OUT}fcc_packing_summary.txt")
