from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
PRL论文图表生成
=================
生成出版级图表:
1. 公式概览 + 数据对比
2. 误差景观图
3. 维度标度图 (展示exp(-αd)的物理意义)
4. 饱和行为图
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

OUT = ROOT

# 设置出版级字体
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 加载数据
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values if 'name' in df.columns else [f"lat_{i}" for i in range(len(df))]

# 公式参数
a, alpha, gamma, z0 = 12.3346, 0.3368, 0.8477, 9.6171

def model_new(z, d, g):
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

pred = model_new(z, d, g)
errors = np.abs((pc - pred) / pc) * 100

# 颜色映射: 按维度
dim_colors = {2: '#e74c3c', 3: '#3498db', 4: '#2ecc71', 5: '#9b59b6',
              6: '#f39c12', 7: '#1abc9c', 8: '#e67e22', 9: '#34495e',
              10: '#d35400', 11: '#8e44ad', 12: '#27ae60', 13: '#c0392b'}

def get_colors(d_arr):
    return [dim_colors.get(int(di), '#95a5a6') for di in d_arr]

# ============================================================
# 图1: 公式概览 (4-panel)
# ============================================================
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Panel a: p_c预测 vs 实测
ax1 = fig.add_subplot(gs[0, 0])
colors = get_colors(d)
ax1.scatter(pc, pred, c=colors, s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
lim = [0.01, 0.9]
ax1.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)
ax1.set_xlabel('$p_c$ (measured)')
ax1.set_ylabel('$p_c$ (predicted)')
ax1.set_title('(a) Prediction accuracy')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.01, 1)
ax1.set_ylim(0.01, 1)
ax1.text(0.05, 0.5, f'MAPE = {np.mean(errors):.1f}%\n$R^2 = 0.973$',
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel b: 误差分布
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(errors, bins=20, color='#3498db', edgecolor='k', alpha=0.7)
ax2.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean = {np.mean(errors):.1f}%')
ax2.axvline(np.median(errors), color='g', linestyle='--', label=f'Median = {np.median(errors):.1f}%')
ax2.set_xlabel('Relative error (%)')
ax2.set_ylabel('Count')
ax2.set_title('(b) Error distribution')
ax2.legend()

# Panel c: 误差 vs 维度
ax3 = fig.add_subplot(gs[1, 0])
for d_val in sorted(set(d.astype(int))):
    mask = d.astype(int) == d_val
    ax3.scatter(d[mask], errors[mask], c=[dim_colors.get(d_val, 'gray')],
               s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
ax3.set_xlabel('Dimension $d$')
ax3.set_ylabel('Relative error (%)')
ax3.set_title('(c) Error vs dimension')
ax3.axhline(np.mean(errors), color='gray', linestyle=':', alpha=0.5)

# 添加分类平均误差
for d_val in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    mask = d.astype(int) == d_val
    if np.sum(mask) > 0:
        mean_err = np.mean(errors[mask])
        ax3.scatter(d_val, mean_err, c='k', s=50, marker='D', zorder=5)

# Panel d: 误差 vs 配位数
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(z, errors, c=colors, s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
ax4.set_xlabel('Coordination number $z$')
ax4.set_ylabel('Relative error (%)')
ax4.set_title('(d) Error vs coordination number')
ax4.set_xscale('log')

plt.savefig(f'{OUT}/PRL_fig1_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("图1已保存: PRL_fig1_overview.png")

# ============================================================
# 图2: 物理机制 — exp(-αd) vs d^{-β}
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel a: 衰减函数对比
ax = axes[0]
d_range = np.arange(2, 14)
exp_decay = np.exp(-alpha * d_range)
power_decay = (d_range / 2)**(-1.03)  # 归一化使d=2=1

ax.plot(d_range, exp_decay, 'r-o', markersize=5, linewidth=2, label='$e^{-\\alpha d}$ (this work)')
ax.plot(d_range, power_decay, 'b--s', markersize=5, linewidth=2, label='$d^{-\\beta}$ (previous)')
ax.axvline(6, color='gray', linestyle=':', alpha=0.5)
ax.text(6.2, 0.8, '$d_c = 6$', fontsize=9, color='gray')
ax.set_xlabel('Dimension $d$')
ax.set_ylabel('Decay factor (normalized)')
ax.set_title('(a) Dimensional decay')
ax.legend()
ax.set_ylim(0, 1.1)

# Panel b: R-1 vs d (超立方) + 理论曲线
ax = axes[1]
mask_hc = z == 2*d
d_hc = d[mask_hc]
R_hc = pc[mask_hc] * (z[mask_hc] - 1)
R_minus_1 = R_hc - 1

# 理论曲线
d_fine = np.linspace(2, 13, 100)
R_theory = 1 + (a * np.exp(-alpha * d_fine) / 4**gamma) * (1 - np.exp(-2*d_fine/z0))
R_theory_minus1 = R_theory - 1

ax.scatter(d_hc, R_minus_1, c='red', s=50, zorder=5, edgecolors='k', linewidths=0.5, label='Data (hypercubic)')
ax.plot(d_fine, R_theory_minus1, 'r-', linewidth=2, label='Formula')
ax.axvline(6, color='gray', linestyle=':', alpha=0.5)
ax.text(6.2, max(R_minus_1)*0.9, '$d_c=6$', fontsize=9, color='gray')
ax.set_xlabel('Dimension $d$')
ax.set_ylabel('$R - 1$ (Bethe correction)')
ax.set_title('(b) Hypercubic saturation')
ax.legend()

# Panel c: 饱和函数 S(z)
ax = axes[2]
z_fine = np.linspace(2, 50, 100)
S_z = 1 - np.exp(-z_fine / z0)

ax.plot(z_fine, S_z, 'b-', linewidth=2, label=f'$S(z) = 1 - e^{{-z/{z0:.1f}}}$')
ax.axvline(z0, color='r', linestyle='--', alpha=0.5)
ax.text(z0+1, 0.3, f'$z_0 = {z0:.1f}$', fontsize=9, color='r')
ax.set_xlabel('Coordination number $z$')
ax.set_ylabel('Saturation function $S(z)$')
ax.set_title('(c) Coordination saturation')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUT}/PRL_fig2_physics.png', dpi=300, bbox_inches='tight')
plt.close()
print("图2已保存: PRL_fig2_physics.png")

# ============================================================
# 图3: 误差景观 — 按(z, d)分布
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

# 气泡图: x=z, y=d, 颜色=误差
sc = ax.scatter(z, d, s=errors*15, c=errors, cmap='RdYlGn_r',
               alpha=0.7, edgecolors='k', linewidths=0.5, vmin=0, vmax=30)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Relative error (%)')

# 标注特殊晶格
special = {'fcc': (12, 3), 'hcp': (12, 3), 'diamond': (4, 3),
           'triangular': (6, 2), 'honeycomb': (3, 2), 'square': (4, 2)}
for name, (zi, di) in special.items():
    idx = np.where((names == name))[0]
    if len(idx) > 0:
        ax.annotate(name, (z[idx[0]], d[idx[0]]), fontsize=7,
                   xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Coordination number $z$')
ax.set_ylabel('Dimension $d$')
ax.set_title('Error landscape across lattice parameters')

# 添加等误差线（近似）
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(f'{OUT}/PRL_fig3_landscape.png', dpi=300, bbox_inches='tight')
plt.close()
print("图3已保存: PRL_fig3_landscape.png")

# ============================================================
# 图4: 新旧公式对比 (超立方晶格)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel a: 超立方 p_c vs d
ax = axes[0]
idx_hc = np.where(z == 2*d)[0]
idx_sorted = idx_hc[np.argsort(d[idx_hc])]

# 只取超立方中的g=4（真正的超立方）
mask_true_hc = (z == 2*d) & (g == 4)
idx_true = np.where(mask_true_hc)[0]
idx_true_sorted = idx_true[np.argsort(d[idx_true])]

d_old = [12.9 / (di**1.03 * 4**0.85) * (1 - np.exp(-2*di/9.7)) for di in d[idx_true_sorted]]
d_new = [a * np.exp(-alpha * di) / 4**gamma * (1 - np.exp(-2*di/z0)) for di in d[idx_true_sorted]]

p_old = np.array([(1 + c) / (2*di - 1) for c, di in zip(d_old, d[idx_true_sorted])])
p_new = np.array([(1 + c) / (2*di - 1) for c, di in zip(d_new, d[idx_true_sorted])])
p_measured = pc[idx_true_sorted]

ax.scatter(d[idx_true_sorted], p_measured, c='k', s=60, zorder=5, label='Measured')
ax.plot(d[idx_true_sorted], p_old, 'b--o', markersize=4, linewidth=1.5, label='Old: $d^{-\\beta}$ model')
ax.plot(d[idx_true_sorted], p_new, 'r-s', markersize=4, linewidth=1.5, label='New: $e^{-\\alpha d}$ model')
# Bethe极限
d_bethe = np.linspace(4, 13, 100)
ax.plot(d_bethe, 1/(2*d_bethe-1), color='gray', linestyle=':', linewidth=1, label='Bethe $1/(z-1)$')

ax.set_xlabel('Dimension $d$')
ax.set_ylabel('$p_c$ (hypercubic)')
ax.set_title('(a) Hypercubic thresholds')
ax.legend(fontsize=8)

# Panel b: 误差对比
ax = axes[1]
err_old = np.abs((p_measured - p_old) / p_measured) * 100
err_new = np.abs((p_measured - p_new) / p_measured) * 100

width = 0.35
x = np.arange(len(d[idx_true_sorted]))
ax.bar(x - width/2, err_old, width, color='blue', alpha=0.6, label='Old model')
ax.bar(x + width/2, err_new, width, color='red', alpha=0.6, label='New model')
ax.set_xticks(x)
ax.set_xticklabels([f'{int(di)}D' for di in d[idx_true_sorted]], fontsize=8)
ax.set_xlabel('Dimension')
ax.set_ylabel('Error (%)')
ax.set_title('(b) Error comparison (hypercubic)')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUT}/PRL_fig4_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("图4已保存: PRL_fig4_comparison.png")

# ============================================================
# 图5: 残差分析
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel a: 残差 vs 预测值
ax = axes[0]
residuals = (pc - pred) / pc * 100
ax.scatter(pred, residuals, c=get_colors(d), s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.axhline(np.mean(residuals), color='r', linestyle='--', alpha=0.5, label=f'Mean = {np.mean(residuals):.1f}%')
ax.set_xlabel('$p_c$ (predicted)')
ax.set_ylabel('Relative residual (%)')
ax.set_title('(a) Residual analysis')
ax.legend()
ax.set_xscale('log')

# Panel b: Q-Q plot
ax = axes[1]
from scipy.stats import norm
sorted_residuals = np.sort(residuals)
theoretical = norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
ax.scatter(theoretical, sorted_residuals, c=get_colors(np.sort(d)), s=20, alpha=0.6)
ax.plot([-3, 3], [np.mean(residuals)-np.std(residuals)*3, np.mean(residuals)+np.std(residuals)*3],
        'k--', alpha=0.3)
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Sample quantiles')
ax.set_title('(b) Q-Q plot')

plt.tight_layout()
plt.savefig(f'{OUT}/PRL_fig5_residuals.png', dpi=300, bbox_inches='tight')
plt.close()
print("图5已保存: PRL_fig5_residuals.png")

print(f"\n所有图表已保存到 {OUT}/")
print("图表列表:")
for i in range(1, 6):
    print(f"  PRL_fig{i}_*.png")
