from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
公式发现 v2 — 可视化
=====================
  生成多个图表展示公式的预测效果
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

OUT = ROOT

# 加载数据
df = pd.read_csv(f'{OUT}/complete_dataset_v2.csv')
df = df[df['d'] > 0].copy().reset_index(drop=True)
n = len(df)
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values
names = df['name'].values

# 最佳公式
a_opt = 1.716
alpha, beta, gamma = 0.6, 1.5, 0.5

def pred(z, d, g):
    return 1/(z-1) * (1 + a_opt * np.power(z, alpha) / (np.power(d, beta) * np.power(g, gamma)))

pred_vals = pred(z, d, g)
errors = np.abs(pc - pred_vals) / pc * 100

# ============================================================
# 图 1: 4-panel 概览
# ============================================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Panel 1: 预测 vs 实际
ax1 = fig.add_subplot(gs[0, 0])
colors = {2: '#2196F3', 3: '#FF5722'}
for d_val in [2, 3]:
    mask = d == d_val
    ax1.scatter(pc[mask], pred_vals[mask], c=[colors[d_val]], label=f'd={int(d_val)}',
               s=30, alpha=0.6, edgecolors='white', linewidth=0.5)
mask_hd = d >= 4
ax1.scatter(pc[mask_hd], pred_vals[mask_hd], c='#4CAF50', label=f'd≥4',
           s=30, alpha=0.6, edgecolors='white', linewidth=0.5)
lims = [0, 0.75]
ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax1.set_xlabel('Actual $p_c$', fontsize=11)
ax1.set_ylabel('Predicted $p_c$', fontsize=11)
ax1.set_title('Predicted vs Actual $p_c$', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_xlim(lims)
ax1.set_ylim(lims)

# Panel 2: 误差分布
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(errors, bins=20, color='#2196F3', alpha=0.7, edgecolor='white')
ax2.axvline(np.median(errors), color='red', linestyle='--', label=f'Median={np.median(errors):.1f}%')
ax2.axvline(np.mean(errors), color='orange', linestyle='--', label=f'Mean={np.mean(errors):.1f}%')
ax2.set_xlabel('Relative Error (%)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

# Panel 3: R-1 vs 公式预测的 R-1
ax3 = fig.add_subplot(gs[1, 0])
R = pc * (z - 1)
R_pred = pred_vals * (z - 1)
Rm1 = R - 1
Rm1_pred = R_pred - 1

for d_val in [2, 3]:
    mask = d == d_val
    ax3.scatter(Rm1[mask], Rm1_pred[mask], c=[colors[d_val]], label=f'd={int(d_val)}',
               s=30, alpha=0.6, edgecolors='white', linewidth=0.5)
mask_hd = d >= 4
ax3.scatter(Rm1[mask_hd], Rm1_pred[mask_hd], c='#4CAF50', label=f'd≥4',
           s=30, alpha=0.6, edgecolors='white', linewidth=0.5)
max_R = max(Rm1.max(), Rm1_pred.max()) * 1.1
ax3.plot([0, max_R], [0, max_R], 'k--', alpha=0.5, linewidth=1)
ax3.set_xlabel('R - 1 (Actual)', fontsize=11)
ax3.set_ylabel('R - 1 (Predicted)', fontsize=11)
ax3.set_title('Bethe Deviation: Actual vs Predicted', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_xscale('log')
ax3.set_yscale('log')

# Panel 4: 按维度的误差
ax4 = fig.add_subplot(gs[1, 1])
d_unique = sorted(np.unique(d.astype(int)))
mean_errs = [np.mean(errors[d == dv]) for dv in d_unique]
max_errs = [np.max(errors[d == dv]) for dv in d_unique]
n_per_d = [np.sum(d == dv) for dv in d_unique]

bar_colors = ['#2196F3' if dv <= 3 else '#4CAF50' for dv in d_unique]
x = np.arange(len(d_unique))
width = 0.35
bars1 = ax4.bar(x - width/2, mean_errs, width, label='Mean Error', color=bar_colors, alpha=0.7)
bars2 = ax4.bar(x + width/2, max_errs, width, label='Max Error', color=bar_colors, alpha=0.4)
ax4.set_xlabel('Dimension d', fontsize=11)
ax4.set_ylabel('Error (%)', fontsize=11)
ax4.set_title('Error by Dimension', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'd={dv}\n(n={n_per_d[i]})' for i, dv in enumerate(d_unique)])
ax4.legend(fontsize=9)

plt.savefig(f'{OUT}/formula_v2_overview.png', dpi=150, bbox_inches='tight')
print("Saved: formula_v2_overview.png")

# ============================================================
# 图 2: 公式结构可视化
# ============================================================
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

# z dependence
z_range = np.linspace(3, 28, 100)
for d_val, color, ls in [(2, '#2196F3', '-'), (3, '#FF5722', '--'), (4, '#4CAF50', ':'), (6, '#9C27B0', '-.')]:
    pc_pred = pred(z_range, np.full_like(z_range, d_val), np.full_like(z_range, 4.0))
    axes[0].plot(z_range, pc_pred, color=color, linestyle=ls, label=f'd={d_val}')
axes[0].set_xlabel('z (coordination number)', fontsize=11)
axes[0].set_ylabel('$p_c$ (predicted)', fontsize=11)
axes[0].set_title('Predicted $p_c$ vs z', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)

# d dependence
d_range = np.linspace(2, 10, 100)
for z_val, color, ls in [(6, '#2196F3', '-'), (8, '#FF5722', '--'), (12, '#4CAF50', ':'), (18, '#9C27B0', '-.')]:
    pc_pred = pred(np.full_like(d_range, z_val), d_range, np.full_like(d_range, 4.0))
    axes[1].plot(d_range, pc_pred, color=color, linestyle=ls, label=f'z={z_val}')
axes[1].set_xlabel('d (dimension)', fontsize=11)
axes[1].set_ylabel('$p_c$ (predicted)', fontsize=11)
axes[1].set_title('Predicted $p_c$ vs d', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)

# g dependence
g_range = np.linspace(3, 8, 100)
for z_val, d_val, color, ls in [(4, 2, '#2196F3', '-'), (6, 3, '#FF5722', '--'), (8, 4, '#4CAF50', ':')]:
    pc_pred = pred(np.full_like(g_range, z_val), np.full_like(g_range, d_val), g_range)
    axes[2].plot(g_range, pc_pred, color=color, linestyle=ls, label=f'z={z_val},d={d_val}')
axes[2].set_xlabel('g (girth)', fontsize=11)
axes[2].set_ylabel('$p_c$ (predicted)', fontsize=11)
axes[2].set_title('Predicted $p_c$ vs g', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT}/formula_v2_structure.png', dpi=150, bbox_inches='tight')
print("Saved: formula_v2_structure.png")

# ============================================================
# 图 3: 残差热力图
# ============================================================
fig3, ax = plt.subplots(1, 1, figsize=(12, 8))

# 按 (z, d) 排列
mask_2d = d == 2
mask_3d = d == 3

# 2D部分
z_2d = z[mask_2d]
errors_2d = errors[mask_2d]
names_2d = names[mask_2d]
g_2d = g[mask_2d]

order = np.argsort(z_2d)
scatter = ax.scatter(z_2d[order], np.full_like(z_2d[order], 2.0),
                     c=errors_2d[order], cmap='RdYlGn_r', vmin=0, vmax=30,
                     s=100, edgecolors='gray', linewidth=0.5)

# 3D部分
z_3d = z[mask_3d]
errors_3d = errors[mask_3d]
names_3d = names[mask_3d]
ax.scatter(z_3d, np.full_like(z_3d, 3.0), c=errors_3d, cmap='RdYlGn_r', vmin=0, vmax=30,
          s=100, edgecolors='gray', linewidth=0.5)

# d≥4
mask_hd = d >= 4
z_hd = z[mask_hd]
errors_hd = errors[mask_hd]
names_hd = names[mask_hd]
for d_val in np.unique(d[mask_hd]):
    mask_d = (d == d_val) & mask_hd
    ax.scatter(z[mask_d], np.full(np.sum(mask_d), d_val), c=errors[mask_d],
              cmap='RdYlGn_r', vmin=0, vmax=30, s=100, edgecolors='gray', linewidth=0.5)

# 标注名字
for i in range(len(z_2d)):
    idx = order[i]
    if errors_2d[idx] > 10 or z_2d[idx] in [3, 4, 6, 28]:
        ax.annotate(names_2d[idx], (z_2d[idx], 2.0), fontsize=6, alpha=0.7,
                   textcoords='offset points', xytext=(5, 5))

for i in range(len(z_3d)):
    ax.annotate(names_3d[i], (z_3d[i], 3.0), fontsize=7, fontweight='bold', alpha=0.8,
               textcoords='offset points', xytext=(5, 5))

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Relative Error (%)', fontsize=11)
ax.set_xlabel('z (coordination number)', fontsize=11)
ax.set_ylabel('d (dimension)', fontsize=11)
ax.set_title('Prediction Error Landscape\n(Bubble color = error, green=good, red=bad)',
            fontsize=12, fontweight='bold')
ax.set_yticks([2, 3, 4, 5, 6, 7, 8, 9])

plt.savefig(f'{OUT}/formula_v2_error_landscape.png', dpi=150, bbox_inches='tight')
print("Saved: formula_v2_error_landscape.png")

print("\nAll visualizations saved.")
