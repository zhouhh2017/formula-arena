from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
Generate Statistical Comparison Figure
Shows exponential vs power-law model comparison.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = ROOT

# Load data
df = pd.read_csv(f'{OUT}/complete_dataset_v4.csv')
z = df['z'].values.astype(float)
d = df['d'].values.astype(float)
g = df['g'].values.astype(float)
pc = df['pc'].values

# Models
a, alpha, gamma, z0 = 12.3346, 0.3368, 0.8477, 9.6171

def model_exp(z, d, g):
    correction = (a * np.exp(-alpha * d) / g**gamma) * (1 - np.exp(-z / z0))
    return (1 + correction) / (z - 1)

def model_power(z, d, g):
    a_p, beta_p, gamma_p, z0_p = 12.9, 1.03, 0.85, 9.7
    correction = (a_p / (d**beta_p * g**gamma_p)) * (1 - np.exp(-z / z0_p))
    return (1 + correction) / (z - 1)

pred_exp = model_exp(z, d, g)
pred_power = model_power(z, d, g)
errors_exp = np.abs((pc - pred_exp) / pc) * 100
errors_power = np.abs((pc - pred_power) / pc) * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel a: Error distribution comparison
ax = axes[0, 0]
bins = np.linspace(0, 30, 31)
ax.hist(errors_exp, bins=bins, alpha=0.6, label=f'Exponential (MAPE={np.mean(errors_exp):.2f}%)',
        color='#2196F3', edgecolor='black', linewidth=0.5)
ax.hist(errors_power, bins=bins, alpha=0.6, label=f'Power-law (MAPE={np.mean(errors_power):.2f}%)',
        color='#FF9800', edgecolor='black', linewidth=0.5)
ax.axvline(np.mean(errors_exp), color='#2196F3', linestyle='--', linewidth=2)
ax.axvline(np.mean(errors_power), color='#FF9800', linestyle='--', linewidth=2)
ax.set_xlabel('Error (%)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('(a) Error Distribution Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 30)

# Panel b: Paired comparison
ax = axes[0, 1]
ax.scatter(errors_power, errors_exp, alpha=0.6, s=40, c='#4CAF50', edgecolors='black', linewidths=0.5)
lim = [0, 30]
ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('Power-law error (%)', fontsize=11)
ax.set_ylabel('Exponential error (%)', fontsize=11)
ax.set_title('(b) Paired Error Comparison', fontsize=12, fontweight='bold')
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.text(2, 25, 'Points below line:\nExponential better', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel c: High-dimensional comparison
ax = axes[1, 0]
mask_hc = (z == 2*d) & (d >= 8)
d_hc = d[mask_hc]
err_exp_hc = errors_exp[mask_hc]
err_power_hc = errors_power[mask_hc]

x = np.arange(len(d_hc))
width = 0.35
bars1 = ax.bar(x - width/2, err_power_hc, width, label='Power-law', color='#FF9800',
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, err_exp_hc, width, label='Exponential', color='#2196F3',
               edgecolor='black', linewidth=0.5)
ax.set_xlabel('Dimension', fontsize=11)
ax.set_ylabel('Error (%)', fontsize=11)
ax.set_title('(c) High-Dimensional Performance (d≥8)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'd={int(di)}' for di in d_hc], fontsize=9)
ax.legend(fontsize=9)

# Add mean lines
ax.axhline(np.mean(err_power_hc), color='#FF9800', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(np.mean(err_exp_hc), color='#2196F3', linestyle=':', linewidth=1.5, alpha=0.7)
ax.text(5.3, np.mean(err_power_hc), f'  Mean: {np.mean(err_power_hc):.1f}%',
        fontsize=8, color='#FF9800', va='center')
ax.text(5.3, np.mean(err_exp_hc), f'  Mean: {np.mean(err_exp_hc):.1f}%',
        fontsize=8, color='#2196F3', va='center')

# Panel d: Statistical significance
ax = axes[1, 1]
ax.axis('off')

# Statistical results
stats_text = """
STATISTICAL SIGNIFICANCE TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Comparison:
  • Exponential MAPE: 8.14%
  • Power-law MAPE:    8.73%
  • Difference:         0.60%

Information Criteria:
  • ΔAIC = 0.86
  • ΔBIC = 0.86
  • Threshold for "positive evidence": ΔAIC > 2

Paired t-test:
  • t-statistic = 1.150
  • p-value = 0.253

VERDICT:
  ✗ NOT STATISTICALLY SIGNIFICANT

The exponential and power-law models
are INDISTINGUISHABLE given this data.

The "factor of 7 improvement" at d≥8
is driven by only 6 data points.
"""

ax.text(0.5, 0.5, stats_text, fontsize=10, ha='center', va='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#C62828',
                  linewidth=2, alpha=0.8))
ax.set_title('(d) Statistical Assessment', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUT}/statistical_comparison.png', dpi=200, bbox_inches='tight')
print("Saved: statistical_comparison.png")
