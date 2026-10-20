from pathlib import Path
ROOT = Path(__file__).resolve().parent
"""
Generate PRL Assessment Schematic
Shows the hypothesis framework and decision tree for PRL readiness.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = ROOT

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# ============================================================
# LEFT PANEL: Hypothesis Framework
# ============================================================
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Competing Hypotheses', fontsize=14, fontweight='bold', pad=20)

# Central question box
central = FancyBboxPatch((2.5, 8.2), 5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#FFD700', edgecolor='black', linewidth=2)
ax.add_patch(central)
ax.text(5, 8.8, 'Does this work meet PRL standards?', fontsize=10,
        ha='center', va='center', fontweight='bold')

# Hypothesis boxes
hypotheses = [
    (0.2, 5.8, 'H1: Exponential form\ncaptures physical crossover', '#E3F2FD', '⚠ Unproven\np=0.25, ΔAIC=0.86'),
    (0.2, 3.6, 'H2: Difference is\nstatistical noise', '#FFF3E0', '✓ Likely correct\nModels indistinguishable'),
    (0.2, 1.4, 'H3: Useful empirical tool\nnot a PRL breakthrough', '#E8F5E9', '✓ Correct\nSolid for PRE/JSTAT'),
]

for x, y, text, color, verdict in hypotheses:
    box = FancyBboxPatch((x, y), 4.2, 1.8, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333333', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 2.1, y + 1.2, text, fontsize=8, ha='center', va='center')
    ax.text(x + 2.1, y + 0.4, verdict, fontsize=7, ha='center', va='center',
            style='italic', color='#555555')

    # Arrow from central
    arrow = FancyArrowPatch((5, 8.2), (x + 2.1, y + 1.8),
                             arrowstyle='->', mutation_scale=15,
                             color='#666666', linewidth=1.5)
    ax.add_patch(arrow)

# Evidence boxes on right
evidence_data = [
    (5.5, 6.2, 'Statistical Evidence', ['#E3F2FD',
     '• Paired t-test: p = 0.253',
     '• ΔAIC = 0.86 (< 2 threshold)',
     '• ΔMAPE = 0.60%']),
    (5.5, 4.0, 'Methodological Strengths', ['#E8F5E9',
     '• 94 lattices, d=2-13',
     '• 12 model variants tested',
     '• 10-fold CV, Bootstrap']),
    (5.5, 1.8, 'Critical Weaknesses', ['#FFEBEE',
     '• No theoretical derivation',
     '• MAPE 8.1%, 30% >10% error',
     '• d=3 bias +5.17%',
     '• Only g∈{3,4,6}']),
]

for x, y, title, (color, *lines) in evidence_data:
    box = FancyBboxPatch((x, y), 4.3, 2.0, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333333', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 2.15, y + 1.65, title, fontsize=9, ha='center', va='center',
            fontweight='bold')
    for i, line in enumerate(lines):
        ax.text(x + 0.2, y + 1.2 - i*0.35, line, fontsize=7, ha='left', va='center')

# ============================================================
# RIGHT PANEL: Decision Tree
# ============================================================
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Recommended Path Forward', fontsize=14, fontweight='bold', pad=20)

# Decision nodes
nodes = [
    (5, 9.0, 'Current manuscript', '#FFD700', '✗ NOT PRL READY'),
    (2.5, 6.8, 'Path A:\nSubmit to PRE', '#E8F5E9', 'RECOMMENDED\n2-4 weeks'),
    (5, 6.8, 'Path B:\nTheoretical derivation', '#E3F2FD', 'HIGH RISK\n6-12 months'),
    (7.5, 6.8, 'Path C:\nExtend dataset', '#FFF3E0', 'MODERATE\n3-6 months'),
    (2.5, 4.2, 'Published in PRE\nSolid contribution', '#C8E6C9', '✓ LIKELY'),
    (5, 4.2, 'PRL if successful\nPRE if failed', '#BBDEFB', '⚠ UNCERTAIN'),
    (7.5, 4.2, 'Stronger evidence\nMay still not reach PRL', '#FFE0B2', '⚠ UNCERTAIN'),
    (2.5, 1.8, 'Outcome:\nPeer-reviewed publication', '#A5D6A7', '✓ PRACTICAL'),
    (5, 1.8, 'Outcome:\nPRL or PRE\n(depends on success)', '#90CAF9', '? HIGH REWARD'),
    (7.5, 1.8, 'Outcome:\nBetter data, same venue?', '#FFCC80', '? MARGINAL'),
]

for x, y, text, color, status in nodes:
    w = 2.2 if y > 6 else 2.0
    box = FancyBboxPatch((x - w/2, y - 0.5), w, 1.2, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='#333333', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y + 0.2, text, fontsize=7.5 if y > 6 else 7, ha='center', va='center',
            fontweight='bold' if y > 6 else 'normal')
    ax.text(x, y - 0.2, status, fontsize=6.5, ha='center', va='center',
            color='#555555', style='italic')

# Arrows
arrow_pairs = [
    ((5, 8.4), (2.5, 7.4)),
    ((5, 8.4), (5, 7.4)),
    ((5, 8.4), (7.5, 7.4)),
    ((2.5, 6.2), (2.5, 4.8)),
    ((5, 6.2), (5, 4.8)),
    ((7.5, 6.2), (7.5, 4.8)),
    ((2.5, 3.6), (2.5, 2.4)),
    ((5, 3.6), (5, 2.4)),
    ((7.5, 3.6), (7.5, 2.4)),
]

for (x1, y1), (x2, y2) in arrow_pairs:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='->', mutation_scale=15,
                             color='#666666', linewidth=1.5)
    ax.add_patch(arrow)

# Key finding callout
callout = FancyBboxPatch((0.5, 0.1), 9, 0.9, boxstyle="round,pad=0.05",
                          facecolor='#FFCDD2', edgecolor='#C62828', linewidth=2)
ax.add_patch(callout)
ax.text(5, 0.55, '⚠ KEY FINDING: Exponential vs power-law difference is NOT statistically significant',
        fontsize=8.5, ha='center', va='center', fontweight='bold', color='#B71C1C')

plt.tight_layout()
plt.savefig(f'{OUT}/PRL_assessment_schematic.png', dpi=200, bbox_inches='tight')
print("Saved: PRL_assessment_schematic.png")
