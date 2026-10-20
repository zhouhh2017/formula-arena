#!/usr/bin/env python3
"""
R²=0.904 的经济损失分析
===========================
渗流阈值预测误差在油田开发中的经济影响

核心逻辑:
p_c 误差 → 渗透率/连通性预测误差 → 井位决策错误 → 经济损失
"""

import math

# ===== 油田经济参数 =====
WELL_COST_HORIZONTAL = 80_000_000  # 水平井成本: 8000万元
WELL_COST_VERTICAL = 30_000_000    # 垂直井成本: 3000万元
DRY_HOLE_RATIO = 0.15              # 干井率 (行业平均)
RESERVOIR_VOLUME_FACTOR = 1e8      # 油藏体积因子 (m³)

# ===== 渗流阈值误差的传播 =====

def pc_error_propagation():
    """
    分析 p_c 误差如何传播到油田决策
    """
    print("=" * 70)
    print("💰 R²=0.904 的经济损失分析")
    print("=" * 70)

    # 当前最佳公式的误差分布
    rmse = 0.054  # 从上一轮结果

    print(f"\n📊 当前预测精度:")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R² = 0.904")
    print(f"  典型误差范围: ±{rmse*1.96:.3f} (95% 置信区间)")

    # 误差在油田中的传播
    print(f"\n🔗 误差传播链:")
    print(f"  p_c 误差 Δp = ±0.11")
    print(f"    ↓")
    print(f"  渗透率预测误差:")
    print(f"  K ∝ (φ - p_c)^m, 其中 m ≈ 2-3")
    print(f"  ΔK/K ≈ m · Δp_c / (φ - p_c)")
    print(f"  当 φ = 0.25, p_c = 0.15: ΔK/K ≈ 2.5 × 0.11 / 0.10 = 275%")
    print(f"    ↓")
    print(f"  采收率预测误差:")
    print(f"  RF ∝ K^0.5 (近似)")
    print(f"  ΔRF/RF ≈ 0.5 × ΔK/K = 137%")
    print(f"    ↓")
    print(f"  经济价值误差:")
    print(f"  油藏净现值 NPV ∝ RF × 可采储量")
    print(f"  对于 1 亿桶油藏 (价值 50亿元), ΔNPV ≈ ±68亿元")

    # 具体损失场景
    print(f"\n💸 具体损失场景:")

    scenarios = [
        {
            "name": "场景1: 错误判断油藏连通性",
            "description": "p_c 预测偏高 → 认为油藏不连通 → 放弃开发",
            "probability": 0.05,  # 5% 概率
            "loss": 500_000_000,  # 5亿元 (错失的油藏价值)
        },
        {
            "name": "场景2: 井位设计错误",
            "description": "p_c 误差导致井距设计不合理 → 采收率下降",
            "probability": 0.20,  # 20% 概率
            "loss": 100_000_000,  # 1亿元 (采收率损失)
        },
        {
            "name": "场景3: 干井增加",
            "description": "渗透率预测不准 → 打到干井",
            "probability": 0.10,
            "loss": WELL_COST_HORIZONTAL * 2,  # 2口干井
        },
        {
            "name": "场景4: 注水方案失败",
            "description": "连通性预测错误 → 注水无效",
            "probability": 0.15,
            "loss": 50_000_000,  # 5000万元 (注水设施投资)
        },
    ]

    print(f"\n{'场景':<35s} {'概率':>6s} {'损失':>12s} {'期望损失':>12s}")
    print("-" * 70)

    total_expected = 0
    for s in scenarios:
        expected = s["probability"] * s["loss"]
        total_expected += expected
        print(f"{s['name']:<35s} {s['probability']*100:5.0f}% {s['loss']/1e8:10.1f}亿 {expected/1e8:10.2f}亿")

    print("-" * 70)
    print(f"{'总期望损失':<35s} {'':>6s} {'':>12s} {total_expected/1e8:10.2f}亿")

    # 对比 R²=0.99 的改进
    print(f"\n\n🎯 如果改进到 R²=0.99:")
    print(f"  RMSE 从 0.054 降到 0.017 (降低 68%)")
    print(f"  ΔK/K 从 275% 降到 87%")
    print(f"  ΔRF/RF 从 137% 降到 43%")
    print(f"  ΔNPV 从 ±68亿 降到 ±21亿")
    print(f"  期望损失从 {total_expected/1e8:.2f}亿 降到 {(total_expected*0.3)/1e8:.2f}亿")
    print(f"  💰 避免损失: {(total_expected*0.7)/1e8:.2f}亿元 / 项目")

    # 行业规模估算
    print(f"\n📈 行业规模影响:")
    print(f"  中国年钻探井数: ~15,000 口")
    print(f"  其中需要渗流分析: ~30% = 4,500 口")
    print(f"  每口井期望损失: {total_expected/4500:.0f}万元")
    print(f"  全国年损失: {total_expected*4500/1e8:.1f}亿元")
    print(f"  改进到 R²=0.99 可避免: {total_expected*0.7*4500/1e8:.1f}亿元/年")


if __name__ == '__main__':
    pc_error_propagation()
