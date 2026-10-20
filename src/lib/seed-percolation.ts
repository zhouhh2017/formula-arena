/**
 * 创建渗流阈值发现赛题
 */
import { db } from '@/lib/db';
import { generatePercolationData, generateOilfieldData, PC_SITE, BETA, GAMMA, NU, T_EXP } from '@/lib/percolation-data';

export async function seedPercolationProblem() {
  // 检查是否已存在
  const existing = await db.problem.findFirst({
    where: { domain: 'percolation' },
  });
  if (existing) {
    console.log('Percolation problem already exists, skipping...');
    return;
  }

  const percData = generatePercolationData(80);
  const oilData = generateOilfieldData(60);

  // ===== 赛题: 渗流概率 θ(p) 的解析结构 =====
  await db.problem.create({
    data: {
      title: '方晶格点渗流阈值与临界指数发现',
      description: `## 问题背景

方晶格点渗流 (square lattice site percolation) 是统计物理中的经典未解问题。

### 已知事实
- **边渗流**阈值 $p_c = 1/2$ (Kesten 1980 严格证明, 基于对偶性)
- **点渗流**阈值 $p_c \\approx 0.592746$ (仅数值估计, 误差 < 10⁻⁶)
- 点渗流**无已知闭式表达** — 60年来数十种方法(级数展开、重整化群、共形场论)均只能逼近

### 数学上的困难
1. 对偶性论证在点渗流上失效(对偶晶格错位)
2. $p_c$ 的无理性/超越性均为未决问题
3. 临界指数 $\\beta = 5/36$ (Nienhuis猜想), $\\nu = 4/3$ 部分情形已被Duminil-Copin严格证明

### 数据说明
- **p**: 格点占据概率 (0.1 ~ 0.99)
- **theta**: 渗流概率 θ(p) (无限簇出现的概率)
- **meanClusterSize**: 平均簇大小 S(p)
- **conductivity**: 导率 σ(p)

### 目标
发现描述 θ(p)、S(p)、σ(p) 在临界点 $p_c$ 附近的**解析公式**。
特别关注: 能否从数据中"发现" $p_c$ 的精确值或其结构?

### 评估指标
- 拟合误差 (RMSE) 权重: 60%
- 公式可解释性 权重: 40%`,
      domain: 'percolation',
      dataUrl: '/api/demo/percolation-data',
      targetVar: 'theta',
      featureVars: JSON.stringify(['p', 'meanClusterSize', 'conductivity']),
      dataPoints: JSON.stringify(percData),
      hiddenFormula: `theta = (p - ${PC_SITE})^${BETA.toFixed(5)} for p > ${PC_SITE}; p_c = ${PC_SITE} (无闭式表达)`,
      metrics: JSON.stringify({ errorWeight: 0.6, interpWeight: 0.4 }),
      publisherName: '统计物理竞技场',
      status: 'open',
    },
  });

  // ===== 赛题: 油田渗透率-孔隙度 percolation scaling =====
  await db.problem.create({
    data: {
      title: '碳酸盐岩储层渗透率临界标度律发现',
      description: `## 问题背景

在石油工程中, 渗透率 k 与孔隙度 φ 的关系是储层评价的核心问题。

### 渗流理论联系
多孔介质的输运性质遵循**渗流标度律**:
$$k = k_0 \\cdot (\\phi - \\phi_c)^\\mu \\quad \\text{for } \\phi > \\phi_c$$

其中:
- $\\phi_c$ 是**临界孔隙度** (低于此值无连通渗流通道)
- $\\mu$ 是**渗透率指数** (与空间维度和拓扑相关)
- 这直接类比于渗流导率 $\\sigma \\propto (p - p_c)^t$

### 真实油田数据特征
- 数据来自碳酸盐岩岩心分析 (类似 Katz & Thompson 1986, Wong et al. 1984 的研究)
- 低孔低渗段 (φ < 0.05): 致密灰岩, 渗透率极低
- 中孔中渗段 (0.05 < φ < 0.20): 过渡区, 幂律行为显著
- 高孔高渗段 (φ > 0.20): 礁灰岩/砂岩, 渗透率高

### 数据说明
- **porosity**: 孔隙度 φ (体积分数, 0-1)
- **permeability**: 渗透率 k (mD, milliDarcy) [目标变量]
- **formationFactor**: 地层因子 F (Archie定律: F = a/φ^m)
- **grainSize**: 平均粒径 (μm)
- **saturation**: 含水饱和度

### 目标
1. 发现 $k(\\phi)$ 的解析公式
2. 确定**临界孔隙度** $\\phi_c$
3. 确定**渗透率指数** $\\mu$
4. 验证是否为幂律 $k \\propto (\\phi - \\phi_c)^\\mu$

### 评估指标
- 拟合误差 (RMSE) 权重: 70%
- 公式可解释性 权重: 30%`,
      domain: 'oilfield',
      dataUrl: '/api/demo/oilfield-data',
      targetVar: 'permeability',
      featureVars: JSON.stringify(['porosity', 'formationFactor', 'grainSize', 'saturation']),
      dataPoints: JSON.stringify(oilData),
      hiddenFormula: `permeability = 5000 * (porosity - 0.035)^3.2 for porosity > 0.035; phi_c = 0.035, mu = 3.2`,
      metrics: JSON.stringify({ errorWeight: 0.7, interpWeight: 0.3 }),
      publisherName: '石油工程竞技场',
      status: 'open',
    },
  });

  console.log('Percolation problems created successfully!');
  console.log(`  - Percolation data: ${percData.length} points`);
  console.log(`  - Oilfield data: ${oilData.length} points`);
}
