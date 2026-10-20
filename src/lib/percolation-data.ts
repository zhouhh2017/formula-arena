/**
 * 渗流阈值发现赛题 - 数据生成器
 *
 * 生成两类数据:
 * 1. 方晶格点渗流模拟数据: 占据概率 p → 渗流概率 θ(p), 平均簇大小 S(p)
 *    已知: p_c ≈ 0.592746, β = 5/36 ≈ 0.1389, γ = 43/18 ≈ 2.389, ν = 4/3
 *    θ(p) ~ (p - p_c)^β  for p > p_c
 *    S(p) ~ |p - p_c|^(-γ)
 *
 * 2. 油田数据: 孔隙度 φ → 渗透率 k
 *    k ~ A * (φ - φ_c)^μ  for φ > φ_c
 *    基于真实碳酸盐岩/砂岩储层数据特征
 */

// ===== 方晶格点渗流模拟数据 =====
// 使用已知临界指数生成带噪声的"实验"数据
// p_c = 0.592746 (精确数值, 但无已知闭式表达)
// β = 5/36 ≈ 0.13889 (Nienhuis 猜想, 2D)
// t ≈ 1.299 (导率指数, 数值估计)

const PC_SITE = 0.592746;  // 方晶格点渗流阈值
const BETA = 5 / 36;       // 渗流概率指数 ≈ 0.13889
const GAMMA = 43 / 18;     // 平均簇大小指数 ≈ 2.3889
const NU = 4 / 3;          // 关联长度指数 ≈ 1.3333
const T_EXP = 1.299;       // 导率指数 (数值估计)

interface PercolationDataPoint {
  p: number;           // 占据概率
  theta: number;       // 渗流概率 (percolation probability)
  meanClusterSize: number;  // 平均簇大小 S(p)
  conductivity: number;     // 导率 σ(p)
}

function generatePercolationData(n: number = 80): PercolationDataPoint[] {
  const data: PercolationDataPoint[] = [];
  // 在 p_c 附近密集采样 (这是临界行为最显著的区域)
  const pValues: number[] = [];

  // 区域1: 远离 p_c (稀疏采样)
  for (let i = 0; i < 15; i++) {
    pValues.push(0.10 + i * 0.025);  // 0.10 ~ 0.45
  }
  // 区域2: p_c 附近 (密集采样)
  for (let i = 0; i < 40; i++) {
    pValues.push(0.50 + i * 0.005);  // 0.50 ~ 0.695
  }
  // 区域3: p > p_c (中等密度)
  for (let i = 0; i < 25; i++) {
    pValues.push(0.70 + i * 0.012);  // 0.70 ~ 0.988
  }

  for (const p of pValues) {
    if (p >= PC_SITE) {
      // p > p_c: 渗流发生
      const dp = p - PC_SITE;
      // θ(p) = θ_0 * (p - p_c)^β  (带噪声模拟有限尺寸效应)
      const theta = Math.pow(dp, BETA) * (1 + (Math.random() - 0.5) * 0.08);
      // S(p) = S_0 * |p - p_c|^(-γ)
      const meanClusterSize = Math.pow(dp, -GAMMA) * 0.01 * (1 + (Math.random() - 0.5) * 0.15);
      // σ(p) = σ_0 * (p - p_c)^t
      const conductivity = Math.pow(dp, T_EXP) * (1 + (Math.random() - 0.5) * 0.06);
      data.push({
        p: parseFloat(p.toFixed(6)),
        theta: parseFloat(Math.max(0, theta).toFixed(6)),
        meanClusterSize: parseFloat(Math.min(meanClusterSize, 1e6).toFixed(4)),
        conductivity: parseFloat(Math.max(0, conductivity).toFixed(6)),
      });
    } else {
      // p < p_c: 无渗流, 但有有限簇
      const dp = PC_SITE - p;
      // θ(p) ≈ 0 (有限尺寸下有微小渗流概率)
      const theta = Math.exp(-20 * dp) * 0.001 * Math.random();
      // S(p) = S_0 * |p - p_c|^(-γ) (发散)
      const meanClusterSize = Math.pow(dp, -GAMMA) * 0.01 * (1 + (Math.random() - 0.5) * 0.15);
      // σ(p) ≈ 0
      const conductivity = Math.exp(-15 * dp) * 0.0001 * Math.random();
      data.push({
        p: parseFloat(p.toFixed(6)),
        theta: parseFloat(theta.toFixed(6)),
        meanClusterSize: parseFloat(Math.min(meanClusterSize, 1e6).toFixed(4)),
        conductivity: parseFloat(conductivity.toFixed(6)),
      });
    }
  }

  return data;
}

// ===== 油田数据: 孔隙度 → 渗透率 =====
// 基于真实碳酸盐岩储层数据特征 (参考 Katz & Thompson 1986, Wong et al. 1984)
// k = k_0 * (φ - φ_c)^μ  for φ > φ_c
// φ_c ≈ 0.02-0.05 (临界孔隙度, 岩石类型相关)
// μ ≈ 2.0-4.5 (渗透率指数, 维度和拓扑相关)

interface OilfieldDataPoint {
  porosity: number;      // 孔隙度 φ (体积分数, 0-1)
  permeability: number;  // 渗透率 k (mD, milliDarcy)
  formationFactor: number; // 地层因子 F (Archie 定律相关)
  grainSize: number;     // 平均粒径 (μm)
  saturation: number;    // 含水饱和度
}

function generateOilfieldData(n: number = 60): OilfieldDataPoint[] {
  const data: OilfieldDataPoint[] = [];

  // 临界孔隙度 (不同岩性不同)
  const phiC = 0.035;  // 碳酸盐岩典型值
  // 渗透率指数 (3D 渗流, μ ≈ 2.5-4)
  const mu = 3.2;
  // 地层因子指数 (Archie: F = a / φ^m, m ≈ 2)
  const archieM = 2.0;

  // 生成不同孔隙度范围的样品 (模拟岩心分析数据)
  const porosityValues: number[] = [];
  // 低孔低渗 (致密碳酸盐岩)
  for (let i = 0; i < 20; i++) {
    porosityValues.push(0.02 + Math.random() * 0.06);  // 0.02-0.08
  }
  // 中孔中渗
  for (let i = 0; i < 25; i++) {
    porosityValues.push(0.08 + Math.random() * 0.12);  // 0.08-0.20
  }
  // 高孔高渗 (砂岩/礁灰岩)
  for (let i = 0; i < 15; i++) {
    porosityValues.push(0.20 + Math.random() * 0.15);  // 0.20-0.35
  }

  for (const phi of porosityValues) {
    if (phi > phiC) {
      // φ > φ_c: 渗透率遵循幂律
      const dphi = phi - phiC;
      // k = k_0 * (φ - φ_c)^μ, k_0 ≈ 5000 mD (岩性相关)
      const k0 = 5000;
      const permeability = k0 * Math.pow(dphi, mu) * (1 + (Math.random() - 0.5) * 0.3);
      // F = a / φ^m (Archie 定律, a ≈ 1)
      const formationFactor = (1.0 / Math.pow(phi, archieM)) * (1 + (Math.random() - 0.5) * 0.1);
      // 粒径 (与渗透率正相关, Kozeny-Carman)
      const grainSize = 50 + phi * 300 + (Math.random() - 0.5) * 30;
      // 含水饱和度
      const saturation = 0.3 + (1 - phi) * 0.4 + (Math.random() - 0.5) * 0.1;

      data.push({
        porosity: parseFloat(phi.toFixed(4)),
        permeability: parseFloat(Math.max(0.001, permeability).toFixed(4)),
        formationFactor: parseFloat(formationFactor.toFixed(4)),
        grainSize: parseFloat(grainSize.toFixed(2)),
        saturation: parseFloat(Math.max(0, Math.min(1, saturation)).toFixed(4)),
      });
    } else {
      // φ < φ_c: 极低渗 (低于临界孔隙度, 无连通路径)
      const permeability = 0.001 * Math.random();
      const formationFactor = (1.0 / Math.pow(Math.max(phi, 0.01), archieM)) * (1 + (Math.random() - 0.5) * 0.2);
      const grainSize = 20 + Math.random() * 30;
      const saturation = 0.8 + Math.random() * 0.15;

      data.push({
        porosity: parseFloat(phi.toFixed(4)),
        permeability: parseFloat(permeability.toFixed(6)),
        formationFactor: parseFloat(formationFactor.toFixed(4)),
        grainSize: parseFloat(grainSize.toFixed(2)),
        saturation: parseFloat(saturation.toFixed(4)),
      });
    }
  }

  return data;
}

// 导出
export { generatePercolationData, generateOilfieldData, PC_SITE, BETA, GAMMA, NU, T_EXP };

// 如果直接运行, 打印数据
if (typeof require !== 'undefined' && require.main === module) {
  const percData = generatePercolationData();
  const oilData = generateOilfieldData();
  console.log(`Percolation data: ${percData.length} points`);
  console.log(`Oilfield data: ${oilData.length} points`);
  console.log('\nSample percolation data (near p_c):');
  percData.filter(d => Math.abs(d.p - PC_SITE) < 0.05).slice(0, 5).forEach(d => {
    console.log(`  p=${d.p}, θ=${d.theta}, S=${d.meanClusterSize}, σ=${d.conductivity}`);
  });
  console.log('\nSample oilfield data:');
  oilData.slice(0, 5).forEach(d => {
    console.log(`  φ=${d.porosity}, k=${d.permeability} mD, F=${d.formationFactor}`);
  });
}
