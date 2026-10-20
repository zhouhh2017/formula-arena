/**
 * 示例赛题数据生成器
 * - 天气预测: 体感温度公式
 * - 湍流模型: 阻力系数公式 (Schiller-Naumann)
 */
import { db } from '@/lib/db';
import type { DataSample } from './monk/types';

// 生成天气预测数据
// 隐藏公式: 体感温度 = T - 0.55 * (1 - RH/100) * (T - 14.5) + 0.05 * W
// (简化的 Steadman 表观温度, 考虑温度、湿度、风速)
function generateWeatherData(n: number = 60): DataSample[] {
  const samples: DataSample[] = [];
  for (let i = 0; i < n; i++) {
    const T = -5 + Math.random() * 40; // 温度: -5 ~ 35 °C
    const RH = 20 + Math.random() * 80; // 相对湿度: 20% ~ 100%
    const W = Math.random() * 15; // 风速: 0 ~ 15 m/s
    const P = 990 + Math.random() * 40; // 气压: 990 ~ 1030 hPa

    // 体感温度 (加入少量噪声)
    const apparentT = T - 0.55 * (1 - RH / 100) * (T - 14.5) + 0.05 * W + (Math.random() - 0.5) * 0.8;

    samples.push({
      temperature: parseFloat(T.toFixed(2)),
      humidity: parseFloat(RH.toFixed(2)),
      windSpeed: parseFloat(W.toFixed(2)),
      pressure: parseFloat(P.toFixed(2)),
      apparentTemp: parseFloat(apparentT.toFixed(2)),
    });
  }
  return samples;
}

// 生成湍流模型数据
// 隐藏公式: 阻力系数 Cd = 24/Re + 6/(1+sqrt(Re)) + 0.4  (Schiller-Naumann)
function generateTurbulenceData(n: number = 60): DataSample[] {
  const samples: DataSample[] = [];
  for (let i = 0; i < n; i++) {
    // 雷诺数: 0.1 ~ 100000 (对数分布)
    const Re = Math.pow(10, -1 + Math.random() * 6);
    const viscosity = 1e-6 + Math.random() * 1e-5; // 动力粘度
    const density = 990 + Math.random() * 30; // 密度
    const velocity = 0.01 + Math.random() * 5; // 速度
    const diameter = 0.001 + Math.random() * 0.01; // 粒径

    // Schiller-Naumann 阻力系数 (加入少量噪声)
    const sqrtRe = Math.sqrt(Re);
    const Cd = 24 / Re + 6 / (1 + sqrtRe) + 0.4 + (Math.random() - 0.5) * 0.05;

    samples.push({
      reynolds: parseFloat(Re.toFixed(4)),
      viscosity: parseFloat(viscosity.toExponential(3)),
      density: parseFloat(density.toFixed(2)),
      velocity: parseFloat(velocity.toFixed(4)),
      diameter: parseFloat(diameter.toFixed(5)),
      dragCoeff: parseFloat(Cd.toFixed(4)),
    });
  }
  return samples;
}

// 初始化示例赛题
export async function seedDemoProblems() {
  // 检查是否已有数据
  const count = await db.problem.count();
  if (count > 0) return;

  // ===== 赛题 1: 天气预测 =====
  const weatherSamples = generateWeatherData(60);
  await db.problem.create({
    data: {
      title: '体感温度预测公式发现',
      description: `给定一组气象观测数据 (温度、相对湿度、风速、气压), 发现预测体感温度 (apparentTemp) 的隐含公式。

## 背景
体感温度是人体实际感受到的温度, 受气温、湿度、风速等综合影响。本赛题要求发现一个数学公式, 能够根据气象变量准确预测体感温度。

## 数据说明
- temperature: 气温 (°C), 范围 -5 ~ 35
- humidity: 相对湿度 (%), 范围 20 ~ 100
- windSpeed: 风速 (m/s), 范围 0 ~ 15
- pressure: 气压 (hPa), 范围 990 ~ 1030
- apparentTemp: 体感温度 (°C) [目标变量]

## 评估指标
- 拟合误差 (RMSE) 权重: 60%
- 公式可解释性 权重: 40%`,
      domain: 'weather',
      dataUrl: '/api/demo/weather-data',
      targetVar: 'apparentTemp',
      featureVars: JSON.stringify(['temperature', 'humidity', 'windSpeed', 'pressure']),
      dataPoints: JSON.stringify(weatherSamples),
      hiddenFormula: 'apparentTemp = temperature - 0.55 * (1 - humidity/100) * (temperature - 14.5) + 0.05 * windSpeed',
      metrics: JSON.stringify({ errorWeight: 0.6, interpWeight: 0.4 }),
      publisherName: '气象科学竞技场',
      status: 'open',
    },
  });

  // ===== 赛题 2: 湍流模型 =====
  const turbSamples = generateTurbulenceData(60);
  await db.problem.create({
    data: {
      title: '颗粒阻力系数湍流模型发现',
      description: `给定流体力学实验数据 (雷诺数、粘度、密度、速度、粒径), 发现预测球形颗粒阻力系数 (dragCoeff) 的隐含公式。

## 背景
阻力系数 Cd 是流体力学中的核心参数, 描述颗粒在流体中运动时受到的阻力。它与雷诺数 Re 有复杂的非线性关系, 在不同流态 (Stokes区、过渡区、牛顿区) 有不同的经验公式。

## 数据说明
- reynolds: 雷诺数 Re, 范围 0.1 ~ 100000
- viscosity: 动力粘度 (Pa·s)
- density: 流体密度 (kg/m³)
- velocity: 相对速度 (m/s)
- diameter: 颗粒直径 (m)
- dragCoeff: 阻力系数 Cd [目标变量]

## 评估指标
- 拟合误差 (RMSE) 权重: 70%
- 公式可解释性 权重: 30%`,
      domain: 'turbulence',
      dataUrl: '/api/demo/turbulence-data',
      targetVar: 'dragCoeff',
      featureVars: JSON.stringify(['reynolds', 'viscosity', 'density', 'velocity', 'diameter']),
      dataPoints: JSON.stringify(turbSamples),
      hiddenFormula: 'dragCoeff = 24/reynolds + 6/(1+sqrt(reynolds)) + 0.4',
      metrics: JSON.stringify({ errorWeight: 0.7, interpWeight: 0.3 }),
      publisherName: '流体力学竞技场',
      status: 'open',
    },
  });
}
