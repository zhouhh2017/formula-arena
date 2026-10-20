/**
 * 扫地僧系统共享类型定义
 */

export interface DataSample {
  [varName: string]: number;
}

export interface ProblemData {
  id: string;
  title: string;
  description: string;
  domain: string;
  targetVar: string;
  featureVars: string[];
  samples: DataSample[];
  hiddenFormula?: string;
}

export interface SRResult {
  method: string;
  formula: string;
  rmse: number;
  mae: number;
  r2: number;
  complexity: number;
  iterations: number;
  durationMs: number;
  success: boolean;
  notes?: string;
}

export interface EvaluationReport {
  formula: string;
  rmse: number;
  mae: number;
  r2: number;
  complexity: number;
  interpretability: number;
  overallScore: number;
  // 细粒度误差分析
  residualStats: {
    mean: number;
    std: number;
    max: number;
    min: number;
    skewness: number;
    kurtosis: number;
  };
  // 高误差样本分析
  highErrorSamples: {
    index: number;
    features: Record<string, number>;
    actual: number;
    predicted: number;
    residual: number;
  }[];
  // 高误差样本的特征模式
  errorPatterns: string[];
  // 误差分布特征
  errorDistribution: {
    bin: string;
    count: number;
    ratio: number;
  }[];
  // 洞察与建议
  insights: string[];
  recommendations: string[];
}

export interface Hypothesis {
  id: string;
  round: number;
  formula: string;
  rationale: string;
  source: 'llm' | 'research' | 'sr_refined' | 'evaluator_feedback';
  confidence: number;
}

export interface AgentStepRecord {
  type: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  input?: string;
  output?: string;
  durationMs?: number;
  startedAt: string;
}

export interface WikiPageData {
  id?: string;
  problemId: string;
  title: string;
  content: string;
  category: 'data_exploration' | 'deep_research' | 'hypothesis' | 'sr_result' | 'evaluation' | 'insight';
  round: number;
  tags?: string[];
}
