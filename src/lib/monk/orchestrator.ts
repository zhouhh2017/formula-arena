/**
 * 扫地僧编排器 - 主工作循环
 *
 * 工作流程:
 * 1. 数据探索 (coding agent)
 * 2. 深度调研 (deep research) -> 落盘 wiki
 * 3. LLM 提出公式假设 -> 落盘 wiki
 * 4. 并行运行多种符号回归方法 (步数限制使各方法同时跑完)
 * 5. 拟合误差评估器分析 -> 落盘 wiki
 * 6. 从第二轮起, 基于评估反馈提出更进一步的假设
 *
 * 代际进化:
 * - runMonkGeneration 包装 runMonk, 增加经验记录和代际追踪
 * - 后代通过 ExperienceWiki 读取前人经验, 避免重复踩坑
 * - 遗传算法从前代种群中选择精英 + 交叉/变异产生种子公式
 */
import { db } from '@/lib/db';
import { WikiStore } from './wiki';
import { ExperienceWiki } from './experience-wiki';
import { exploreData, buildExplorationWiki, type DataExplorationResult } from './data-explorer';
import { deepResearch, buildResearchWiki, type ResearchResult } from './deep-research';
import { generateHypotheses, buildHypothesisWiki, type Hypothesis } from './hypothesis';
import { runGenetic } from './sr/genetic';
import { runSINDy } from './sr/sindy';
import { runLLMGuided } from './sr/llm-guided';
import { runTemplateMatching } from './sr/template';
import { evaluateFormula, buildEvaluationWiki, buildEvaluatorFeedback } from './evaluator';
import type { ProblemData, SRResult, EvaluationReport, AgentStepRecord } from './types';

export interface MonkRunOptions {
  maxRounds?: number;
  maxSRSteps?: number;
  onProgress?: (status: string, step?: AgentStepRecord) => void;
  // 代际进化上下文
  generation?: number;
  generationId?: string;
  seedFormulas?: string[]; // 来自遗传算法的种子公式
  experienceContext?: string; // 来自 llm-wiki 的前人经验
  avoidPitfalls?: string[]; // 避坑指南
}

export interface MonkRunResult {
  runId: string;
  rounds: {
    round: number;
    exploration?: DataExplorationResult;
    research?: ResearchResult;
    hypotheses: Hypothesis[];
    srResults: SRResult[];
    evaluation?: EvaluationReport;
  }[];
  bestFormula: string;
  bestError: number;
  bestReport?: EvaluationReport;
}

export async function runMonk(
  problem: ProblemData,
  metrics: { errorWeight: number; interpWeight: number },
  options: MonkRunOptions = {}
): Promise<MonkRunResult> {
  const maxRounds = options.maxRounds ?? 2;
  const maxSRSteps = options.maxSRSteps ?? 4;

  // 创建 AgentRun 记录
  const run = await db.agentRun.create({
    data: {
      problemId: problem.id,
      round: 1,
      status: 'exploring',
      currentStep: '数据探索',
      stepsJson: '[]',
    },
  });

  const wiki = new WikiStore(problem.id);
  const rounds: MonkRunResult['rounds'] = [];
  let bestFormula = '';
  let bestError = Infinity;
  let bestReport: EvaluationReport | undefined;
  let evaluatorFeedback: string | undefined;

  const updateProgress = async (status: string, currentStep: string) => {
    await db.agentRun.update({
      where: { id: run.id },
      data: { status, currentStep },
    });
    options.onProgress?.(`${status}: ${currentStep}`);
  };

  const addStep = async (step: AgentStepRecord) => {
    const steps = JSON.parse((await db.agentRun.findUnique({ where: { id: run.id } }))?.stepsJson ?? '[]');
    steps.push(step);
    await db.agentRun.update({
      where: { id: run.id },
      data: { stepsJson: JSON.stringify(steps) },
    });
    await db.agentStep.create({
      data: {
        runId: run.id,
        type: step.type,
        name: step.name,
        input: step.input ?? '',
        output: step.output ?? '',
        status: step.status,
        duration: step.durationMs ?? 0,
      },
    });
  };

  for (let round = 1; round <= maxRounds; round++) {
    const roundData: MonkRunResult['rounds'][0] = {
      round,
      hypotheses: [],
      srResults: [],
    };

    // ===== 步骤 1: 数据探索 (仅第一轮) =====
    if (round === 1) {
      await updateProgress('exploring', `第${round}轮 - 数据探索`);
      const stepStart = Date.now();
      const exploration = exploreData(problem);
      const wikiContent = buildExplorationWiki(problem, exploration);
      await wiki.write({
        title: `第${round}轮数据探索`,
        content: wikiContent,
        category: 'data_exploration',
        round,
        tags: ['statistics', 'correlation', problem.domain],
      });
      roundData.exploration = exploration;
      await addStep({
        type: 'data_explore',
        name: `第${round}轮数据探索`,
        status: 'completed',
        output: `统计了${exploration.stats.length}个变量, 发现${exploration.correlations.length}个相关关系`,
        durationMs: Date.now() - stepStart,
        startedAt: new Date(stepStart).toISOString(),
      });
    }

    // ===== 步骤 2: 深度调研 (仅第一轮或需要扩充知识时) =====
    if (round === 1) {
      await updateProgress('researching', `第${round}轮 - 深度调研`);
      const stepStart = Date.now();
      try {
        const research = await deepResearch(
          problem,
          roundData.exploration?.insights ?? []
        );
        const wikiContent = buildResearchWiki(problem, research);
        await wiki.write({
          title: `第${round}轮深度调研`,
          content: wikiContent,
          category: 'deep_research',
          round,
          tags: ['web_search', 'literature', problem.domain],
        });
        roundData.research = research;
        await addStep({
          type: 'deep_research',
          name: `第${round}轮深度调研`,
          status: 'completed',
          output: `搜索到${research.findings.length}个发现, 提取${research.candidateFormulas.length}个候选公式`,
          durationMs: Date.now() - stepStart,
          startedAt: new Date(stepStart).toISOString(),
        });
      } catch (e) {
        await addStep({
          type: 'deep_research',
          name: `第${round}轮深度调研`,
          status: 'failed',
          output: `调研失败: ${(e as Error).message}`,
          durationMs: Date.now() - stepStart,
          startedAt: new Date(stepStart).toISOString(),
        });
      }
    }

    // ===== 步骤 3: 提出公式假设 =====
    await updateProgress('hypothesizing', `第${round}轮 - 提出公式假设`);
    const stepStart = Date.now();
    let wikiContext = await wiki.asContext(6000);
    // 代际进化: 注入前人经验和种子公式
    if (options.experienceContext) {
      wikiContext += `\n\n## 前人经验 (llm-wiki)\n${options.experienceContext}`;
    }
    if (options.seedFormulas && options.seedFormulas.length > 0) {
      wikiContext += `\n\n## 遗传算法种子公式 (来自前代进化)\n${options.seedFormulas.map((f, i) => `${i + 1}. \`${f}\``).join('\n')}`;
    }
    if (options.avoidPitfalls && options.avoidPitfalls.length > 0) {
      wikiContext += `\n\n## 避坑指南 (前代踩过的坑)\n${options.avoidPitfalls.map((p, i) => `${i + 1}. ${p}`).join('\n')}`;
    }
    const hypotheses = await generateHypotheses(
      problem,
      wikiContext,
      round,
      evaluatorFeedback
    );
    await wiki.write({
      title: `第${round}轮公式假设`,
      content: buildHypothesisWiki(hypotheses, round),
      category: 'hypothesis',
      round,
      tags: ['llm', 'hypothesis'],
    });
    roundData.hypotheses = hypotheses;
    await addStep({
      type: 'hypothesis',
      name: `第${round}轮公式假设`,
      status: 'completed',
      output: `提出${hypotheses.length}个候选公式`,
      durationMs: Date.now() - stepStart,
      startedAt: new Date(stepStart).toISOString(),
    });

    // ===== 步骤 4: 并行运行多种符号回归方法 =====
    await updateProgress('regressing', `第${round}轮 - 符号回归探索`);
    const srStepStart = Date.now();

    // 选择 top-3 假设作为各方法的种子
    const topHypotheses = hypotheses.slice(0, 3);
    const seedFormula = topHypotheses[0]?.formula || problem.featureVars[0];

    // 并行运行 (限制步数使各方法同时跑完)
    const srPromises: Promise<SRResult>[] = [
      // 遗传规划: 步数 = maxSRSteps * 5 代
      runGenetic(problem.samples, problem.featureVars, problem.targetVar, seedFormula, { maxSteps: maxSRSteps * 5, populationSize: 30 }),
      // SINDy: 步数 = maxSRSteps (迭代次数)
      runSINDy(problem.samples, problem.featureVars, problem.targetVar, seedFormula, { maxSteps: maxSRSteps }),
      // LLM 引导: 步数 = maxSRSteps
      runLLMGuided(problem.samples, problem.featureVars, problem.targetVar, seedFormula, wikiContext, { maxSteps: maxSRSteps }),
      // 模板匹配: 步数 = maxSRSteps * 3 个模板
      runTemplateMatching(problem.samples, problem.featureVars, problem.targetVar, seedFormula, { maxSteps: maxSRSteps * 3 }),
    ];

    const srResults = await Promise.allSettled(srPromises);
    const successfulResults: SRResult[] = [];
    for (let i = 0; i < srResults.length; i++) {
      const r = srResults[i];
      if (r.status === 'fulfilled') {
        successfulResults.push(r.value);
      } else {
        successfulResults.push({
          method: ['遗传规划', 'SINDy', 'LLM引导', '模板匹配'][i],
          formula: seedFormula,
          rmse: Infinity,
          mae: Infinity,
          r2: -Infinity,
          complexity: 0,
          iterations: 0,
          durationMs: 0,
          success: false,
          notes: `执行失败: ${r.reason?.message ?? 'unknown'}`,
        });
      }
    }

    // 落盘 SR 结果到 wiki
    let srWikiContent = `### 第 ${round} 轮符号回归结果\n\n`;
    for (const r of successfulResults) {
      srWikiContent += `#### ${r.method}\n\n`;
      srWikiContent += `- **公式**: \`${r.formula}\`\n`;
      srWikiContent += `- **RMSE**: ${r.rmse.toFixed(4)}\n`;
      srWikiContent += `- **R²**: ${r.r2.toFixed(4)}\n`;
      srWikiContent += `- **复杂度**: ${r.complexity}\n`;
      srWikiContent += `- **迭代次数**: ${r.iterations}\n`;
      srWikiContent += `- **耗时**: ${r.durationMs}ms\n`;
      srWikiContent += `- **备注**: ${r.notes}\n\n`;
    }
    await wiki.write({
      title: `第${round}轮符号回归结果`,
      content: srWikiContent,
      category: 'sr_result',
      round,
      tags: ['symbolic_regression', 'parallel'],
    });
    roundData.srResults = successfulResults;

    await addStep({
      type: 'symbolic_regression',
      name: `第${round}轮符号回归 (4种方法并行)`,
      status: 'completed',
      input: `种子公式: ${seedFormula}`,
      output: successfulResults.map((r) => `${r.method}: RMSE=${r.rmse.toFixed(4)}`).join('; '),
      durationMs: Date.now() - srStepStart,
      startedAt: new Date(srStepStart).toISOString(),
    });

    // ===== 步骤 5: 评估器分析 =====
    await updateProgress('evaluating', `第${round}轮 - 误差评估与洞察`);
    const evalStepStart = Date.now();

    // 选择最佳 SR 结果
    const bestSR = successfulResults.sort((a, b) => a.rmse - b.rmse)[0];
    if (bestSR && bestSR.success) {
      const report = await evaluateFormula(problem, bestSR.formula, bestSR, metrics);
      const evalWiki = await buildEvaluationWiki(problem, successfulResults, report, round);
      await wiki.write({
        title: `第${round}轮评估报告`,
        content: evalWiki,
        category: 'evaluation',
        round,
        tags: ['evaluation', 'error_analysis', 'insight'],
      });
      roundData.evaluation = report;

      if (report.rmse < bestError) {
        bestError = report.rmse;
        bestFormula = report.formula;
        bestReport = report;
      }

      // 生成下一轮反馈
      evaluatorFeedback = buildEvaluatorFeedback(report);

      await addStep({
        type: 'evaluation',
        name: `第${round}轮评估分析`,
        status: 'completed',
        input: `最佳公式: ${bestSR.formula}`,
        output: `RMSE=${report.rmse.toFixed(4)}, R²=${report.r2.toFixed(4)}, 发现${report.errorPatterns.length}个误差模式, ${report.insights.length}条洞察`,
        durationMs: Date.now() - evalStepStart,
        startedAt: new Date(evalStepStart).toISOString(),
      });
    } else {
      await addStep({
        type: 'evaluation',
        name: `第${round}轮评估分析`,
        status: 'failed',
        output: '所有符号回归方法均失败',
        durationMs: Date.now() - evalStepStart,
        startedAt: new Date(evalStepStart).toISOString(),
      });
    }

    rounds.push(roundData);

    // 更新 run 记录
    await db.agentRun.update({
      where: { id: run.id },
      data: {
        round,
        bestFormula,
        bestError: bestError === Infinity ? null : bestError,
      },
    });
  }

  // ===== 完成: 提交最佳公式到竞技场 =====
  if (bestFormula && bestReport) {
    await db.submission.create({
      data: {
        problemId: problem.id,
        submitterName: '扫地僧 (Monk Baseline)',
        formula: bestFormula,
        report: `扫地僧自动发现\n\n${bestReport.insights.join('\n')}`,
        fittingError: bestReport.rmse,
        r2Score: bestReport.r2,
        maeScore: bestReport.mae,
        interpretabilityScore: bestReport.interpretability,
        complexityScore: bestReport.complexity,
        overallScore: bestReport.overallScore,
        isMonk: true,
        status: 'evaluated',
      },
    });
  }

  await db.agentRun.update({
    where: { id: run.id },
    data: {
      status: 'completed',
      currentStep: '完成',
      completedAt: new Date(),
      bestFormula,
      bestError: bestError === Infinity ? null : bestError,
    },
  });

  return {
    runId: run.id,
    rounds,
    bestFormula,
    bestError,
    bestReport,
  };
}

// ===== 代际进化: 记录经验到 llm-wiki =====
async function recordExperiences(
  problemId: string,
  generationId: string | undefined,
  generation: number,
  rounds: MonkRunResult['rounds'],
  bestFormula: string,
  bestReport: EvaluationReport | undefined
) {
  const expWiki = new ExperienceWiki(problemId);

  // 1. 记录成功经验
  if (bestFormula && bestReport) {
    await expWiki.ingest({
      generationId,
      generation,
      type: 'success',
      title: `第${generation}代最佳公式: ${bestFormula.slice(0, 60)}`,
      content: `**公式**: \`${bestFormula}\`\n**RMSE**: ${bestReport.rmse.toFixed(4)}\n**R²**: ${bestReport.r2.toFixed(4)}\n**可解释性**: ${bestReport.interpretability.toFixed(1)}/100\n**综合评分**: ${bestReport.overallScore.toFixed(2)}\n\n**关键洞察**: ${bestReport.insights.join('; ')}`,
      tags: ['best_formula', `gen_${generation}`],
      formula: bestFormula,
      error: bestReport.rmse,
      importance: 0.9,
    });
  }

  // 2. 记录各轮评估洞察
  for (const r of rounds) {
    if (!r.evaluation) continue;
    const report = r.evaluation;

    // 记录误差模式 (作为避坑指南)
    if (report.errorPatterns.length > 0) {
      await expWiki.ingest({
        generationId,
        generation,
        type: 'pitfall_avoidance',
        title: `第${generation}代发现的误差模式 (第${r.round}轮)`,
        content: `最佳公式 \`${report.formula}\` 在以下样本上误差较大:\n${report.errorPatterns.join('\n')}\n\n**高误差样本特征**: ${report.highErrorSamples.slice(0, 3).map(s => JSON.stringify(s.features)).join('; ')}`,
        tags: ['error_pattern', 'high_error_samples', `gen_${generation}`],
        formula: report.formula,
        error: report.rmse,
        importance: 0.7,
      });
    }

    // 记录改进建议 (作为技巧)
    if (report.recommendations.length > 0) {
      await expWiki.ingest({
        generationId,
        generation,
        type: 'technique',
        title: `第${generation}代改进建议 (第${r.round}轮)`,
        content: `针对公式 \`${report.formula}\` 的改进建议:\n${report.recommendations.join('\n')}`,
        tags: ['improvement', 'recommendations', `gen_${generation}`],
        importance: 0.6,
      });
    }

    // 记录洞察
    if (report.insights.length > 0) {
      await expWiki.ingest({
        generationId,
        generation,
        type: 'insight',
        title: `第${generation}代数据洞察 (第${r.round}轮)`,
        content: report.insights.join('\n'),
        tags: ['insight', 'data_analysis', `gen_${generation}`],
        importance: 0.5,
      });
    }
  }

  // 3. 记录失败的SR方法
  for (const r of rounds) {
    for (const sr of r.srResults) {
      if (!sr.success || sr.rmse > 1e8) {
        await expWiki.ingest({
          generationId,
          generation,
          type: 'failure',
          title: `第${generation}代SR方法失败: ${sr.method}`,
          content: `方法 ${sr.method} 在第${r.round}轮失败。\n公式: \`${sr.formula}\`\n备注: ${sr.notes || '无'}`,
          tags: ['failure', 'sr_method', sr.method, `gen_${generation}`],
          formula: sr.formula,
          error: sr.rmse,
          importance: 0.4,
        });
      }
    }
  }
}

// ===== 代际进化: 运行某一代扫地僧 =====
export async function runMonkGeneration(
  problem: ProblemData,
  metrics: { errorWeight: number; interpWeight: number },
  generationId: string,
  generation: number,
  options: {
    seedFormulas?: string[];
    experienceContext?: string;
    avoidPitfalls?: string[];
    maxRounds?: number;
    maxSRSteps?: number;
  } = {}
): Promise<MonkRunResult> {
  // 更新代际状态为 running
  await db.monkGeneration.update({
    where: { id: generationId },
    data: { status: 'running', startedAt: new Date() },
  });

  try {
    const result = await runMonk(problem, metrics, {
      maxRounds: options.maxRounds ?? 2,
      maxSRSteps: options.maxSRSteps ?? 4,
      generation,
      generationId,
      seedFormulas: options.seedFormulas,
      experienceContext: options.experienceContext,
      avoidPitfalls: options.avoidPitfalls,
    });

    // 记录经验到 llm-wiki
    await recordExperiences(
      problem.id,
      generationId,
      generation,
      result.rounds,
      result.bestFormula,
      result.bestReport
    );

    // 更新代际记录
    await db.monkGeneration.update({
      where: { id: generationId },
      data: {
        status: 'completed',
        bestFormula: result.bestFormula,
        bestError: result.bestError === Infinity ? null : result.bestError,
        bestR2: result.bestReport?.r2 ?? null,
        bestScore: result.bestReport?.overallScore ?? null,
        runId: result.runId,
        completedAt: new Date(),
      },
    });

    return result;
  } catch (e) {
    await db.monkGeneration.update({
      where: { id: generationId },
      data: { status: 'failed', completedAt: new Date() },
    });
    throw e;
  }
}
