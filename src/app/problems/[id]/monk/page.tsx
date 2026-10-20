"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";

interface WikiPage {
  id: string;
  title: string;
  content: string;
  category: string;
  round: number;
  tags: string[];
  createdAt: string;
}

interface AgentStep {
  id: string;
  type: string;
  name: string;
  status: string;
  output: string;
  duration: number;
  createdAt: string;
}

interface AgentRun {
  id: string;
  round: number;
  status: string;
  currentStep: string;
  bestFormula: string | null;
  bestError: number | null;
  startedAt: string;
  completedAt: string | null;
  steps: AgentStep[];
}

interface Generation {
  id: string;
  generation: number;
  status: string;
  bestFormula?: string;
  bestError?: number;
  bestScore?: number;
  totalExperiences: number;
  parentFormula?: string | null;
  parentError?: number | null;
  parentGen?: number | null;
  autoExecute?: boolean;
  plan?: any;
  createdAt?: string;
  approvedAt?: string | null;
  startedAt?: string | null;
  completedAt?: string | null;
  experiences?: { id: string; type: string; title: string; content: string; formula?: string | null; error?: number | null; importance: number }[];
}

interface Experience {
  id: string;
  type: string;
  title: string;
  content: string;
  formula?: string;
  error?: number;
  importance: number;
  generation?: number;
  tags?: string[];
  createdAt: string;
}

const CATEGORY_META: Record<string, { label: string; icon: string; color: string }> = {
  data_exploration: { label: "数据探索", icon: "📊", color: "bg-blue-500/10 text-blue-600 border-blue-500/20" },
  deep_research: { label: "深度调研", icon: "🔍", color: "bg-purple-500/10 text-purple-600 border-purple-500/20" },
  hypothesis: { label: "公式假设", icon: "💡", color: "bg-amber-500/10 text-amber-600 border-amber-500/20" },
  sr_result: { label: "符号回归", icon: "🧮", color: "bg-cyan-500/10 text-cyan-600 border-cyan-500/20" },
  evaluation: { label: "评估分析", icon: "🎯", color: "bg-green-500/10 text-green-600 border-green-500/20" },
  insight: { label: "洞察", icon: "✨", color: "bg-pink-500/10 text-pink-600 border-pink-500/20" },
};

const STEP_TYPE_META: Record<string, { label: string; icon: string }> = {
  data_explore: { label: "数据探索", icon: "📊" },
  deep_research: { label: "深度调研", icon: "🔍" },
  hypothesis: { label: "公式假设", icon: "💡" },
  symbolic_regression: { label: "符号回归", icon: "🧮" },
  evaluation: { label: "评估分析", icon: "🎯" },
};

export default function MonkPage() {
  const params = useParams<{ id: string }>();
  const problemId = params.id;
  const { toast } = useToast();

  const [wikiPages, setWikiPages] = useState<WikiPage[]>([]);
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [experiences, setExperiences] = useState<Experience[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [planning, setPlanning] = useState(false);
  const [executingGen, setExecutingGen] = useState<string | null>(null);
  const [selectedPage, setSelectedPage] = useState<WikiPage | null>(null);
  const [selectedGen, setSelectedGen] = useState<Generation | null>(null);
  const [autoGenerations, setAutoGenerations] = useState(0);

  const load = useCallback(async () => {
    const [wRes, rRes, gRes, eRes] = await Promise.all([
      fetch(`/api/problems/${problemId}/wiki`).then((r) => r.json()),
      fetch(`/api/problems/${problemId}/monk/status`).then((r) => r.json()),
      fetch(`/api/problems/${problemId}/generations`).then((r) => r.json()),
      fetch(`/api/problems/${problemId}/experiences`).then((r) => r.json()),
    ]);
    setWikiPages(wRes.pages || []);
    setRuns(rRes.runs || []);
    setGenerations(gRes.generations || []);
    setExperiences(eRes.experiences || []);
    setLoading(false);
  }, [problemId]);

  useEffect(() => {
    load();
    // 轮询: 如果有运行中的任务, 每3秒刷新
    const interval = setInterval(() => {
      if (runs.some((r) => r.status === "running" || r.status === "pending")) {
        load();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [load, runs]);

  const handleRun = async () => {
    setRunning(true);
    toast({ title: "扫地僧启动", description: "正在执行完整工作循环..." });
    try {
      const res = await fetch(`/api/problems/${problemId}/monk/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ maxRounds: 2 }),
      });
      const data = await res.json();
      if (data.error) {
        toast({ title: "运行失败", description: data.error, variant: "destructive" });
      } else {
        toast({
          title: "扫地僧完成!",
          description: `最佳公式: ${data.bestFormula?.slice(0, 50)}...`,
        });
        await load();
      }
    } catch (e) {
      toast({ title: "运行失败", description: (e as Error).message, variant: "destructive" });
    }
    setRunning(false);
  };

  const handlePlanNextGen = async () => {
    setPlanning(true);
    toast({ title: "规划下一代", description: "基于 llm-wiki 经验和遗传算法生成探索计划..." });
    try {
      const res = await fetch(`/api/problems/${problemId}/generations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ autoExecute: false }),
      });
      const data = await res.json();
      if (data.error) {
        toast({ title: "规划失败", description: data.error, variant: "destructive" });
      } else {
        toast({
          title: `第${data.generation}代计划已生成`,
          description: "请审阅计划后手动执行",
        });
        await load();
      }
    } catch (e) {
      toast({ title: "规划失败", description: (e as Error).message, variant: "destructive" });
    }
    setPlanning(false);
  };

  const handleExecuteGen = async (genId: string, gen: number) => {
    setExecutingGen(genId);
    toast({ title: `执行第${gen}代`, description: "扫地僧正在基于前人经验探索..." });
    try {
      const res = await fetch(`/api/problems/${problemId}/generations/${genId}/execute`, {
        method: "POST",
      });
      const data = await res.json();
      if (data.error) {
        toast({ title: "执行失败", description: data.error, variant: "destructive" });
      } else {
        toast({
          title: `第${gen}代完成!`,
          description: `最佳公式: ${data.bestFormula?.slice(0, 50)}...`,
        });
        await load();
      }
    } catch (e) {
      toast({ title: "执行失败", description: (e as Error).message, variant: "destructive" });
    }
    setExecutingGen(null);
  };

  const handleToggleAutoExecute = async (genId: string, current: boolean) => {
    try {
      await fetch(`/api/problems/${problemId}/generations/${genId}/plan`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ autoExecute: !current }),
      });
      await load();
    } catch (e) {
      toast({ title: "更新失败", description: (e as Error).message, variant: "destructive" });
    }
  };

  const handleViewGenDetail = async (genId: string) => {
    try {
      const res = await fetch(`/api/problems/${problemId}/generations/${genId}/plan`);
      const data = await res.json();
      setSelectedGen(data.generation);
    } catch (e) {
      toast({ title: "加载失败", description: (e as Error).message, variant: "destructive" });
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto max-w-6xl px-4 py-8 space-y-4">
        <Skeleton className="h-12 w-1/2" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const latestRun = runs[0];
  const isRunning = latestRun && (latestRun.status === "running" || latestRun.status === "pending");

  // 按轮次分组 wiki
  const wikiByRound = wikiPages.reduce((acc, p) => {
    if (!acc[p.round]) acc[p.round] = [];
    acc[p.round].push(p);
    return acc;
  }, {} as Record<number, WikiPage[]>);

  return (
    <div className="container mx-auto max-w-6xl px-4 py-8 space-y-6">
      {/* 头部 */}
      <div>
        <Link href={`/problems/${problemId}`} className="text-sm text-muted-foreground hover:text-foreground mb-2 inline-block">
          ← 返回赛题
        </Link>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-3">
              <span className="text-4xl">🧘</span>
              扫地僧工作面板
            </h1>
            <p className="text-muted-foreground mt-1">观察扫地僧的完整工作循环：探索→调研→假设→回归→评估→迭代</p>
          </div>
          <Button onClick={handleRun} disabled={running || isRunning} className="bg-gradient-to-r from-indigo-500 to-purple-500">
            {running || isRunning ? "运行中..." : "▶ 启动扫地僧"}
          </Button>
        </div>
      </div>

      {/* 最新运行概览 */}
      {latestRun && (
        <Card className="bg-gradient-to-br from-indigo-500/5 to-purple-500/5 border-indigo-500/20">
          <CardHeader>
            <CardTitle className="text-lg flex items-center justify-between">
              <span>最新运行 · Round {latestRun.round}</span>
              <RunStatusBadge status={latestRun.status} />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">当前步骤</div>
                <div className="font-medium">{latestRun.currentStep || "—"}</div>
              </div>
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">最佳公式</div>
                <div className="font-mono text-sm truncate" title={latestRun.bestFormula || ""}>
                  {latestRun.bestFormula || "—"}
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">最佳误差 (RMSE)</div>
                <div className="font-mono font-bold text-lg">
                  {latestRun.bestError !== null ? latestRun.bestError.toFixed(4) : "—"}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="workflow">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="workflow">⚙️ 工作流</TabsTrigger>
          <TabsTrigger value="wiki">📚 Wiki ({wikiPages.length})</TabsTrigger>
          <TabsTrigger value="generations">🧬 代际进化 ({generations.length})</TabsTrigger>
          <TabsTrigger value="experiences">🧠 经验库 ({experiences.length})</TabsTrigger>
          <TabsTrigger value="runs">📜 运行历史 ({runs.length})</TabsTrigger>
        </TabsList>

        {/* 工作流 Tab */}
        <TabsContent value="workflow" className="space-y-4">
          {runs.length === 0 ? (
            <Card>
              <CardContent className="py-16 text-center text-muted-foreground">
                <div className="text-5xl mb-4">🧘</div>
                <p>扫地僧尚未运行。点击右上角「启动扫地僧」开始工作循环。</p>
              </CardContent>
            </Card>
          ) : (
            <>
              {runs.slice().reverse().map((run) => (
                <Card key={run.id}>
                  <CardHeader>
                    <CardTitle className="text-base flex items-center justify-between">
                      <span>Round {run.round}</span>
                      <RunStatusBadge status={run.status} />
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {run.steps.length === 0 ? (
                        <div className="text-sm text-muted-foreground">暂无步骤记录</div>
                      ) : (
                        run.steps.map((step) => {
                          const meta = STEP_TYPE_META[step.type] || { label: step.type, icon: "⚙️" };
                          return (
                            <div key={step.id} className="flex items-start gap-3 p-3 rounded-md bg-muted/30">
                              <div className="text-xl">{meta.icon}</div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <span className="font-medium text-sm">{step.name}</span>
                                  <Badge variant="outline" className="text-xs">{meta.label}</Badge>
                                  <StepStatusBadge status={step.status} />
                                  {step.duration > 0 && (
                                    <span className="text-xs text-muted-foreground">{(step.duration / 1000).toFixed(1)}s</span>
                                  )}
                                </div>
                                {step.output && (
                                  <pre className="mt-2 text-xs font-mono whitespace-pre-wrap text-muted-foreground max-h-40 overflow-y-auto bg-background/50 p-2 rounded">
                                    {step.output.slice(0, 800)}
                                    {step.output.length > 800 && "..."}
                                  </pre>
                                )}
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </>
          )}
        </TabsContent>

        {/* Wiki Tab */}
        <TabsContent value="wiki" className="space-y-4">
          {wikiPages.length === 0 ? (
            <Card>
              <CardContent className="py-16 text-center text-muted-foreground">
                <div className="text-5xl mb-4">📚</div>
                <p>Wiki 记忆库为空。扫地僧运行后，所有调研、假设、评估结果将落盘到这里。</p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-6">
              {Object.entries(wikiByRound)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([round, pages]) => (
                  <div key={round}>
                    <h3 className="text-sm font-semibold text-muted-foreground mb-2">Round {round}</h3>
                    <div className="grid md:grid-cols-2 gap-3">
                      {pages.map((page) => {
                        const meta = CATEGORY_META[page.category] || CATEGORY_META.insight;
                        return (
                          <Card
                            key={page.id}
                            className="cursor-pointer hover:shadow-md hover:border-indigo-500/30 transition-all"
                            onClick={() => setSelectedPage(page)}
                          >
                            <CardHeader className="pb-2">
                              <div className="flex items-start justify-between gap-2">
                                <CardTitle className="text-sm flex items-center gap-2">
                                  <span>{meta.icon}</span>
                                  {page.title}
                                </CardTitle>
                                <Badge variant="outline" className={meta.color + " text-xs shrink-0"}>
                                  {meta.label}
                                </Badge>
                              </div>
                            </CardHeader>
                            <CardContent>
                              <pre className="text-xs text-muted-foreground whitespace-pre-wrap line-clamp-3 font-mono">
                                {page.content.slice(0, 200)}
                                {page.content.length > 200 && "..."}
                              </pre>
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                  </div>
                ))}
            </div>
          )}
        </TabsContent>

        {/* 代际进化 Tab */}
        <TabsContent value="generations" className="space-y-4">
          {/* 代际进化控制面板 */}
          <Card className="bg-gradient-to-br from-indigo-500/5 to-purple-500/5 border-indigo-500/20">
            <CardContent className="pt-6">
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div className="flex-1 min-w-[200px]">
                  <h3 className="font-semibold mb-1">🧬 代际进化系统</h3>
                  <p className="text-sm text-muted-foreground">
                    基于 llm-wiki 记忆和种群遗传算法，让扫地僧代代进化。后代通过读取前人经验避免重复踩坑。
                  </p>
                  <div className="mt-3 flex items-center gap-3">
                    <label className="text-sm">自动执行代数:</label>
                    <input
                      type="number"
                      min="0"
                      max="10"
                      value={autoGenerations}
                      onChange={(e) => setAutoGenerations(Number(e.target.value))}
                      className="w-20 px-2 py-1 border rounded text-sm"
                    />
                    <span className="text-xs text-muted-foreground">
                      (0=不自动执行, 需手动审批)
                    </span>
                  </div>
                </div>
                <Button
                  onClick={handlePlanNextGen}
                  disabled={planning || generations.length === 0}
                  variant="outline"
                  size="sm"
                >
                  {planning ? "规划中..." : "🧬 规划下一代"}
                </Button>
              </div>
            </CardContent>
          </Card>

          {generations.length === 0 ? (
            <Card>
              <CardContent className="py-16 text-center text-muted-foreground">
                <div className="text-5xl mb-4">🧬</div>
                <p>尚无代际记录。先运行初代扫地僧，然后可以规划二代目。</p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {generations.map((gen) => (
                <Card key={gen.id} className={gen.status === "planned" ? "border-amber-500/30" : ""}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      <CardTitle className="text-base flex items-center gap-2">
                        <span className="text-xl">
                          {gen.generation === 1 ? "🌱" : "🧬"}
                        </span>
                        第 {gen.generation} 代
                        {gen.generation === 1 && <Badge variant="secondary" className="text-xs">初代</Badge>}
                      </CardTitle>
                      <GenStatusBadge status={gen.status} />
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {/* 父本信息 */}
                    {gen.parentFormula && (
                      <div className="text-xs bg-muted/50 rounded p-2">
                        <span className="text-muted-foreground">父本 (第{gen.parentGen}代): </span>
                        <code className="font-mono">{gen.parentFormula.slice(0, 80)}</code>
                        {gen.parentError && <span className="text-muted-foreground"> (RMSE: {gen.parentError.toFixed(4)})</span>}
                      </div>
                    )}

                    {/* 结果 */}
                    {gen.bestFormula && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                        <div>
                          <div className="text-xs text-muted-foreground">最佳公式</div>
                          <code className="font-mono text-xs">{gen.bestFormula.slice(0, 50)}...</code>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">RMSE</div>
                          <div className="font-mono font-bold">{gen.bestError?.toFixed(4) ?? "—"}</div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">综合评分</div>
                          <div className="font-mono font-bold">{gen.bestScore?.toFixed(2) ?? "—"}</div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">经验记录</div>
                          <div className="font-mono font-bold">{gen.totalExperiences}</div>
                        </div>
                      </div>
                    )}

                    {/* 计划信息 (planned 状态) */}
                    {gen.status === "planned" && gen.plan && (
                      <div className="text-xs space-y-1 bg-amber-500/5 border border-amber-500/20 rounded p-2">
                        <div className="font-semibold">📋 探索计划:</div>
                        {gen.plan.focusAreas && (
                          <div>探索重点: {gen.plan.focusAreas.join(", ")}</div>
                        )}
                        {gen.plan.seedGenes && gen.plan.seedGenes.length > 0 && (
                          <div>种子公式 ({gen.plan.seedGenes.length}): {gen.plan.seedGenes.map((g: any) => g.formula.slice(0, 30)).join("; ")}...</div>
                        )}
                        {gen.plan.avoidPitfalls && gen.plan.avoidPitfalls.length > 0 && (
                          <div>避坑指南 ({gen.plan.avoidPitfalls.length}条)</div>
                        )}
                      </div>
                    )}

                    {/* 操作按钮 */}
                    <div className="flex items-center gap-2 flex-wrap">
                      {gen.status === "planned" && (
                        <>
                          <Button
                            size="sm"
                            onClick={() => handleExecuteGen(gen.id, gen.generation)}
                            disabled={executingGen === gen.id}
                          >
                            {executingGen === gen.id ? "执行中..." : "▶ 执行此代"}
                          </Button>
                          <label className="flex items-center gap-1 text-xs cursor-pointer">
                            <input
                              type="checkbox"
                              checked={gen.autoExecute ?? false}
                              onChange={() => handleToggleAutoExecute(gen.id, gen.autoExecute ?? false)}
                            />
                            自动执行
                          </label>
                          <Button size="sm" variant="outline" onClick={() => handleViewGenDetail(gen.id)}>
                            查看计划
                          </Button>
                        </>
                      )}
                      {gen.status === "completed" && gen.totalExperiences > 0 && (
                        <Button size="sm" variant="outline" onClick={() => handleViewGenDetail(gen.id)}>
                          查看经验记录
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* 经验库 Tab */}
        <TabsContent value="experiences" className="space-y-4">
          {experiences.length === 0 ? (
            <Card>
              <CardContent className="py-16 text-center text-muted-foreground">
                <div className="text-5xl mb-4">🧠</div>
                <p>经验库为空。扫地僧运行后，成功/失败/洞察/避坑经验将记录到这里。</p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {experiences.map((exp) => {
                const meta = EXP_TYPE_META[exp.type] || EXP_TYPE_META.insight;
                return (
                  <Card key={exp.id}>
                    <CardHeader className="pb-2">
                      <div className="flex items-start justify-between gap-2">
                        <CardTitle className="text-sm flex items-center gap-2">
                          <span>{meta.icon}</span>
                          {exp.title}
                        </CardTitle>
                        <Badge variant="outline" className={meta.color + " text-xs shrink-0"}>
                          {meta.label}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <pre className="text-xs text-muted-foreground whitespace-pre-wrap font-mono max-h-40 overflow-y-auto bg-muted/30 p-2 rounded">
                        {exp.content.slice(0, 500)}
                        {exp.content.length > 500 && "..."}
                      </pre>
                      {exp.formula && (
                        <div className="mt-2 text-xs">
                          <span className="text-muted-foreground">公式: </span>
                          <code className="font-mono">{exp.formula}</code>
                          {exp.error !== undefined && (
                            <span className="text-muted-foreground ml-2">(RMSE: {exp.error.toFixed(4)})</span>
                          )}
                        </div>
                      )}
                      <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                        <span>重要性: {(exp.importance * 100).toFixed(0)}%</span>
                        <span>·</span>
                        <span>{new Date(exp.createdAt).toLocaleString()}</span>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </TabsContent>

        {/* 运行历史 Tab */}
        <TabsContent value="runs">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">运行历史</CardTitle>
            </CardHeader>
            <CardContent>
              {runs.length === 0 ? (
                <div className="py-8 text-center text-muted-foreground">暂无运行记录</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left">
                        <th className="py-2 px-2">Round</th>
                        <th className="py-2 px-2">状态</th>
                        <th className="py-2 px-2">最佳公式</th>
                        <th className="py-2 px-2 text-right">误差</th>
                        <th className="py-2 px-2">开始时间</th>
                        <th className="py-2 px-2">步骤数</th>
                      </tr>
                    </thead>
                    <tbody>
                      {runs.map((r) => (
                        <tr key={r.id} className="border-b hover:bg-muted/30">
                          <td className="py-2 px-2 font-bold">{r.round}</td>
                          <td className="py-2 px-2"><RunStatusBadge status={r.status} /></td>
                          <td className="py-2 px-2 font-mono text-xs max-w-xs truncate" title={r.bestFormula || ""}>
                            {r.bestFormula || "—"}
                          </td>
                          <td className="py-2 px-2 text-right font-mono">
                            {r.bestError !== null ? r.bestError.toFixed(4) : "—"}
                          </td>
                          <td className="py-2 px-2 text-xs text-muted-foreground">
                            {new Date(r.startedAt).toLocaleString()}
                          </td>
                          <td className="py-2 px-2 text-center">{r.steps.length}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Wiki 详情弹窗 */}
      {selectedPage && (
        <div
          className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedPage(null)}
        >
          <Card className="max-w-3xl w-full max-h-[80vh] overflow-hidden flex flex-col" onClick={(e) => e.stopPropagation()}>
            <CardHeader>
              <div className="flex items-start justify-between gap-2">
                <div>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <span>{CATEGORY_META[selectedPage.category]?.icon}</span>
                    {selectedPage.title}
                  </CardTitle>
                  <div className="flex gap-2 mt-2">
                    <Badge variant="outline" className={CATEGORY_META[selectedPage.category]?.color + " text-xs"}>
                      {CATEGORY_META[selectedPage.category]?.label}
                    </Badge>
                    <Badge variant="secondary" className="text-xs">Round {selectedPage.round}</Badge>
                  </div>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setSelectedPage(null)}>✕</Button>
              </div>
            </CardHeader>
            <CardContent className="overflow-y-auto">
              <pre className="text-sm whitespace-pre-wrap font-mono leading-relaxed">{selectedPage.content}</pre>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

function RunStatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; className: string }> = {
    pending: { label: "等待中", className: "bg-gray-500/10 text-gray-600 border-gray-500/20" },
    exploring: { label: "数据探索", className: "bg-blue-500/10 text-blue-600 border-blue-500/20" },
    researching: { label: "深度调研", className: "bg-purple-500/10 text-purple-600 border-purple-500/20" },
    hypothesizing: { label: "提出假设", className: "bg-amber-500/10 text-amber-600 border-amber-500/20" },
    regressing: { label: "符号回归", className: "bg-cyan-500/10 text-cyan-600 border-cyan-500/20" },
    evaluating: { label: "评估中", className: "bg-green-500/10 text-green-600 border-green-500/20" },
    completed: { label: "已完成", className: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" },
    failed: { label: "失败", className: "bg-red-500/10 text-red-600 border-red-500/20" },
  };
  const m = map[status] || { label: status, className: "" };
  return <Badge variant="outline" className={m.className + " text-xs"}>{m.label}</Badge>;
}

function StepStatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; className: string }> = {
    pending: { label: "⏳", className: "" },
    running: { label: "⚡ 运行中", className: "bg-blue-500/10 text-blue-600 border-blue-500/20" },
    completed: { label: "✓ 完成", className: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" },
    failed: { label: "✗ 失败", className: "bg-red-500/10 text-red-600 border-red-500/20" },
  };
  const m = map[status] || { label: status, className: "" };
  return <Badge variant="outline" className={m.className + " text-xs"}>{m.label}</Badge>;
}

function GenStatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; className: string }> = {
    planned: { label: "📋 已规划", className: "bg-amber-500/10 text-amber-600 border-amber-500/20" },
    approved: { label: "✓ 已批准", className: "bg-blue-500/10 text-blue-600 border-blue-500/20" },
    running: { label: "⚡ 运行中", className: "bg-cyan-500/10 text-cyan-600 border-cyan-500/20" },
    completed: { label: "✓ 已完成", className: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" },
    failed: { label: "✗ 失败", className: "bg-red-500/10 text-red-600 border-red-500/20" },
    skipped: { label: "⏭ 已跳过", className: "bg-gray-500/10 text-gray-600 border-gray-500/20" },
  };
  const m = map[status] || { label: status, className: "" };
  return <Badge variant="outline" className={m.className + " text-xs"}>{m.label}</Badge>;
}

const EXP_TYPE_META: Record<string, { label: string; icon: string; color: string }> = {
  success: { label: "成功", icon: "✅", color: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" },
  failure: { label: "失败", icon: "❌", color: "bg-red-500/10 text-red-600 border-red-500/20" },
  insight: { label: "洞察", icon: "💡", color: "bg-amber-500/10 text-amber-600 border-amber-500/20" },
  pitfall_avoidance: { label: "避坑", icon: "⚠️", color: "bg-orange-500/10 text-orange-600 border-orange-500/20" },
  technique: { label: "技巧", icon: "🔧", color: "bg-blue-500/10 text-blue-600 border-blue-500/20" },
  contradiction: { label: "矛盾", icon: "⚡", color: "bg-purple-500/10 text-purple-600 border-purple-500/20" },
};
