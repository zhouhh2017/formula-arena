"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";

interface ProblemDetail {
  id: string;
  title: string;
  description: string;
  domain: string;
  targetVar: string;
  featureVars: string[];
  samples: Record<string, number>[];
  hiddenFormula?: string;
  metrics: { errorWeight: number; interpWeight: number };
  publisherName: string;
  status: string;
  createdAt: string;
}

interface LeaderboardEntry {
  rank: number;
  id: string;
  submitterName: string;
  formula: string;
  rmse: number;
  r2: number;
  mae: number;
  interpretability: number;
  complexity: number;
  overallScore: number;
  isMonk: boolean;
  createdAt: string;
}

interface EvalResult {
  formula: string;
  optimizedFormula: string;
  rmse: number;
  mae: number;
  r2: number;
  complexity: number;
  interpretability: number;
  overallScore: number;
  valid: boolean;
  error?: string;
}

export default function ProblemDetailPage() {
  const params = useParams<{ id: string }>();
  const problemId = params.id;
  const { toast } = useToast();

  const [problem, setProblem] = useState<ProblemDetail | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const [submitterName, setSubmitterName] = useState("");
  const [formula, setFormula] = useState("");
  const [report, setReport] = useState("");
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [monkRunning, setMonkRunning] = useState(false);

  const loadProblem = useCallback(async () => {
    const [pRes, lRes] = await Promise.all([
      fetch(`/api/problems/${problemId}`).then((r) => r.json()),
      fetch(`/api/problems/${problemId}/leaderboard`).then((r) => r.json()),
    ]);
    setProblem(pRes);
    setLeaderboard(lRes.leaderboard || []);
    setLoading(false);
  }, [problemId]);

  useEffect(() => {
    loadProblem();
  }, [loadProblem]);

  const handleEvaluate = async () => {
    if (!formula.trim()) {
      toast({ title: "请输入公式", variant: "destructive" });
      return;
    }
    setEvaluating(true);
    try {
      const res = await fetch(`/api/problems/${problemId}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ formula }),
      });
      const data = await res.json();
      setEvalResult(data.evaluation);
      if (!data.evaluation.valid) {
        toast({ title: "公式语法错误", description: data.evaluation.error, variant: "destructive" });
      } else {
        toast({ title: "评估完成", description: `RMSE: ${data.evaluation.rmse.toFixed(4)}, R²: ${data.evaluation.r2.toFixed(4)}` });
      }
    } catch (e) {
      toast({ title: "评估失败", description: (e as Error).message, variant: "destructive" });
    }
    setEvaluating(false);
  };

  const handleSubmit = async () => {
    if (!submitterName.trim() || !formula.trim()) {
      toast({ title: "请填写提交者名称和公式", variant: "destructive" });
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch(`/api/problems/${problemId}/submissions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ submitterName, formula, report }),
      });
      const data = await res.json();
      if (data.error) {
        toast({ title: "提交失败", description: data.error, variant: "destructive" });
      } else {
        toast({ title: "提交成功!", description: `综合评分: ${data.evaluation.overallScore.toFixed(2)}` });
        setSubmitterName("");
        setFormula("");
        setReport("");
        setEvalResult(null);
        await loadProblem();
      }
    } catch (e) {
      toast({ title: "提交失败", description: (e as Error).message, variant: "destructive" });
    }
    setSubmitting(false);
  };

  const handleRunMonk = async () => {
    setMonkRunning(true);
    toast({ title: "扫地僧启动中", description: "正在执行数据探索→深度调研→假设→符号回归→评估..." });
    try {
      const res = await fetch(`/api/problems/${problemId}/monk/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ maxRounds: 2 }),
      });
      const data = await res.json();
      if (data.error) {
        toast({ title: "扫地僧运行失败", description: data.error, variant: "destructive" });
      } else {
        toast({
          title: "扫地僧完成!",
          description: `最佳公式: ${data.bestFormula?.slice(0, 40)}... 误差: ${data.bestError?.toFixed(4)}`,
        });
        await loadProblem();
      }
    } catch (e) {
      toast({ title: "扫地僧运行失败", description: (e as Error).message, variant: "destructive" });
    }
    setMonkRunning(false);
  };

  if (loading) {
    return (
      <div className="container mx-auto max-w-6xl px-4 py-8 space-y-4">
        <Skeleton className="h-12 w-2/3" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (!problem) {
    return <div className="container mx-auto py-16 text-center">赛题不存在</div>;
  }

  return (
    <div className="container mx-auto max-w-6xl px-4 py-8 space-y-6">
      {/* 头部 */}
      <div>
        <Link href="/" className="text-sm text-muted-foreground hover:text-foreground mb-2 inline-block">
          ← 返回赛题大厅
        </Link>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-3xl font-bold mb-2">{problem.title}</h1>
            <div className="flex flex-wrap gap-2 text-sm">
              <Badge variant="secondary">领域: {problem.domain}</Badge>
              <Badge variant="outline">目标变量: <span className="font-mono ml-1">{problem.targetVar}</span></Badge>
              <Badge variant="outline">发布者: {problem.publisherName}</Badge>
            </div>
          </div>
          <Button onClick={handleRunMonk} disabled={monkRunning} className="bg-gradient-to-r from-indigo-500 to-purple-500">
            {monkRunning ? "🧘 扫地僧运行中..." : "🧘 启动扫地僧"}
          </Button>
        </div>
      </div>

      {/* 描述 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">赛题说明</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="whitespace-pre-wrap text-sm font-sans leading-relaxed">{problem.description}</pre>
          {problem.hiddenFormula && (
            <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-md">
              <div className="text-xs text-amber-700 dark:text-amber-400 mb-1">🔒 隐藏公式 (发布者参考)</div>
              <code className="text-sm font-mono">{problem.hiddenFormula}</code>
            </div>
          )}
          <div className="mt-4 flex gap-4 text-sm text-muted-foreground">
            <span>评估权重: 误差 {problem.metrics.errorWeight * 100}% / 可解释性 {problem.metrics.interpWeight * 100}%</span>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="submit">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="submit">📝 提交公式</TabsTrigger>
          <TabsTrigger value="leaderboard">🏆 排行榜 ({leaderboard.length})</TabsTrigger>
          <TabsTrigger value="data">📊 数据预览</TabsTrigger>
        </TabsList>

        {/* 提交 Tab */}
        <TabsContent value="submit" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">提交你的公式</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="submitter">提交者名称</Label>
                  <Input
                    id="submitter"
                    value={submitterName}
                    onChange={(e) => setSubmitterName(e.target.value)}
                    placeholder="你的名字或团队名"
                  />
                </div>
                <div className="space-y-2">
                  <Label>可用变量</Label>
                  <div className="flex flex-wrap gap-1.5 p-2 border rounded-md bg-muted/30 min-h-[36px] items-center">
                    {problem.featureVars.map((v) => (
                      <Badge key={v} variant="outline" className="font-mono cursor-pointer hover:bg-accent"
                        onClick={() => setFormula((f) => f + " " + v)}>
                        {v}
                      </Badge>
                    ))}
                    <Badge variant="secondary" className="font-mono">+ - * / ^</Badge>
                    <Badge variant="secondary" className="font-mono">sin cos exp log sqrt</Badge>
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="formula">公式表达式</Label>
                <Input
                  id="formula"
                  value={formula}
                  onChange={(e) => setFormula(e.target.value)}
                  placeholder={`例如: a * ${problem.featureVars[0]} + b * ${problem.featureVars[1] || ''} + c`}
                  className="font-mono"
                />
                <p className="text-xs text-muted-foreground">
                  支持变量、参数 (a, b, c...)、运算符 (+ - * / ^)、函数 (sin cos tan exp log sqrt abs)。参数会自动优化。
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="report">报告 (可选)</Label>
                <Textarea
                  id="report"
                  value={report}
                  onChange={(e) => setReport(e.target.value)}
                  placeholder="说明你的公式思路、推导过程、适用条件..."
                  rows={3}
                />
              </div>
              <div className="flex gap-2">
                <Button onClick={handleEvaluate} disabled={evaluating} variant="outline">
                  {evaluating ? "评估中..." : "🔍 预评估"}
                </Button>
                <Button onClick={handleSubmit} disabled={submitting}>
                  {submitting ? "提交中..." : "🚀 提交到竞技场"}
                </Button>
              </div>

              {evalResult && (
                <div className={`p-4 rounded-md border ${evalResult.valid ? "bg-green-500/5 border-green-500/20" : "bg-red-500/5 border-red-500/20"}`}>
                  <div className="font-semibold mb-2">{evalResult.valid ? "✅ 评估结果" : "❌ 评估失败"}</div>
                  {evalResult.valid ? (
                    <div className="space-y-2">
                      <div className="text-sm">
                        <span className="text-muted-foreground">优化后公式: </span>
                        <code className="font-mono bg-muted px-1.5 py-0.5 rounded">{evalResult.optimizedFormula}</code>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
                        <Metric label="RMSE" value={evalResult.rmse.toFixed(4)} />
                        <Metric label="R²" value={evalResult.r2.toFixed(4)} />
                        <Metric label="MAE" value={evalResult.mae.toFixed(4)} />
                        <Metric label="可解释性" value={`${evalResult.interpretability.toFixed(1)}/100`} />
                        <Metric label="综合分" value={evalResult.overallScore.toFixed(2)} highlight />
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-red-600">{evalResult.error}</div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* 排行榜 Tab */}
        <TabsContent value="leaderboard">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">🏆 排行榜</CardTitle>
            </CardHeader>
            <CardContent>
              {leaderboard.length === 0 ? (
                <div className="py-12 text-center text-muted-foreground">
                  暂无提交，快来成为第一个上榜的选手！
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left">
                        <th className="py-2 px-2">#</th>
                        <th className="py-2 px-2">提交者</th>
                        <th className="py-2 px-2">公式</th>
                        <th className="py-2 px-2 text-right">RMSE</th>
                        <th className="py-2 px-2 text-right">R²</th>
                        <th className="py-2 px-2 text-right">可解释性</th>
                        <th className="py-2 px-2 text-right">综合分</th>
                      </tr>
                    </thead>
                    <tbody>
                      {leaderboard.map((e) => (
                        <tr key={e.id} className="border-b hover:bg-muted/30">
                          <td className="py-2 px-2 font-bold">
                            {e.rank === 1 ? "🥇" : e.rank === 2 ? "🥈" : e.rank === 3 ? "🥉" : e.rank}
                          </td>
                          <td className="py-2 px-2">
                            <div className="flex items-center gap-1">
                              {e.submitterName}
                              {e.isMonk && <Badge className="bg-indigo-500/10 text-indigo-600 border-indigo-500/20 text-xs">扫地僧</Badge>}
                            </div>
                          </td>
                          <td className="py-2 px-2 font-mono text-xs max-w-xs truncate" title={e.formula}>
                            {e.formula}
                          </td>
                          <td className="py-2 px-2 text-right font-mono">{e.rmse?.toFixed(4)}</td>
                          <td className="py-2 px-2 text-right font-mono">{e.r2?.toFixed(4)}</td>
                          <td className="py-2 px-2 text-right font-mono">{e.interpretability?.toFixed(1)}</td>
                          <td className="py-2 px-2 text-right font-mono font-bold">{e.overallScore?.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* 数据预览 Tab */}
        <TabsContent value="data">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">📊 数据预览 (前 10 行)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="py-2 px-2 text-left">#</th>
                      {[...problem.featureVars, problem.targetVar].map((v) => (
                        <th key={v} className="py-2 px-2 text-right font-mono">
                          {v}
                          {v === problem.targetVar && <span className="text-indigo-500 ml-1">★</span>}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {problem.samples.slice(0, 10).map((s, i) => (
                      <tr key={i} className="border-b hover:bg-muted/30">
                        <td className="py-1.5 px-2 text-muted-foreground">{i + 1}</td>
                        {[...problem.featureVars, problem.targetVar].map((v) => (
                          <td key={v} className="py-1.5 px-2 text-right font-mono">
                            {s[v]?.toFixed(3)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground mt-3">共 {problem.samples.length} 个样本</p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* 扫地僧入口 */}
      <Card className="bg-gradient-to-br from-indigo-500/5 to-purple-500/5 border-indigo-500/20">
        <CardContent className="pt-6">
          <div className="flex items-start gap-4">
            <div className="text-4xl">🧘</div>
            <div className="flex-1">
              <h3 className="font-semibold mb-1">扫地僧 (Monk Baseline)</h3>
              <p className="text-sm text-muted-foreground mb-3">
                平台内置的强力解题 baseline。工作循环：数据探索 → 深度调研 → LLM 假设 → 并行符号回归 (遗传规划/SINDy/LLM引导/模板匹配) → 拟合误差评估器细粒度分析 → 迭代改进。
                所有中间结果落盘到 Wiki 记忆库。
              </p>
              <Link href={`/problems/${problemId}/monk`}>
                <Button variant="outline" size="sm">查看扫地僧工作面板 →</Button>
              </Link>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function Metric({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`p-2 rounded ${highlight ? "bg-indigo-500/10 border border-indigo-500/20" : "bg-muted/50"}`}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`font-mono font-bold ${highlight ? "text-indigo-600" : ""}`}>{value}</div>
    </div>
  );
}
