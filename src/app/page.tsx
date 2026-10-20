"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

interface Problem {
  id: string;
  title: string;
  domain: string;
  targetVar: string;
  featureVars: string[];
  status: string;
  publisherName: string;
  submissionCount: number;
  hiddenFormula?: string;
  createdAt: string;
}

const DOMAIN_META: Record<string, { label: string; icon: string; color: string }> = {
  weather: { label: "气象科学", icon: "🌤️", color: "bg-blue-500/10 text-blue-600 border-blue-500/20" },
  turbulence: { label: "流体力学", icon: "🌊", color: "bg-cyan-500/10 text-cyan-600 border-cyan-500/20" },
  custom: { label: "自定义", icon: "🔬", color: "bg-purple-500/10 text-purple-600 border-purple-500/20" },
};

export default function HomePage() {
  const [problems, setProblems] = useState<Problem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    fetch("/api/problems")
      .then((r) => r.json())
      .then((d) => {
        setProblems(d.problems || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const domains = ["all", ...Array.from(new Set(problems.map((p) => p.domain)))];
  const filtered = filter === "all" ? problems : problems.filter((p) => p.domain === filter);

  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      {/* Hero */}
      <div className="mb-10 text-center">
        <h1 className="text-4xl font-bold tracking-tight mb-3">
          <span className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            科学公式发现竞技场
          </span>
        </h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          发布数据集与目标变量，选手提交候选公式，平台自动评估拟合误差与可解释性。
          内置强力解题 baseline <span className="font-semibold text-foreground">「扫地僧」</span>，
          综合深度调研、LLM 假设、多种符号回归方法自动发现隐含公式。
        </p>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="活跃赛题" value={problems.length} icon="🎯" />
        <StatCard
          label="总提交数"
          value={problems.reduce((s, p) => s + p.submissionCount, 0)}
          icon="📝"
        />
        <StatCard
          label="研究领域"
          value={new Set(problems.map((p) => p.domain)).size}
          icon="🔬"
        />
        <StatCard label="扫地僧" value="在线" icon="🧘" isText />
      </div>

      {/* 筛选 */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <h2 className="text-2xl font-semibold">赛题大厅</h2>
        <div className="flex gap-2 flex-wrap">
          {domains.map((d) => (
            <Button
              key={d}
              variant={filter === d ? "default" : "outline"}
              size="sm"
              onClick={() => setFilter(d)}
            >
              {d === "all" ? "全部" : DOMAIN_META[d]?.label || d}
            </Button>
          ))}
        </div>
      </div>

      {/* 赛题列表 */}
      {loading ? (
        <div className="grid md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-48 rounded-lg" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card>
          <CardContent className="py-16 text-center text-muted-foreground">
            暂无赛题，点击右上角「发布赛题」创建第一个赛题
          </CardContent>
        </Card>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {filtered.map((p) => {
            const meta = DOMAIN_META[p.domain] || DOMAIN_META.custom;
            return (
              <Link key={p.id} href={`/problems/${p.id}`}>
                <Card className="hover:shadow-lg hover:border-indigo-500/40 transition-all cursor-pointer h-full group">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between gap-2">
                      <CardTitle className="text-lg group-hover:text-indigo-500 transition-colors line-clamp-2">
                        {p.title}
                      </CardTitle>
                      <Badge variant="outline" className={meta.color + " shrink-0"}>
                        {meta.icon} {meta.label}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex flex-wrap gap-2 text-xs">
                      <Badge variant="secondary">目标: {p.targetVar}</Badge>
                      {p.featureVars.slice(0, 3).map((v) => (
                        <Badge key={v} variant="outline" className="font-mono">
                          {v}
                        </Badge>
                      ))}
                      {p.featureVars.length > 3 && (
                        <Badge variant="outline">+{p.featureVars.length - 3}</Badge>
                      )}
                    </div>
                    {p.hiddenFormula && (
                      <div className="text-xs font-mono bg-muted/50 rounded p-2 line-clamp-1 text-muted-foreground">
                        🔒 {p.hiddenFormula}
                      </div>
                    )}
                    <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
                      <span>👤 {p.publisherName}</span>
                      <span>📝 {p.submissionCount} 提交</span>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
  icon,
  isText,
}: {
  label: string;
  value: number | string;
  icon: string;
  isText?: boolean;
}) {
  return (
    <Card>
      <CardContent className="pt-5 pb-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-2xl font-bold">{value}</div>
            <div className="text-xs text-muted-foreground mt-1">{label}</div>
          </div>
          <div className="text-2xl opacity-80">{icon}</div>
        </div>
      </CardContent>
    </Card>
  );
}
