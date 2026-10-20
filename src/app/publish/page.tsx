"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";

interface DataRow {
  [key: string]: number;
}

export default function PublishPage() {
  const router = useRouter();
  const { toast } = useToast();

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [domain, setDomain] = useState("custom");
  const [targetVar, setTargetVar] = useState("");
  const [featureVarsText, setFeatureVarsText] = useState("");
  const [csvData, setCsvData] = useState("");
  const [hiddenFormula, setHiddenFormula] = useState("");
  const [errorWeight, setErrorWeight] = useState(60);
  const [publisherName, setPublisherName] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const featureVars = featureVarsText
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);

  const parseCsv = (csv: string): DataRow[] => {
    const lines = csv.trim().split("\n");
    if (lines.length < 2) return [];
    const headers = lines[0].split(",").map((h) => h.trim());
    return lines.slice(1).map((line) => {
      const values = line.split(",");
      const row: DataRow = {};
      headers.forEach((h, i) => {
        const v = parseFloat(values[i]);
        row[h] = isNaN(v) ? 0 : v;
      });
      return row;
    });
  };

  const handleQuickFill = (preset: "weather" | "turbulence") => {
    if (preset === "weather") {
      setTitle("体感温度预测");
      setDomain("weather");
      setTargetVar("apparentTemp");
      setFeatureVarsText("temperature, humidity, windSpeed, pressure");
      setDescription(`给定气象数据 (温度、湿度、风速、气压), 发现预测体感温度的公式。

## 数据说明
- temperature: 温度 (°C)
- humidity: 相对湿度 (%)
- windSpeed: 风速 (m/s)
- pressure: 气压 (hPa)
- apparentTemp: 体感温度 (°C) [目标变量]

## 评估指标
- 拟合误差 (RMSE) 权重: 60%
- 公式可解释性 权重: 40%`);
      setFeatureVarsText("temperature, humidity, windSpeed, pressure");
      // 生成示例 CSV
      const rows = [];
      for (let i = 0; i < 30; i++) {
        const T = -5 + Math.random() * 40;
        const RH = 20 + Math.random() * 80;
        const W = Math.random() * 15;
        const P = 990 + Math.random() * 40;
        const apparentT = T - 0.55 * (1 - RH / 100) * (T - 14.5) + 0.05 * W;
        rows.push(`${T.toFixed(2)},${RH.toFixed(2)},${W.toFixed(2)},${P.toFixed(2)},${apparentT.toFixed(2)}`);
      }
      setCsvData(`temperature,humidity,windSpeed,pressure,apparentTemp\n` + rows.join("\n"));
      setHiddenFormula("temperature - 0.55 * (1 - humidity/100) * (temperature - 14.5) + 0.05 * windSpeed");
    } else {
      setTitle("颗粒阻力系数湍流模型");
      setDomain("turbulence");
      setTargetVar("dragCoeff");
      setFeatureVarsText("reynolds, viscosity, density, velocity, diameter");
      setDescription(`给定流体力学实验数据, 发现预测球形颗粒阻力系数的公式。

## 数据说明
- reynolds: 雷诺数 Re
- viscosity: 动力粘度 (Pa·s)
- density: 流体密度 (kg/m³)
- velocity: 相对速度 (m/s)
- diameter: 颗粒直径 (m)
- dragCoeff: 阻力系数 Cd [目标变量]

## 评估指标
- 拟合误差 (RMSE) 权重: 70%
- 公式可解释性 权重: 30%`);
      const rows = [];
      for (let i = 0; i < 30; i++) {
        const Re = Math.pow(10, -1 + Math.random() * 6);
        const Cd = 24 / Re + 6 / (1 + Math.sqrt(Re)) + 0.4;
        rows.push(`${Re.toFixed(4)},${(1e-6 + Math.random() * 1e-5).toFixed(8)},${(990 + Math.random() * 30).toFixed(2)},${(0.01 + Math.random() * 5).toFixed(3)},${(0.001 + Math.random() * 0.01).toFixed(5)},${Cd.toFixed(4)}`);
      }
      setCsvData(`reynolds,viscosity,density,velocity,diameter,dragCoeff\n` + rows.join("\n"));
      setHiddenFormula("24/reynolds + 6/(1+sqrt(reynolds)) + 0.4");
    }
  };

  const handleSubmit = async () => {
    if (!title || !description || !targetVar || featureVars.length === 0 || !csvData) {
      toast({ title: "请填写所有必填字段", variant: "destructive" });
      return;
    }
    const samples = parseCsv(csvData);
    if (samples.length < 5) {
      toast({ title: "数据至少需要 5 行", variant: "destructive" });
      return;
    }
    // 验证变量
    const allVars = [...featureVars, targetVar];
    const missing = allVars.filter((v) => !(v in samples[0]));
    if (missing.length > 0) {
      toast({ title: `CSV 缺少列: ${missing.join(", ")}`, variant: "destructive" });
      return;
    }

    setSubmitting(true);
    try {
      const res = await fetch("/api/problems", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          description,
          domain,
          targetVar,
          featureVars,
          samples,
          hiddenFormula: hiddenFormula || undefined,
          metrics: {
            errorWeight: errorWeight / 100,
            interpWeight: (100 - errorWeight) / 100,
          },
          publisherName: publisherName || "匿名",
        }),
      });
      const data = await res.json();
      if (data.error) {
        toast({ title: "发布失败", description: data.error, variant: "destructive" });
      } else {
        toast({ title: "发布成功!", description: "赛题已创建" });
        router.push(`/problems/${data.id}`);
      }
    } catch (e) {
      toast({ title: "发布失败", description: (e as Error).message, variant: "destructive" });
    }
    setSubmitting(false);
  };

  return (
    <div className="container mx-auto max-w-4xl px-4 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">📢 发布新赛题</h1>
        <p className="text-muted-foreground">提交问题目标和数据，让选手和扫地僧来发现隐含的公式</p>
      </div>

      {/* 快速填充 */}
      <Card className="bg-muted/30">
        <CardContent className="pt-5 pb-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <span className="text-sm text-muted-foreground">⚡ 快速模板:</span>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => handleQuickFill("weather")}>
                🌤️ 天气预测
              </Button>
              <Button size="sm" variant="outline" onClick={() => handleQuickFill("turbulence")}>
                🌊 湍流模型
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">赛题信息</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>赛题标题 *</Label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="例如: 体感温度预测" />
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>领域</Label>
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
              >
                <option value="weather">🌤️ 气象科学</option>
                <option value="turbulence">🌊 流体力学</option>
                <option value="custom">🔬 自定义</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label>发布者名称</Label>
              <Input value={publisherName} onChange={(e) => setPublisherName(e.target.value)} placeholder="匿名" />
            </div>
          </div>

          <div className="space-y-2">
            <Label>赛题描述 *</Label>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="描述问题背景、数据说明、评估指标..."
              rows={6}
            />
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>目标变量 *</Label>
              <Input value={targetVar} onChange={(e) => setTargetVar(e.target.value)} placeholder="例如: apparentTemp" className="font-mono" />
            </div>
            <div className="space-y-2">
              <Label>特征变量 (逗号或空格分隔) *</Label>
              <Input value={featureVarsText} onChange={(e) => setFeatureVarsText(e.target.value)} placeholder="temperature, humidity, windSpeed" className="font-mono" />
              {featureVars.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {featureVars.map((v) => (
                    <Badge key={v} variant="outline" className="font-mono text-xs">{v}</Badge>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <Label>隐藏公式 (可选, 供发布者参考)</Label>
            <Input value={hiddenFormula} onChange={(e) => setHiddenFormula(e.target.value)} placeholder="例如: a * x + b" className="font-mono" />
            <p className="text-xs text-muted-foreground">真实公式，仅发布者可见，用于验证选手发现的公式是否接近真相</p>
          </div>

          <div className="space-y-2">
            <Label>评估权重: 误差 {errorWeight}% / 可解释性 {100 - errorWeight}%</Label>
            <input
              type="range"
              min="0"
              max="100"
              value={errorWeight}
              onChange={(e) => setErrorWeight(Number(e.target.value))}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">数据 (CSV 格式) *</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-sm text-muted-foreground">
            第一行为表头 (必须包含所有特征变量和目标变量)，后续每行一个样本。
          </p>
          <Textarea
            value={csvData}
            onChange={(e) => setCsvData(e.target.value)}
            placeholder={`temperature,humidity,windSpeed,apparentTemp\n25.0,60,3.0,24.8\n...`}
            rows={8}
            className="font-mono text-xs"
          />
          {csvData && (
            <p className="text-xs text-muted-foreground">
              解析到 {parseCsv(csvData).length} 行数据
            </p>
          )}
        </CardContent>
      </Card>

      <div className="flex gap-2 justify-end">
        <Button variant="outline" onClick={() => router.push("/")}>取消</Button>
        <Button onClick={handleSubmit} disabled={submitting} className="bg-gradient-to-r from-indigo-500 to-purple-500">
          {submitting ? "发布中..." : "🚀 发布赛题"}
        </Button>
      </div>
    </div>
  );
}
