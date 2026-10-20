# 科学公式发现竞技场

基于 Next.js 的科学公式发现竞技场，包含默认公式发现智能体“扫地僧”。同一仓库也整理了 Project 4 的渗流阈值研究案例、数据和关键审计材料。

## 目录

- `src/`、`prisma/`、`public/`：网站源码、SQLite 数据模型和静态资源。
- `research/percolation/data/`：研究使用的 94 条数据集。
- `research/percolation/reports/`：核心结果、模型选择、复现审计和发表准备度评估。

## 本地开发

需要 Bun 和 Node.js。项目调用 `z-ai-web-dev-sdk` 提供模型、网页搜索和页面阅读能力；该 SDK 的凭据及服务配置由运行环境提供，仓库不包含凭据。

```sh
cp .env.example .env
bun install
bun run db:generate
bun run db:push
bun run dev
```

打开 `http://localhost:3000`。检查 `.env.example` 并按你的运行环境配置 SDK 所需的凭据。应用源码中启用了 `typescript.ignoreBuildErrors`，生产构建前应单独检查类型错误。

## 研究案例结论

94 个晶格数据上的经验公式及其报告指标已整理在 `research/percolation/reports/`。训练拟合 MAPE 约 8.14%、R² 约 0.973；审计报告更正了交叉验证 MAPE（9.84%，早期 8.83% 声明未能复现）。报告明确列出数据规模、子类误差和局限性。这些数值来自随附材料，**本次整理没有重新运行分析脚本或独立验证数值**。

较晚的受约束候选模型和理论解释单独存放，作为后续研究假设。材料本身指出目前没有严格推导，不能将其当作已证明的理论公式。详见 [案例说明](research/percolation/README.md)。

## 来源与范围

网站源码来自 `formula-arena-source-v3.zip`；研究材料来自 `Project 4_2026-10-20 13_13_28.zip`。研究目录只收录核心数据及关键报告，未复制缓存、临时实验脚本或原包中的外部论文 PDF。原始实验脚本含机器专用绝对路径，故未作为可直接复现的代码发布。

当前源码包没有附带许可证文件，也未在此仓库新增许可证；使用和再分发权限仍由源码权利人决定。
