# Project 4：渗流阈值经验公式

## 数据与主要模型

`data/complete_dataset_v4.csv` 是材料中使用的 94 个晶格数据（维度 `d=2–13`，配位数 `z=3–146`，girth `g∈{3,4,6}`）。经验模型为：

\[
p_c = \frac{1}{z-1}\left[1 + \frac{12.3346 e^{-0.3368d}}{g^{0.8477}}\left(1-e^{-z/9.6171}\right)\right].
\]

随附报告给出的训练拟合结果：MAPE 8.14%、R² 0.9730、AIC −668.3。它是对现有数据的经验拟合，报告没有给出严格的第一性原理推导。

## 怎样阅读报告

1. `reports/empirical_formula_report_2026-10-04.md`：公式、训练拟合指标和分组误差。
2. `reports/final_model_decision.json`：候选模型选择记录。
3. `reports/proof_audit_2026-10-04.md`：独立审计；指出交叉验证 MAPE 为 9.84%，而先前报告的 8.83% 未复现，并审查物理解释措辞。
4. `reports/publication_readiness_assessment.md`：发表准备度审查和统计限制。
5. `reports/later_constrained_candidate_2026-10-15.md`：较晚的受约束候选公式及解释，属于未经严格推导的假设；不应与审计后的经验基线混为一谈。

## 原始研究脚本

`archive/` 保存了原研究包中的脚本、数据、图表和报告（不含 Python 缓存及外部论文 PDF）。脚本引用的原机器绝对目录已统一改为以脚本文件目录为根的 `Path(__file__).resolve().parent`，并保留原有文件名和相对布局。路径调整仅解决目录可迁移性；没有统一各脚本的依赖、输入输出约定或执行顺序，也没有运行这些脚本。

本次整理没有重新运行拟合、交叉验证或模拟，也没有独立校验 CSV 的来源文献与授权。运行前应先检查目标脚本及其依赖。
