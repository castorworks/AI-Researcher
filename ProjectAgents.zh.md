### 项目智能体总览

| 代理名称 | 文件 | 工厂函数 | 作用 | 使用位置 |
|---|---|---|---|---|
| 准备代理（Prepare Agent） | `research_agent/inno/agents/inno_agent/prepare_agent.py` | `get_prepare_agent` | 选择/克隆 5–8 个相关的参考仓库，浏览代码树与 README，输出所选仓库/路径/论文。 | `research_agent/run_infer_plan.py`（`InnoFlow.__init__`），`research_agent/run_infer_idea.py`（`InnoFlow.__init__`） |
| 编码规划代理（Coding Plan Agent） | `research_agent/inno/agents/inno_agent/plan_agent.py` | `get_coding_plan_agent` | 产出可执行且详细的计划：数据集、模型（基于调研笔记）、训练、测试；必须先阅读代码库。 | `research_agent/run_infer_plan.py`，`research_agent/run_infer_idea.py` |
| 机器学习代理（Machine Learning Agent） | `research_agent/inno/agents/inno_agent/ml_agent.py` | `get_ml_agent` | 在 `/{working_dir}/project` 下自包含实现项目；创建文件/目录；改写（不直接导入）参考代码；可运行命令/脚本。 | `research_agent/run_infer_plan.py`，`research_agent/run_infer_idea.py` |
| 评审代理（Judge Agent） | `research_agent/inno/agents/inno_agent/judge_agent.py` | `get_judge_agent` | 按创新点与调研笔记审计 ML 实现；可移交至代码审查代理；通过 `case_resolved` 返回最终建议。 | `research_agent/run_infer_plan.py`，`research_agent/run_infer_idea.py` |
| 代码审查代理（Code Review Agent） | `research_agent/inno/agents/inno_agent/judge_agent.py`（内部辅助） | `get_code_review_agent` | 审查工作目录中的代码，浏览目录/文件后，将审查报告传回评审代理。 | 在 `get_judge_agent` 内部实例化 |
| 实验分析代理（Experiment Analysis Agent） | `research_agent/inno/agents/inno_agent/exp_analyser.py` | `get_exp_analyser_agent` | 分析实验产物（图像/视频/日志）、论文与代码；生成分析报告与后续实验计划。 | `research_agent/run_infer_plan.py`，`research_agent/run_infer_idea.py` |
| 调研代理（Survey Agent） | `research_agent/inno/agents/inno_agent/survey_agent.py` | `get_survey_agent` | 编排论文调研与代码调研；将想法拆分为原子定义；在论文→代码→合并之间循环；输出整合笔记。 | `research_agent/run_infer_plan.py`；在 `research_agent/run_infer_idea.py` 中被注释掉 |
| 论文调研代理（Paper Survey Agent） | `research_agent/inno/agents/inno_agent/survey_agent.py` | `get_paper_survey_agent` | 阅读本地论文（`/papers/`），提取形式化定义与公式；交接给代码调研代理。 | 通过 `get_survey_agent` 使用 |
| 代码调研代理（调研流程） | `research_agent/inno/agents/inno_agent/survey_agent.py` | `get_code_survey_agent` | 将理论映射到参考仓库中的代码；将发现返回给调研代理。 | 通过 `get_survey_agent` 使用 |
| 创意生成代理（Idea Generation Agent） | `research_agent/inno/agents/inno_agent/idea_agent.py` | `get_idea_agent` | 阅读论文以生成、选择或增强创新点，并给出详细方案。 | `research_agent/run_infer_idea.py` |
| 代码调研代理（创意流程） | `research_agent/inno/agents/inno_agent/idea_agent.py` | `get_code_survey_agent` | 围绕创新点分析代码库；列出文件、阅读代码、抽取实现；输出实现报告。 | `research_agent/run_infer_idea.py` |
| 演示：sales_agent | `research_agent/inno/tools/dummy_tool.py` |（内联 `Agent(...)`）| 演示/示例用代理，不用于主流程。 | 仅用于演示 |

备注：
- 代理基类/数据模型与运行时：`research_agent/inno/types.py`（`Agent`，`Response`），`research_agent/inno/core.py`（`MetaChain.run/run_async`）。
- 代理装配与缓存：`research_agent/inno/workflow/flowcache.py`（`AgentModule`，`ToolModule`）。
- 动态注册与导出：`research_agent/inno/registry.py`，`research_agent/inno/agents/__init__.py`。


