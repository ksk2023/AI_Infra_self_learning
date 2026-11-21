# AGENTS.md — 本仓库工作规则（每个会话、每次任务都必须遵守）

> 本文件是给 AI 助手（和我自己）的持久记忆。用户已明确要求：**所有修改和对话记录都要保存，并用 git 管理。**

## 1. Git 管理（强制）

- 一切修改都在本仓库（`/home/dev/infra`）内完成并纳入版本管理；
- 每次改动 = 一个 commit（一次实验 / 一个主题 / 一份记录），完成后立即 `git push origin main`；
- 远程已配置：`git@github.com:ksk2023/AI_Infra_self_learning.git`（SSH 走 `ssh.github.com:443`，密钥已就绪）；
- 模型权重（`models/`）、`__pycache__/` 不入库（`.gitignore` 已排除）。

## 2. 对话记录保存（强制）

- 重要的问答、原理讲解、结论，**当轮就**保存为文件入库，不要只留在聊天窗口里；
- 实验记录放 `experiments/experiment-NN.md`（结构：改动点 Diff / 核心代码 / 运行方式 / 实测结果 / 原理与结论 / 延伸推论 / Limitations）；
- 对话实录放 `conversation-records/YYYY-MM-DD-主题.md`，**逐字保留问答**（用户提问用【提问】引用块，助手回答用【回答】），文件头注明来源会话；
- 流程图用 Mermaid（GitHub 原生渲染）。

## 3. 仓库外的产物必须回收（强制）

- 在仓库之外产生的脚本、文档、分析（如 `/home/dev/models/`、`/home/dev/opt4090/` 等），必须复制进本仓库并 commit；
- 否则记录会丢失（已经发生过一次：Roofline 脚本和优化路线图散落在仓库外，用户找不到记录）。

## 4. 实验方法论（延续既有传统）

- **一次只改一处 → 预测输出变化 → 实测验证 → 概念闭环**；
- 性能测量必须：固定工作量（min=max_new_tokens）、预热、交替顺序、多次测量取中位数（教训见 `experiments/experiment-11.md`：不严谨的测量曾给出 x1.09/x1.34，严格重测实为 0.998x）；
- 结论要诚实标注 Limitations；推翻自己旧结论时，同步修正旧文档并注明（不悄悄改）。

## 5. 仓库结构速查

```
README.md                     总览 + 实验改动一览 + 概念速查表
AGENTS.md                     本规则文件
experiments/                  实验 1~11 详细文档 + concepts-guide + pipeline-walkthrough
conversation-records/         对话实录（逐字）
scripts/                      实验脚本（first_inference.py / exp11_kv_cache_ab.py / roofline.py）
models/                       本地权重（git 忽略）
```
