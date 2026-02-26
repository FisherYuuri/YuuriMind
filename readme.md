# YuuriMind

一个从 0 开始学习并实现大模型（LLM）的实践仓库。

目标不是只“会调用 API”，而是完整走通并理解 LLM 的核心链路：
模型结构 -> 数据集处理 -> 预训练 -> 微调训练 -> 强化学习 -> 推理部署 -> Agent 系统。

## 项目目标

- 从底层代码理解 Transformer/LLM 的关键结构与训练机制。
- 构建可运行、可复现、可扩展的训练与部署流程。
- 将模型能力通过 Agent 形式落地到真实任务。

## 学习与实现路线

1. 模型结构（Model Architecture）
- Token Embedding、位置编码（RoPE/YARN）
- Attention 变体（MHA/GQA）
- FFN 与 MoE
- RMSNorm、残差连接、KV Cache、Causal Mask

2. 数据集（Dataset Pipeline）
- 原始语料清洗、去重、分片
- 分词与 Tokenization
- 训练样本构建（SFT / Preference / RL）

3. 预训练（Pretraining）
- Causal Language Modeling 目标
- 训练循环、日志、Checkpoint
- 学习率策略、梯度累计、混合精度

4. 微调训练（Fine-tuning）
- SFT（监督微调）
- 指令数据组织与模板化
- 全量微调与参数高效微调（后续可扩展 LoRA）

5. 强化学习（RL for LLM）
- 奖励建模（RM）
- PPO/DPO 等对齐训练范式
- 偏好数据构建与离线评估

6. 部署（Deployment）
- 本地推理与服务化部署
- GGUF 导出与轻量部署
- vLLM 高吞吐推理

7. Agent
- 工具调用（Tool Use）
- 规划与记忆
- 任务编排与评测

## 当前仓库结构

```text
YuuriMind/
|- base/
|  |- model/        # 模型结构实现（block, ffn, gqa, moe, norm, yarn 等）
|  |- dataset/      # 数据处理与数据集构建（待完善）
|  |- train/        # 预训练/微调/强化学习训练逻辑（待完善）
|- deploy/
|  |- gguf/         # GGUF 相关导出与部署（待完善）
|  |- vllm/         # vLLM 推理部署（待完善）
|- agent/           # Agent 系统实现（待完善）
|- readme.md
```

## 阶段性里程碑

- [x] 搭建项目主目录与模型子模块框架
- [x] 初步实现部分核心模型组件（位于 `base/model/`）
- [ ] 完成可训练的最小预训练闭环
- [ ] 完成 SFT 微调与基础评测
- [ ] 完成对齐训练（RM + PPO/DPO）
- [ ] 完成 GGUF/vLLM 部署链路
- [ ] 完成可用的 Agent Demo

## 推荐使用方式

1. 先读 `base/model/`，理解模型前向计算和关键模块。
2. 补齐 `base/dataset/` 与 `base/train/`，打通最小训练闭环。
3. 在 `deploy/` 目录实现导出与推理服务。
4. 在 `agent/` 中将模型能力封装为可执行任务流。

## 未来计划

- 增加完整实验配置与复现实验脚本
- 增加训练/评测指标看板
- 增加多任务 Agent benchmark
- 增加中英文教程与注释文档

## 免责声明

本仓库用于学习与研究目的，代码与流程会持续演进，不保证当前阶段具备生产可用性。
