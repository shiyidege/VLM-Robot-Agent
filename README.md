# 🤖 VLM Robot Agent — 基于视觉语言模型的机器人操作 Agent

> **在 Google Colab 免费 T4 GPU 上可运行的机器人 VLM Agent 项目**

---

## 项目概述

本项目实现一个两阶段 **VLM Agent**，输入机器人摄像头图像和自然语言指令，输出结构化操作动作。整体设计针对 Colab 免费 T4 GPU（15GB 显存）优化，无需昂贵算力。

```
输入: [图像] + [自然语言指令]
        ↓
Stage 1: VLM 场景感知
  → 识别物体 / 位置 / 状态
        ↓
Stage 2: 动作规划
  → 输出 JSON 动作计划
        ↓
输出: {"action": "pick_and_place", "target": "red cup", "confidence": 0.87}
```

---

## 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| **VLM 骨干** | InstructBLIP-Vicuna-7B (4bit 量化) | 4bit 量化后约 6GB VRAM，T4 可运行 |
| **开源数据集** | BridgeData V2 (Open X-Embodiment) | HuggingFace 开放下载，WidowX 机械臂厨房操作数据 |
| **框架** | HuggingFace Transformers + BitsAndBytes | 量化推理 |
| **动作增强** | Chain-of-Thought 多步推理 | 提升复杂任务规划能力 |

---

## GPU 资源需求

| 阶段 | VRAM 需求 | Colab 免费 T4 |
|------|-----------|---------------|
| 模型加载（4bit） | ~6 GB | ✅ 可运行 |
| 推理 (batch=1) | ~8 GB | ✅ 可运行 |
| LoRA 微调（进阶） | ~14 GB | ✅ 勉强可运行 |
| 全量微调 | 25 GB+ | ❌ 需 A100 |

---

## 数据集：BridgeData V2

- **来源**：Open X-Embodiment 联合数据集（Google / Stanford / Berkeley 等联合开源）
- **规模**：60,000+ 条机器人操作轨迹
- **机器人**：WidowX 250 机械臂
- **场景**：厨房桌面操作（抓取、放置、打开抽屉等）
- **格式**：RLDS (TensorFlow Datasets)，HuggingFace 可 streaming 加载
- **HuggingFace 页面**：`jxu124/OpenX-Embodiment`（bridge 子集）

---

## 快速开始

### 1. 打开 Colab Notebook

将 `vlm_robot_agent_colab.ipynb` 上传到 Google Colab，或直接在 Colab 中打开。

### 2. 设置 GPU

```
运行时 → 更改运行时类型 → GPU → T4
```

### 3. 按顺序执行 Cell

```
Cell 1: 检查 GPU 环境
Cell 2: 安装依赖（约 3-5 分钟）
Cell 3: 加载 InstructBLIP 模型（约 5-10 分钟，下载 ~4GB）
Cell 4: 初始化 VLMRobotAgent
Cell 5: 加载 BridgeData V2 样本
Cell 6: 运行推理评估
Cell 7: 可视化结果
Cell 8: CoT 推理实验
```

---

## Agent 架构

### 基础 Agent：`VLMRobotAgent`

```python
# 两阶段推理
result = agent.act(image, instruction="Pick up the red cup")
# → {"action": "grasp", "target": "red cup", "direction": "left", "confidence": 0.82}
```

**动作空间（8 类）**：
- `move_to` - 移动到目标位置
- `grasp` - 抓取物体
- `release` - 释放物体
- `push` - 推动物体
- `pick_and_place` - 抓取并放置
- `open` - 打开容器/抽屉
- `close` - 关闭容器/抽屉
- `wait` - 等待

### 增强 Agent：`CoTRobotAgent`

```python
# Chain-of-Thought 多步推理
result = cot_agent.act_with_cot(image, instruction="Open the drawer and get the orange object")
# 输出包含逐步思考过程 + 最终动作
```

---

## 项目结构

```
vlm_robot_agent_colab.ipynb   # 主 Notebook（在 Colab 运行）
README.md                      # 本文档
```

---

## 进阶拓展方向

### 方向一：LoRA 微调（Colab Pro A100）
用 BridgeData V2 的 (image, instruction, action) 三元组对 VLM 做监督微调：
```bash
# 参考 OpenVLA LoRA 微调脚本
python vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/bridge_data \
  --lora_rank 32
```

### 方向二：替换骨干模型
| 模型 | 参数量 | VRAM (4bit) | 预期效果 |
|------|--------|-------------|----------|
| InstructBLIP-Vicuna-7B | 7B | ~6GB | 基线 |
| LLaVA-1.6-Mistral-7B | 7B | ~6GB | 更好的指令跟随 |
| Qwen-VL-Chat | 7B | ~6GB | 中文友好 |
| OpenVLA-7B (量化) | 7B | ~6GB | 专为机器人设计 |

### 方向三：LIBERO 仿真集成
```python
# 集成 LIBERO 仿真环境，测试闭环控制
from libero.libero import benchmark
env = benchmark.get_benchmark_dict()["libero_spatial"]
# Agent 在仿真中实际执行动作
```

---

## 参考资料

- [OpenVLA 论文](https://arxiv.org/abs/2406.09246) - Open-Source Vision-Language-Action Model
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) - 开源机器人操作数据集
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) - 多机器人开源数据集合集
- [InstructBLIP](https://github.com/salesforce/LAVIS) - Salesforce 开源 VLM
- [LIBERO Benchmark](https://libero-project.github.io/) - 机器人学习仿真基准

---

## License

本项目代码采用 MIT License。使用的预训练模型和数据集请遵守各自的开源协议。
- InstructBLIP: BSD-3-Clause
- BridgeData V2: CC BY 4.0
- Open X-Embodiment: Apache 2.0
