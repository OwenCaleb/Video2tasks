<div align="center">

# 🤖 Robot Video Segmentor

**基于视觉语言模型的机器人操作视频分布式切分系统**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

[English](README.md) | [中文文档](README_CN.md)

</div>

---

## 📖 概览

### 🎯 解决什么问题？

训练 **VLA（Vision-Language-Action）模型**（如 [π₀ (pi-zero)](https://www.physicalintelligence.company/blog/pi0)）时，你需要的是**单任务视频片段** —— 每个视频只包含一个任务。然而，真实的机器人演示视频往往包含**多个连续任务**：

```
原始视频: [拿起叉子] → [放下叉子] → [拿起勺子] → [放下勺子]
                ↓ Robot Video Segmentor ↓
输出:     segment_001.mp4   segment_002.mp4   segment_003.mp4   segment_004.mp4
          "拿起叉子"         "放下叉子"         "拿起勺子"         "放下勺子"
```

**Robot Video Segmentor 自动检测任务边界，将多任务视频切分成干净的单任务片段，直接用于 VLA 训练。**

### 🔧 工作原理

本工具采用**分布式 Client-Server 架构**，使用视觉语言模型（如 Qwen3-VL）分析视频帧，智能检测任务切换点。

| 组件 | 描述 |
|------|------|
| **Server** | 读取视频、分窗抽帧、管理任务队列并聚合结果 |
| **Worker** | 调用 VLM 推理，输出切换点与指令 |

---

## 📊 输出示例

### VLM 推理过程

VLM 会分析每个视频窗口，并提供详细的任务切换推理：

<details>
<summary>🔍 点击查看 VLM 推理过程</summary>

```json
{
  "thought": "帧 0-2: 人站立，双手张开，戴着手套，面向房间。尚无物体交互。
              帧 3: 人伸手去拿沙发上的白色手提袋。
              帧 4: 人抓住手提袋并开始提起。
              帧 5-11: 人继续操作手提袋，打开它，调整肩带，处理里面的物品。
              这是与同一物体（手提袋）的连续交互。
              帧 12: 人伸手进包里，拿出一个带黑色绑带的白色物体（可能是口罩或头戴设备）。
              从帧 12 开始，交互对象从手提袋切换到白色物体。
              因此，切换点发生在帧 12。",
  "transitions": [12],
  "instructions": ["拿起并操作手提袋", "取出并调整白色口罩"]
}
```

</details>

### 最终切分结果

一个 4501 帧的视频自动切分成 16 个单任务片段：

```json
{
  "video_id": "1765279974654",
  "nframes": 4501,
  "segments": [
    {"seg_id": 0,  "start_frame": 0,    "end_frame": 373,  "instruction": "拿起并操作手提袋"},
    {"seg_id": 1,  "start_frame": 373,  "end_frame": 542,  "instruction": "取出并调整白色口罩"},
    {"seg_id": 2,  "start_frame": 542,  "end_frame": 703,  "instruction": "打开袋子并放入物品"},
    {"seg_id": 3,  "start_frame": 703,  "end_frame": 912,  "instruction": "将第一个黑色物体放入手提袋"},
    {"seg_id": 4,  "start_frame": 912,  "end_frame": 1214, "instruction": "将第二个黑色物体放入手提袋"},
    {"seg_id": 5,  "start_frame": 1214, "end_frame": 1375, "instruction": "将白色杯子放在桌上"},
    {"seg_id": 6,  "start_frame": 1375, "end_frame": 1524, "instruction": "将杯子移到右边的桌子"},
    {"seg_id": 7,  "start_frame": 1524, "end_frame": 1784, "instruction": "将电源适配器连接到电缆"},
    {"seg_id": 8,  "start_frame": 1784, "end_frame": 2991, "instruction": "将设备插入电源插排"},
    {"seg_id": 9,  "start_frame": 2991, "end_frame": 3135, "instruction": "与茶几上的黑色物体交互"},
    {"seg_id": 10, "start_frame": 3135, "end_frame": 3238, "instruction": "调整烟灰缸"},
    {"seg_id": 11, "start_frame": 3238, "end_frame": 3359, "instruction": "与白色马克杯交互"},
    {"seg_id": 12, "start_frame": 3359, "end_frame": 3478, "instruction": "移动黑色长方形物体和杯子"},
    {"seg_id": 13, "start_frame": 3478, "end_frame": 3711, "instruction": "拿起烟灰缸"},
    {"seg_id": 14, "start_frame": 3711, "end_frame": 4095, "instruction": "将白色拖鞋从鞋架移走"},
    {"seg_id": 15, "start_frame": 4095, "end_frame": 4501, "instruction": "升起窗帘"}
  ]
}
```

> 🎯 每个片段只包含**一个任务**，并自动生成自然语言指令 —— 直接用于 VLA 训练！

---

## 💡 为什么选择这套架构？

<table>
<tr>
<td width="50%">

### 🧠 分布式架构

这不是一个死循环脚本。FastAPI 作为调度中心，Worker 只负责推理。

**你可以在一台 4090 上跑 Server，再挂 10 台机器跑 Worker 并行处理海量数据。**

这是工业级的思路。

</td>
<td width="50%">

### 🛡️ 工程化容错

- ⏱️ Inflight 超时重发
- 🔄 失败重试上限
- 📍 `.DONE` 断点续传标记

这些机制是大规模任务稳定跑完的关键。

</td>
</tr>
<tr>
<td width="50%">

### 🎯 智能切分算法

不是简单把图片丢给模型。`build_segments_via_cuts` 对多窗口结果做**加权投票**，并引入 **Hanning Window** 处理窗口边缘权重。

解决了"窗口边缘识别不稳"的经典问题。

</td>
<td width="50%">

### ✍️ 专业 Prompt 设计

`prompt_switch_detection` 明确区分：
- **True Switch**：切换到新物体
- **False Switch**：同一物体不同操作

贴合 Manipulation 数据集的痛点，**显著降低过切**。

</td>
</tr>
</table>

---

## ✨ 特性

| 特性 | 描述 |
|------|------|
| 🎥 **视频分窗** | 可配置的视频窗口抽样参数 |
| 🧩 **可插拔后端** | 支持 Qwen3-VL / 远程 API / 自定义 VLM |
| 📊 **智能聚合** | 加权投票 + Hanning Window 自动聚合分段结果 |
| 🔄 **分布式处理** | 支持多 Worker 水平扩展 |
| ⚙️ **YAML 配置** | 简洁的声明式配置管理 |
| 🧪 **跨平台** | 推荐 Linux + GPU；Windows/CPU 可用 dummy 后端 |

---

## 🏗️ 架构图

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│     Server      │────────▶│   Job Queue     │◀────────│     Worker      │
│    (FastAPI)    │         │                 │         │     (VLM)       │
│                 │         │                 │         │                 │
└────────┬────────┘         └─────────────────┘         └────────┬────────┘
         │                                                       │
         ▼                                                       ▼
┌─────────────────┐                                     ┌─────────────────┐
│   Video Files   │                                     │    VLM Model    │
└─────────────────┘                                     └─────────────────┘
```

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/ly-geming/robot-video-segmentor.git
cd robot-video-segmentor

# 安装核心依赖
pip install -e .

# 如果使用 Qwen3-VL（需要 GPU）
pip install -e ".[qwen3vl]"
```

### 配置

```bash
# 复制示例配置
cp config.example.yaml config.yaml

# 根据需要修改配置
vim config.yaml  # 或使用你喜欢的编辑器
```

### 运行

**终端 1 - 启动服务器：**
```bash
rvs-server --config config.yaml
```

**终端 2 - 启动 Worker：**
```bash
rvs-worker --config config.yaml
```

> 💡 **提示：** 可以启动多个 Worker 来并行处理视频！

---

## ⚙️ 配置说明

查看 [`config.example.yaml`](config.example.yaml) 了解所有可用选项：

| 配置项 | 描述 |
|--------|------|
| `datasets` | 视频数据集路径和子集 |
| `run` | 输出目录配置 |
| `server` | 主机、端口和队列设置 |
| `worker` | VLM 后端选择和模型路径 |
| `windowing` | 帧采样参数 |

---

## 🔌 VLM 后端

### Dummy 后端（默认）

轻量级后端，用于测试和 Windows/CPU 环境。返回模拟结果，不加载重型模型。

```yaml
worker:
  backend: dummy
```

### Qwen3-VL 后端

使用 Qwen3-VL-32B-Instruct（或其他变体）进行完整推理。

**要求：**
- 🐧 Linux + NVIDIA GPU
- 💾 24GB+ 显存（32B 模型）
- 🔥 PyTorch + CUDA 支持

```yaml
worker:
  backend: qwen3vl
  model_path: /path/to/model
```

### 远程 API 后端

如不想本地部署模型，可配置远程 API：

```yaml
worker:
  backend: remote_api
  api_url: http://your-api-server/infer
```

<details>
<summary>📡 API 请求/响应格式</summary>

**请求体：**
```json
{
  "prompt": "...",
  "images_b64_png": ["...", "..."]
}
```

**响应格式（两种皆可）：**
```json
{
  "transitions": [6],
  "instructions": ["Place the fork", "Place the spoon"],
  "thought": "..."
}
```

或者：
```json
{
  "vlm_json": {
    "transitions": [6],
    "instructions": ["Place the fork", "Place the spoon"],
    "thought": "..."
  }
}
```

</details>

### 自定义后端

实现 `VLMBackend` 接口来添加你自己的 VLM：

```python
from robot_video_segmentor.vlm.base import VLMBackend

class MyBackend(VLMBackend):
    def infer(self, images, prompt):
        # 你的推理逻辑
        return {"transitions": [], "instructions": []}
```

---

## 📁 项目结构

```
robot-video-segmentor/
├── 📂 src/robot_video_segmentor/
│   ├── config.py              # 配置模型
│   ├── prompt.py              # Prompt 模板
│   ├── 📂 server/             # FastAPI 服务端
│   │   ├── app.py
│   │   └── windowing.py
│   ├── 📂 worker/             # Worker 实现
│   │   └── runner.py
│   ├── 📂 vlm/                # VLM 后端
│   │   ├── dummy.py
│   │   ├── qwen3vl.py
│   │   └── remote_api.py
│   └── 📂 cli/                # CLI 入口
│       ├── server.py
│       └── worker.py
├── 📄 config.example.yaml
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 README_CN.md
└── 📄 LICENSE
```

---

## 🧪 测试与验证

```bash
# 验证配置文件
rvs-validate-config --config config.yaml

# 运行测试
pytest
```

---

## 💻 系统要求

<table>
<tr>
<th>最低配置（Dummy 后端）</th>
<th>推荐配置（Qwen3-VL）</th>
</tr>
<tr>
<td>

- Python 3.8+
- 4GB 内存
- 任意操作系统

</td>
<td>

- Python 3.8+
- Linux + NVIDIA GPU
- 24GB+ 显存
- CUDA 11.8+ / 12.x

</td>
</tr>
</table>

---

## 🤝 贡献

欢迎贡献代码！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 基于 [FastAPI](https://fastapi.tiangolo.com/) 构建
- VLM 支持来自 [Transformers](https://huggingface.co/docs/transformers/)
- 灵感来源于机器人视频分析研究

---

<div align="center">

**⭐ 如果觉得有用，请给个 Star！⭐**

</div>
