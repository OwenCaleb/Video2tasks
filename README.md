<div align="center">

# 🤖 Robot Video Segmentor

**A distributed video segmentation system for robotic manipulation tasks using Vision-Language Models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

[English](README.md) | [中文文档](README_CN.md)

</div>

---

## 📖 Overview

Robot Video Segmentor provides a **client-server architecture** for analyzing robot videos and detecting task boundaries (switch points) using VLMs like Qwen3-VL.

| Component | Description |
|-----------|-------------|
| **Server** | Manages job queues, video frame extraction, and result aggregation |
| **Worker** | Runs VLM inference to detect task transitions in video windows |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Video Windowing** | Configurable video window sampling parameters |
| 🤖 **Pluggable Backends** | Support for Qwen3-VL, Remote API, or custom VLM implementations |
| 📊 **Smart Aggregation** | Automatic segment generation with weighted voting & Hanning window |
| 🔄 **Distributed Processing** | Scale horizontally with multiple workers |
| ⚙️ **YAML Config** | Simple, declarative configuration management |
| 🖥️ **Cross-Platform** | Linux/GPU recommended; Windows/CPU with dummy backend |

---

## 🏗️ Architecture

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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ly-geming/robot-video-segmentor.git
cd robot-video-segmentor

# Install with core dependencies
pip install -e .

# Or install with Qwen3-VL support (requires GPU)
pip install -e ".[qwen3vl]"
```

### Configuration

```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit with your paths and settings
vim config.yaml  # or your preferred editor
```

### Running

**Terminal 1 - Start the Server:**
```bash
rvs-server --config config.yaml
```

**Terminal 2 - Start a Worker:**
```bash
rvs-worker --config config.yaml
```

> 💡 **Tip:** You can start multiple workers to process videos in parallel!

---

## ⚙️ Configuration

See [`config.example.yaml`](config.example.yaml) for all available options:

| Section | Description |
|---------|-------------|
| `datasets` | Video dataset paths and subsets |
| `run` | Output directory configuration |
| `server` | Host, port, and queue settings |
| `worker` | VLM backend selection and model paths |
| `windowing` | Frame sampling parameters |

---

## 🔌 VLM Backends

### Dummy Backend (Default)

Lightweight backend for testing and Windows/CPU environments. Returns mock results without loading heavy models.

```yaml
worker:
  backend: dummy
```

### Qwen3-VL Backend

Full inference using Qwen3-VL-32B-Instruct (or other variants).

**Requirements:**
- 🐧 Linux with NVIDIA GPU
- 💾 24GB+ VRAM (for 32B model)
- 🔥 PyTorch with CUDA support

```yaml
worker:
  backend: qwen3vl
  model_path: /path/to/model
```

### Remote API Backend

Use an external API endpoint for inference:

```yaml
worker:
  backend: remote_api
  api_url: http://your-api-server/infer
```

<details>
<summary>📡 API Request/Response Format</summary>

**Request:**
```json
{
  "prompt": "...",
  "images_b64_png": ["...", "..."]
}
```

**Response:**
```json
{
  "transitions": [6],
  "instructions": ["Place the fork", "Place the spoon"],
  "thought": "..."
}
```

</details>

### Custom Backend

Implement the `VLMBackend` interface to add your own:

```python
from robot_video_segmentor.vlm.base import VLMBackend

class MyBackend(VLMBackend):
    def infer(self, images, prompt):
        # Your inference logic
        return {"transitions": [], "instructions": []}
```

---

## 📁 Project Structure

```
robot-video-segmentor/
├── 📂 src/robot_video_segmentor/
│   ├── config.py              # Configuration models
│   ├── prompt.py              # Prompt templates
│   ├── 📂 server/             # FastAPI server
│   │   ├── app.py
│   │   └── windowing.py
│   ├── 📂 worker/             # Worker implementation
│   │   └── runner.py
│   ├── 📂 vlm/                # VLM backends
│   │   ├── dummy.py
│   │   ├── qwen3vl.py
│   │   └── remote_api.py
│   └── 📂 cli/                # CLI entrypoints
│       ├── server.py
│       └── worker.py
├── 📄 config.example.yaml
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 README_CN.md
└── 📄 LICENSE
```

---

## 🧪 Testing

```bash
# Validate configuration
rvs-validate-config --config config.yaml

# Run tests
pytest
```

---

## 💻 Requirements

<table>
<tr>
<th>Minimum (Dummy Backend)</th>
<th>Recommended (Qwen3-VL)</th>
</tr>
<tr>
<td>

- Python 3.8+
- 4GB RAM
- Any OS

</td>
<td>

- Python 3.8+
- Linux + NVIDIA GPU
- 24GB+ VRAM
- CUDA 11.8+ / 12.x

</td>
</tr>
</table>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- VLM support via [Transformers](https://huggingface.co/docs/transformers/)
- Inspired by robotic video analysis research

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

</div>
