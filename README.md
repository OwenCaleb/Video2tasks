# Robot Video Segmentor

A distributed video segmentation system for robotic manipulation tasks using Vision-Language Models (VLMs).

## Overview

This project provides a client-server architecture for analyzing robot videos and detecting task boundaries (switch points) using VLMs like Qwen3-VL. The system:

- **Server**: Manages job queues, video frame extraction, and result aggregation
- **Worker**: Runs VLM inference to detect task transitions in video windows

## Features

- 🎥 Video window sampling with configurable parameters
- 🤖 Pluggable VLM backends (Qwen3-VL, or custom implementations)
- 📊 Automatic segment generation from VLM outputs
- 🔄 Distributed processing support (multiple workers)
- ⚙️ YAML-based configuration
- 🖥️ Cross-platform support (Linux/GPU recommended, Windows/CPU with dummy backend)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/robot-video-segmentor.git
cd robot-video-segmentor

# Install with core dependencies
pip install -e .

# Or install with Qwen3-VL support (requires GPU)
pip install -e ".[qwen3vl]"
```

### Configuration

Copy the example configuration and customize:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your paths and settings
```

### Running

Start the server:
```bash
rvs-server --config config.yaml
```

Start a worker (in another terminal):
```bash
rvs-worker --config config.yaml
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Server    │────▶│  Job Queue  │◀────│   Worker    │
│  (FastAPI)  │     │             │     │  (VLM)      │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                    │
       ▼                                    ▼
┌─────────────┐                       ┌─────────────┐
│ Video Files │                       │  VLM Model  │
└─────────────┘                       └─────────────┘
```

## Configuration

See `config.example.yaml` for all available options:

- **datasets**: Video dataset paths and subsets
- **run**: Output directory configuration
- **server**: Host, port, and queue settings
- **worker**: VLM backend selection and model paths
- **windowing**: Frame sampling parameters

## VLM Backends

### Dummy Backend (Default)
Lightweight backend for testing and Windows/CPU environments. Returns mock results without loading heavy models.

### Qwen3-VL Backend
Full inference using Qwen3-VL-32B-Instruct (or other Qwen3-VL variants). Requires:
- Linux with NVIDIA GPU
- 24GB+ VRAM recommended for 32B model
- PyTorch with CUDA support

### Custom Backend
Implement the `VLMBackend` interface to add your own VLM:

```python
from robot_video_segmentor.vlm.base import VLMBackend

class MyBackend(VLMBackend):
    def infer(self, images, prompt):
        # Your inference logic
        return {"transitions": [], "instructions": []}
```

## Development

### Project Structure

```
robot-video-segmentor/
├── src/robot_video_segmentor/
│   ├── __init__.py
│   ├── config.py          # Configuration models
│   ├── server/            # FastAPI server
│   │   ├── app.py
│   │   └── windowing.py
│   ├── worker/            # Worker implementation
│   │   ├── runner.py
│   │   └── backends/
│   │       ├── dummy.py
│   │       └── qwen3vl.py
│   └── cli/               # CLI entrypoints
│       ├── server.py
│       └── worker.py
├── config.example.yaml
├── pyproject.toml
├── README.md
└── LICENSE
```

### Testing

```bash
# Validate configuration
python -m robot_video_segmentor.validate_config --config config.yaml

# Run tests
pytest
```

## Requirements

### Minimum (Dummy Backend)
- Python 3.8+
- 4GB RAM
- Any OS (Windows/Linux/macOS)

### Recommended (Qwen3-VL Backend)
- Python 3.8+
- Linux with NVIDIA GPU
- 24GB+ VRAM
- CUDA 11.8+ / 12.x

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- VLM support via [Transformers](https://huggingface.co/docs/transformers/)
- Inspired by robotic video analysis research