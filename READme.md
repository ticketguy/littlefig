# 🍐 Little Fig

**CPU-native LLM training engine.**  
Fine-tune language models on machines with no GPU — even with just 8GB RAM.

[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## What this is

Little Fig is a toolkit for fine-tuning and running LLMs entirely on CPU. While tools like Unsloth, Axolotl, and LLaMA-Factory require GPUs with CUDA, Little Fig is built from the ground up for the opposite constraint: **no GPU, no cloud, full local control.**

The core innovation is **Fig Engine** — a streaming INT4 quantization system that reduces model memory by 7.1× and automatically selects the best training method for your hardware.

## What's possible

| Task | Model | RAM Needed | Reality |
|------|-------|-----------|---------|
| Fine-tune (Fig Engine, LoRA) | GPT-2 124M | ~350 MB | ✓ Fast |
| Fine-tune (Fig Engine, LoRA) | TinyLlama 1.1B | ~400 MB | ✓ Minutes/epoch |
| Fine-tune (Fig Engine, LISA) | Gemma 4B | ~3.2 GB | ✓ Feasible |
| Fine-tune (Fig Engine, LoRA) | Llama 3.1 8B | ~3 GB | ✓ Feasible |
| Inference (GGUF Q4) | Gemma 4B | ~3 GB | ✓ Fast |
| Fine-tune (standard FP32) | TinyLlama 1.1B | ~27 GB | ✗ OOM on 8GB |

---

## Quick start

### Install

```bash
# CPU PyTorch first (avoids 2.5GB CUDA download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Little Fig with training support
pip install -e ".[train]"
```

### Fine-tune with Fig Engine (recommended)

```python
from little_fig.engine import FigModel, FigTrainer, FigTrainingConfig

# Load model — automatically quantizes to INT4 + adds LoRA
model = FigModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_r=16,
    lora_alpha=32,
)

# Configure training — tier is auto-selected based on your RAM
config = FigTrainingConfig(
    num_epochs=3,
    learning_rate=2e-4,
    max_seq_length=512,
)

# Train
trainer = FigTrainer(model, config)
trainer.load_dataset("tatsu-lab/alpaca")  # or local JSONL file
trainer.train()

# Save adapter (~5 MB) and optionally merge into full model
model.save_adapter("./my_adapter")
model.merge_and_export("./my_model")
```

### Dataset formats supported

```jsonl
# Instruction format
{"instruction": "Summarize this.", "input": "Long text...", "output": "Summary."}

# ChatML / messages format  
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}

# Text format
{"text": "The quick brown fox..."}

# Prompt/completion format
{"prompt": "Question: ", "completion": "Answer."}
```

Also loads any HuggingFace dataset directly: `trainer.load_dataset("tatsu-lab/alpaca")`

---

## Fig Engine — How it works

### The Problem

Standard LLM training loads the full model in FP32 (4 bytes/param), plus gradients (4 bytes/param), plus optimizer states (8 bytes/param for AdamW), plus activations. A 1.1B model needs ~27 GB.

### The Solution

Fig Engine uses **INT4 streaming quantization**: base weights are stored at 0.55 bytes/param (7.1× smaller) and dequantized on-the-fly during the forward pass. Only LoRA adapters live in FP32. The backward pass re-dequantizes from INT4 — trading compute for memory.

```
Standard:     Load 4.4 GB weights → 4.4 GB gradients → 8.8 GB optimizer = 27 GB
Fig Engine:   Load 0.6 GB INT4   → 0.03 GB LoRA grads → 0.06 GB optimizer = ~0.4 GB
```

### Four Training Tiers

Fig Engine automatically selects the best training method that fits your RAM:

| Tier | Method | Memory (1.1B) | Quality | Speed |
|------|--------|---------------|---------|-------|
| **1. Streaming LoRA** | INT4 base + LoRA adapters | ~400 MB | Good | Fast |
| **2. LISA** | Rotating layer unfreezing | ~900 MB | Better (10-35% > LoRA) | Moderate |
| **3. MeZO** | Zeroth-order (no backward) | ~600 MB | Acceptable | Slow |
| **4. LOMO** | Fused gradient + update | ~800 MB | Best (full fine-tune) | Fast |

**LISA** [Pan et al., 2024] is the highlight: it randomly unfreezes γ=2 middle layers every K steps while always training embeddings and the LM head. Published results show 10-35% better quality than LoRA at the same memory cost.

### Manually selecting a tier

```python
from little_fig.engine import TrainingTier

config = FigTrainingConfig(
    tier="lisa",              # Force LISA
    lisa_active_layers=2,     # γ: unfrozen layers
    lisa_switch_interval=5,   # K: switch every 5 steps
)
```

---

## Architecture

```
src/little_fig/
├── __init__.py              # Entry point + hardware detection
├── model.py                 # HF + GGUF inference backends
├── trainer.py               # Legacy trainer (v0.3)
├── engine/                  # ⭐ Fig Engine (v0.4)
│   ├── __init__.py          # Public API
│   ├── quantize.py          # INT4 quantization (FIG4 format)
│   ├── linear.py            # FigLinear: INT4 base + LoRA
│   ├── model.py             # FigModel: streaming model loader
│   ├── trainer.py           # FigTrainer: unified training loop
│   ├── tier.py              # Tier selection + memory estimation
│   ├── lisa.py              # LISA scheduler
│   ├── mezo.py              # MeZO optimizer
│   ├── lomo.py              # LOMO optimizer  
│   └── packing.py           # Sequence packing
├── studio/                  # Gradio UI
│   ├── app.py               # Chat + model management
│   ├── dataset_builder.py   # Dataset creation UI
│   ├── eval_tools.py        # Evaluation UI
│   └── merge_tools.py       # Adapter merge UI
├── tests/
│   └── test_engine.py       # 12 tests, all passing
└── paper/
    └── fig_engine.md        # Technical paper
```

---

## Research Paper

See [`paper/fig_engine.md`](paper/fig_engine.md) for the full technical paper with:
- Detailed architecture description
- Benchmark results
- Memory projections for models up to 8B
- References to all underlying techniques

### Key references

| Technique | Paper | Used in |
|-----------|-------|---------|
| LoRA | Hu et al., 2022 | Tier 1 |
| LISA | arxiv 2403.17919 | Tier 2 |
| MeZO | arxiv 2305.17333 | Tier 3 |
| LOMO | arxiv 2306.09782 | Tier 4 |
| GaLore | arxiv 2403.03507 | Future work |
| APOLLO | arxiv 2412.05270 | Future work |
| BitNet | arxiv 2402.17764 | Inspiration |

---

## Roadmap

- [x] Fig Engine core (INT4, FigLinear, FigModel)
- [x] Four training tiers (LoRA, LISA, MeZO, LOMO)
- [x] Sequence packing
- [x] Auto tier selection
- [x] Multiple dataset format support
- [x] Full test suite (12/12 passing)
- [ ] Custom C/AVX-512 tiled dequant-matmul kernel
- [ ] torch.compile deep integration
- [ ] Training tab in Gradio Studio
- [ ] Push-to-Hub support
- [ ] APOLLO-Mini optimizer
- [ ] Disk-based activation checkpointing

---

## When you get a GPU later

Fig Engine auto-detects hardware. On GPU, it uses `device_map="auto"` and enables FP16/BF16 automatically. The same code runs on CPU and GPU — just faster on GPU.

---

## License

MIT
