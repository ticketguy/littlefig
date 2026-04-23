# 🍐 Little Fig

**CPU-native LLM training engine with embedded cognitive memory.**  
Fine-tune language models on machines with no GPU — even with just 8GB RAM.  
Train models with Ember's Diaries cognitive memory built into their weights.

[![Tests](https://img.shields.io/badge/tests-21%2F21%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## What this is

Little Fig is a toolkit for fine-tuning and running LLMs entirely on CPU. While tools like Unsloth, Axolotl, and LLaMA-Factory require GPUs with CUDA, Little Fig is built from the ground up for the opposite constraint: **no GPU, no cloud, full local control.**

The system is also designed to embed [Ember's Diaries](https://github.com/ticketguy/embers-diaries) — a cognitive memory database — directly into LLM weights, making models that can store, recall, consolidate, and forget.

## Engine Stack

Little Fig is powered by four layers that work together:

| Layer | Module | What it does |
|-------|--------|-------------|
| **Quantization** | FigQuant | Adaptive codebook INT4 (9.7% less MSE than NF4, 7.4× compression) |
| **Compute** | FigKernel | torch.compile fused ops (2.95× faster RMSNorm, fused LoRA) |
| **Training** | 4 Tiers | LoRA, LISA, MeZO, LOMO — auto-selected by available RAM |
| **Optimizer** | FigPipeline | Async GPU-CPU training with CPU-resident optimizer states |
| **Memory** | Ember Integration | Cognitive memory tokens + training data for embedded memory |

## What's possible

| Task | Model | RAM Needed | Reality |
|------|-------|-----------|---------|
| Fine-tune (LoRA) | GPT-2 124M | ~350 MB | ✓ Fast |
| Fine-tune (LoRA) | TinyLlama 1.1B | ~400 MB | ✓ Minutes/epoch |
| Fine-tune (LISA) | Gemma 4B | ~3.2 GB | ✓ Feasible |
| Fine-tune (LoRA) | Llama 3.1 8B | ~3 GB | ✓ Feasible |
| Inference (GGUF Q4) | Any model | ~model size | ✓ Auto-detects arch |
| Ember memory training | Any model | +tokens overhead | ✓ 9 memory operations |

---

## Quick start

### Install

```bash
# CPU PyTorch first (avoids 2.5GB CUDA download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Little Fig
pip install -e ".[full]"
```

### Fine-tune with Fig Engine

```python
from little_fig.engine import FigModel, FigTrainer, FigTrainingConfig

# Load model — automatically quantizes with FigQuant + adds LoRA
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
trainer.load_dataset("tatsu-lab/alpaca")
trainer.train()

# Save adapter (~5 MB) and optionally merge into full model
model.save_adapter("./my_adapter")
model.merge_and_export("./my_model")
```

### Train with Ember Memory

```python
from little_fig.engine import FigModel, FigTrainer, FigTrainingConfig

# Load with Ember mode — injects memory tokens into tokenizer
model = FigModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_r=16,
    ember_mode=True,  # Adds <|mem_store|>, <|mem_recall|>, etc.
)

config = FigTrainingConfig(num_epochs=3, max_seq_length=512)
trainer = FigTrainer(model, config)

# Generate + load Ember cognitive memory training data
trainer.load_ember_dataset(n_examples=1000)
trainer.train()
```

### Launch the Studio UI

```bash
little-fig
# Opens at http://localhost:8888
# Chat, train, benchmark, manage checkpoints — all in browser
```

---

## v0.6 — What's New

### FigQuant → Primary Quantization Engine

FigQuant adaptive codebook quantization is now the **only** quantization engine. It replaces the old FIG4 format throughout the entire stack.

| Method | Cosine Sim | MSE | SNR (dB) | Bits/param |
|--------|-----------|-----|----------|------------|
| **FigQuant** | **0.9955** | **0.0090** | **20.5** | 4.31 |
| NF4 (QLoRA baseline) | ~0.995 | ~0.010 | ~20.0 | 4.00 |

### FigKernel → Wired into Model Loading

FigKernel fused operations are now automatically applied when loading a model:
- **FigRMSNorm** replaces all `RMSNorm` modules at load time (2.95× faster)
- **fig_fused_linear_lora** is used in every FigLinear forward pass
- Works on CPU (AVX-512) and GPU (CUDA) from the same source code

### FigPipeline → Wired into Trainer

When GPU is available, FigTrainer automatically uses FigPipeline:
- Optimizer states (exp_avg, exp_avg_sq) live on CPU
- Forward + backward runs on GPU
- For LoRA, gradient transfer is ~100KB/layer — negligible

### Ember Memory → Full Integration

Ember's Diaries cognitive memory system is now fully integrated:
- `ember_mode=True` in `FigModel.from_pretrained()` injects 9 memory tokens
- `FigTrainer.load_ember_dataset()` generates + loads memory training data
- Memory tokens, training data generator, and chat manager exported from engine
- Web UI has dedicated Ember page for training data generation

### Custom Web UI

A custom FastAPI + HTML/JS UI replaces the old Gradio-based studio:
- Professional dark/light theme with labeled sidebar navigation
- Chat with WebSocket streaming
- Training studio with config panel + live monitor
- Ember memory training page
- FigSpace arena for model comparison
- Benchmarks dashboard
- Checkpoints manager
- Terminal with live log streaming

### Dead Code Removed

- Gradio studio (`studio/`) — removed entirely
- Old FIG4 quantizer (`engine/quantize.py`) — replaced by FigQuant
- Legacy PEFT-based trainer (`trainer.py`) — replaced by Fig Engine trainer

---

## Architecture

```
src/little_fig/
├── __init__.py              # Entry point + hardware detection
├── model.py                 # HF + GGUF universal inference
├── engine/                  # ⭐ Fig Engine (v0.6)
│   ├── figquant.py          # FigQuant: adaptive codebook INT4 (PRIMARY)
│   ├── figkernel.py         # Fused ops: RMSNorm, SwiGLU, CE, Linear+LoRA
│   ├── figpipeline.py       # Async GPU-CPU training pipeline
│   ├── linear.py            # FigLinear: FigQuant base + LoRA (uses FigKernel)
│   ├── model.py             # FigModel: streaming loader (FigQuant + FigKernel)
│   ├── trainer.py           # FigTrainer: unified loop (FigPipeline + Ember)
│   ├── ember_integration.py # 🔥 Ember memory tokens + training data + chat
│   ├── tier.py              # Tier selection + memory estimation
│   ├── lisa.py              # LISA scheduler
│   ├── mezo.py              # MeZO optimizer
│   ├── lomo.py              # LOMO optimizer
│   ├── packing.py           # Sequence packing
│   └── gguf_loader.py       # Universal GGUF loader (auto-detects any arch)
├── web/                     # Custom FastAPI + HTML/JS UI
│   ├── server.py            # REST API + WebSocket endpoints
│   └── static/index.html    # Production frontend
└── tests/
    ├── test_engine.py       # Core tests (12/12 passing)
    └── test_v05.py          # FigQuant/FigKernel/FigPipeline tests (9/9 passing)
```

---

## Comparison with Other Tools

| Feature | Little Fig | Unsloth | LLaMA-Factory | TRL |
|---------|-----------|---------|---------------|-----|
| CPU training | ✅ Native | ❌ GPU-only | ⚠️ Slow fallback | ⚠️ Slow fallback |
| Min RAM (1.1B) | **~400 MB** | ~6 GB (GPU) | ~8 GB | ~8 GB |
| GGUF loading | ✅ Universal | ❌ | ❌ | ❌ |
| Fused ops | ✅ torch.compile | ✅ Triton | ❌ | ❌ |
| Cognitive memory | ✅ Ember | ❌ | ❌ | ❌ |
| Training tiers | 4 (auto) | 1 (LoRA) | Multiple | Multiple |
| Custom UI | ✅ FastAPI | ❌ | ✅ | ❌ |
| GPU support | ✅ FigPipeline | ✅ | ✅ | ✅ |

---

## Research Paper

See [`paper/fig_engine.md`](paper/fig_engine.md) for the full technical paper.

---

## Roadmap

- [x] FigQuant adaptive codebook quantization (primary engine)
- [x] FigKernel fused ops wired into model loading
- [x] FigPipeline wired into trainer
- [x] Ember memory integration (tokens + training data + chat)
- [x] Custom FastAPI web UI (no Gradio)
- [x] Full test suite (21/21 passing)
- [ ] APOLLO-Mini optimizer
- [ ] Disk-based activation checkpointing
- [ ] Push-to-Hub from web UI
- [ ] Live benchmark execution in UI

---

## License

MIT

---

Built for [Lila](https://github.com/ticketguy/Lila) — a private family ASI assistant.  
Designed by Sammie (@ticketguy).
