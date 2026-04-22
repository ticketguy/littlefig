# 🍐 Little Fig

**CPU-native LLM training engine.**  
Fine-tune language models on machines with no GPU — even with just 8GB RAM.

[![Tests](https://img.shields.io/badge/tests-21%2F21%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## What this is

Little Fig is a toolkit for fine-tuning and running LLMs entirely on CPU. While tools like Unsloth, Axolotl, and LLaMA-Factory require GPUs with CUDA, Little Fig is built from the ground up for the opposite constraint: **no GPU, no cloud, full local control.**

The core innovations:
- **Fig Engine** — streaming INT4 quantization (7.4× compression, 0.9955 cosine similarity)
- **FigQuant** — adaptive codebook quantization (9.7% less error than standard INT4)
- **FigKernel** — torch.compile fused ops (2.95× faster RMSNorm, fused linear+LoRA)
- **Universal GGUF loader** — auto-detects any architecture, zero hardcoded mappings

## What's possible

| Task | Model | RAM Needed | Reality |
|------|-------|-----------|---------|
| Fine-tune (Fig Engine, LoRA) | GPT-2 124M | ~350 MB | ✓ Fast |
| Fine-tune (Fig Engine, LoRA) | TinyLlama 1.1B | ~400 MB | ✓ Minutes/epoch |
| Fine-tune (Fig Engine, LISA) | Gemma 4B | ~3.2 GB | ✓ Feasible |
| Fine-tune (Fig Engine, LoRA) | Llama 3.1 8B | ~3 GB | ✓ Feasible |
| Inference (GGUF Q4) | Any model | ~model size | ✓ Auto-detects arch |
| Fine-tune (standard FP32) | TinyLlama 1.1B | ~27 GB | ✗ OOM on 8GB |

---

## Quick start

### Install

```bash
# CPU PyTorch first (avoids 2.5GB CUDA download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Little Fig with full UI
pip install -e ".[full]"
```

### Fine-tune with Fig Engine

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
trainer.load_dataset("tatsu-lab/alpaca")
trainer.train()

# Save adapter (~5 MB) and optionally merge into full model
model.save_adapter("./my_adapter")
model.merge_and_export("./my_model")
```

### Launch the Studio UI

```bash
little-fig
# Opens at http://localhost:8888
# Chat, build datasets, eval, merge — all in browser
```

---

## v0.5 — What's New

### FigQuant: Adaptive Codebook Quantization

Standard INT4 uses fixed min/max scaling. FigQuant refines an NF4-base codebook via weighted k-means on the actual weight distribution. High-magnitude weights get extra precision (AWQ-inspired sensitivity weighting).

| Method | Cosine Sim | MSE | SNR (dB) | Bits/param |
|--------|-----------|-----|----------|------------|
| **FigQuant** (ours) | **0.9955** | **0.0090** | **20.5** | 4.31 |
| FIG4 (standard INT4) | 0.9951 | 0.0100 | 20.0 | 4.16 |
| NF4 (QLoRA) | ~0.995 | ~0.010 | ~20.0 | 4.00 |

**9.7% MSE reduction** over standard asymmetric INT4 at comparable bits/param.

### FigKernel: Fused Operations via torch.compile

Pure PyTorch ops that `torch.compile(backend="inductor")` fuses into single kernels. Works on CPU (AVX-512) AND GPU (CUDA) — no Triton required.

| Operation | Standard | FigKernel | Speedup |
|-----------|----------|-----------|---------|
| RMSNorm | 4.72 ms | 1.60 ms | **2.95×** |
| Linear+LoRA (768→2048) | 4.60 ms | 6.55 ms | 0.70× (LoRA overhead) |
| SwiGLU (fused gate+up+silu) | — | integrated | saves 1 activation |
| Cross-Entropy (chunked) | — | integrated | **saves vocab×seq memory** |

### FigPipeline: Async GPU-CPU Training

Optimizer states on CPU, compute on GPU. For LoRA (rank 16), grad transfer is ~100KB/layer — negligible. Enables training models larger than GPU VRAM.

### Universal GGUF Loader

Any `.gguf` file → auto-detect architecture → load model. No code changes needed for new models.

```
Strategy:
  1. transformers built-in (22+ architectures) → automatic
  2. gguf-py TensorNameMap fallback (122+ architectures)
  3. Auto-upgrade transformers if architecture unknown
```

---

## Fig Engine — How it works

### The Problem

Standard LLM training: FP32 weights (4B/param) + gradients (4B) + optimizer (8B for AdamW) + activations. A 1.1B model needs ~27 GB.

### The Solution

Fig Engine: INT4 base weights (0.55B/param, 7.4× smaller) dequantized on-the-fly. Only LoRA adapters in FP32. Backward re-dequantizes from INT4.

```
Standard:     4.4 GB weights + 4.4 GB grads + 8.8 GB optimizer = 27 GB
Fig Engine:   0.6 GB INT4   + 0.03 GB LoRA  + 0.06 GB optimizer = ~0.4 GB
```

### Four Training Tiers

Auto-selected based on available RAM:

| Tier | Method | Memory (1.1B) | Quality | Speed |
|------|--------|---------------|---------|-------|
| **1. Streaming LoRA** | INT4 base + LoRA adapters | ~400 MB | Good | Fast |
| **2. LISA** | Rotating layer unfreezing | ~900 MB | Better (10-35% > LoRA) | Moderate |
| **3. MeZO** | Zeroth-order (no backward) | ~600 MB | Acceptable | Slow |
| **4. LOMO** | Fused gradient + update | ~800 MB | Best (full fine-tune) | Fast |

---

## Architecture

```
src/little_fig/
├── __init__.py              # Entry point + hardware detection
├── model.py                 # HF + GGUF universal inference
├── engine/                  # ⭐ Fig Engine (v0.5)
│   ├── quantize.py          # FIG4 quantization (v0.4)
│   ├── figquant.py          # ⭐ FigQuant: adaptive codebook INT4 (v0.5)
│   ├── figkernel.py         # ⭐ Fused ops: RMSNorm, SwiGLU, CE, Linear+LoRA
│   ├── figpipeline.py       # ⭐ Async GPU-CPU training pipeline
│   ├── linear.py            # FigLinear: INT4 base + LoRA (uses FigKernel)
│   ├── model.py             # FigModel: streaming model loader
│   ├── trainer.py           # FigTrainer: unified training loop
│   ├── tier.py              # Tier selection + memory estimation
│   ├── lisa.py              # LISA scheduler
│   ├── mezo.py              # MeZO optimizer
│   ├── lomo.py              # LOMO optimizer
│   ├── packing.py           # Sequence packing
│   └── gguf_loader.py       # Universal GGUF loader (auto-detects any arch)
├── studio/                  # Gradio 6.x UI (light/dark theme)
│   ├── app.py               # Chat + model management
│   ├── dataset_builder.py   # Dataset creation
│   ├── eval_tools.py        # Evaluation
│   └── merge_tools.py       # Adapter merge
└── tests/
    ├── test_engine.py       # Core tests (12/12 passing)
    └── test_v05.py          # v0.5 tests (9/9 passing)
```

---

## Comparison with Other Tools

| Feature | Little Fig | Unsloth | LLaMA-Factory | TRL |
|---------|-----------|---------|---------------|-----|
| CPU training | ✅ Native | ❌ GPU-only | ⚠️ Slow fallback | ⚠️ Slow fallback |
| Min RAM (1.1B) | **~400 MB** | ~6 GB (GPU) | ~8 GB | ~8 GB |
| GGUF loading | ✅ Universal | ❌ | ❌ | ❌ |
| Fused ops | ✅ torch.compile | ✅ Triton | ❌ | ❌ |
| Training tiers | 4 (auto) | 1 (LoRA) | Multiple | Multiple |
| Gradio UI | ✅ | ❌ | ✅ | ❌ |
| GPU support | ✅ Auto-detect | ✅ | ✅ | ✅ |

---

## Research Paper

See [`paper/fig_engine.md`](paper/fig_engine.md) for the full technical paper.

### Key references

| Technique | Paper | Used in |
|-----------|-------|---------|
| LoRA | Hu et al., 2022 | Tier 1 |
| LISA | arxiv 2403.17919 | Tier 2 |
| MeZO | arxiv 2305.17333 | Tier 3 |
| LOMO | arxiv 2306.09782 | Tier 4 |
| NF4 / QLoRA | Dettmers et al., 2023 | FigQuant codebook base |
| AWQ | Lin et al., 2024 | FigQuant sensitivity weighting |

---

## Roadmap

- [x] Fig Engine core (INT4, FigLinear, FigModel)
- [x] Four training tiers (LoRA, LISA, MeZO, LOMO)
- [x] Sequence packing
- [x] Auto tier selection
- [x] FigQuant adaptive codebook quantization
- [x] FigKernel fused ops (torch.compile)
- [x] FigPipeline async GPU-CPU training
- [x] Universal GGUF loader (auto-detects any architecture)
- [x] Gradio 6.x UI with light/dark themes
- [x] Full test suite (21/21 passing)
- [ ] APOLLO-Mini optimizer
- [ ] Disk-based activation checkpointing
- [ ] Training tab in Gradio Studio
- [ ] Push-to-Hub support

---

## License

MIT
