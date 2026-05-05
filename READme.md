# 🍐 Little Fig

**CPU-native LLM training engine with embedded cognitive memory.**  
Fine-tune language models on machines with no GPU — even with just 8GB RAM.  
Train models with [Ember's Diaries](https://github.com/ticketguy/embers-diaries) cognitive memory built into their weights.

[![Tests](https://img.shields.io/badge/tests-21%2F21%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## What's New (v0.6)

| Research Finding | Improvement | Validated |
|---|---|---|
| **FigMeZO** — inverse error-shaped zeroth-order optimization | −18.6% loss vs standard MeZO | ✓ 3 seeds |
| **Sensitivity-guided LISA** — probe layers, weight selection by importance | −10% loss vs random LISA | ✓ controlled |
| **Shared codebook** — reuse one layer's codebook for all | 5× faster loading, 0.1% quality cost | ✓ 50 layers |
| **GPU mixed-precision** — dtype-safe forward pass | Prevents fp16/bf16 crashes on GPU | ✓ all modes |

All findings are original research — experimentally validated, not derived from other papers.

---

## Engine Stack

| Layer | Module | What it does |
|-------|--------|-------------|
| **Quantization** | FigQuant | Adaptive codebook INT4 (5.4% less MSE than NF4, 7.4× compression) |
| **Compute** | FigKernel | torch.compile fused ops (1.68× faster RMSNorm, fused LoRA) |
| **Training** | 4 Tiers | LoRA, LISA, MeZO, LOMO — auto-selected by available RAM |
| **Optimizer** | FigMeZO | Error-shaped zeroth-order (−18.6% vs standard MeZO) |
| **Pipeline** | FigPipeline | Async GPU-CPU training with CPU-resident optimizer states |
| **Memory** | Memory Fabric | Multi-adapter weight-space memory with gating, decay, and conflict detection |
| **Cognition** | Ember Integration | 9 memory tokens trained into model vocabulary |

## Benchmark Results (TinyLlama 1.1B, live data)

| Method | Cosine Sim | MSE | Wins |
|--------|:-:|:-:|:-:|
| **FigQuant** | **0.9956** | **5.64e-6** | **156/156** |
| NF4 (QLoRA) | 0.9953 | 5.97e-6 | 0/156 |
| Absmax INT4 | 0.9936 | 8.94e-6 | 0/156 |

FigQuant beats NF4 on every single layer of TinyLlama 1.1B.

## What's possible

| Task | Model | RAM Needed |
|------|-------|-----------|
| Fine-tune (LoRA) | GPT-2 124M | ~350 MB |
| Fine-tune (LoRA) | TinyLlama 1.1B | ~400 MB |
| Fine-tune (LISA) | Gemma 4B | ~3.2 GB |
| Fine-tune (LoRA) | Llama 3.1 8B | ~3 GB |
| Inference (GGUF Q4) | Any model | ~model size |
| Ember memory training | Any model | +tokens overhead |

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

### Fast loading (shared codebook mode)

```python
# 5× faster quantization at 0.1% quality cost — great for iteration
model = FigModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    shared_codebook=True,  # Reuse first layer's codebook for all
)
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
```

---

## Architecture

```
src/little_fig/
├── __init__.py              # Entry point + hardware detection
├── model.py                 # HF + GGUF universal inference
├── engine/
│   ├── figquant.py          # FigQuant: adaptive codebook INT4
│   ├── figkernel.py         # Fused ops: RMSNorm, SwiGLU, CE, Linear+LoRA
│   ├── figpipeline.py       # Async GPU-CPU training pipeline
│   ├── figmezo.py           # FigMeZO: error-shaped zeroth-order optimizer
│   ├── memory_fabric.py     # Memory Fabric: multi-adapter weight-space memory
│   ├── micro_trainer.py     # Micro-training: write memories between turns
│   ├── linear.py            # FigLinear: FigQuant base + LoRA (GPU dtype-safe)
│   ├── model.py             # FigModel: streaming loader + shared codebook
│   ├── trainer.py           # FigTrainer: unified training loop
│   ├── ember_integration.py # Ember memory tokens + training data
│   ├── tier.py              # Tier selection + memory estimation
│   ├── lisa.py              # LISA scheduler (sensitivity-weighted)
│   ├── mezo.py              # MeZO optimizer (standard)
│   ├── lomo.py              # LOMO optimizer
│   ├── packing.py           # Sequence packing
│   └── gguf_loader.py       # Universal GGUF loader
├── web/
│   ├── server.py            # REST API + WebSocket endpoints
│   └── static/index.html    # Frontend
└── tests/
    ├── test_engine.py       # 12 tests
    └── test_v05.py          # 9 tests
```

---

## Research

See [`paper/fig_engine.md`](paper/fig_engine.md) for the full technical paper.  
See [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) for ongoing research directions.

---

## License

MIT
