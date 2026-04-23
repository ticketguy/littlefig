# 🍐 Little Fig

**CPU-native LLM training engine with embedded cognitive memory.**  
Fine-tune language models on machines with no GPU — even with just 8GB RAM.  
Train models with [Ember's Diaries](https://github.com/ticketguy/embers-diaries) cognitive memory built into their weights.

[![Tests](https://img.shields.io/badge/tests-21%2F21%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Engine Stack

| Layer | Module | What it does |
|-------|--------|-------------|
| **Quantization** | FigQuant | Adaptive codebook INT4 (9.7% less MSE than NF4, 7.4× compression) |
| **Compute** | FigKernel | torch.compile fused ops (2.95× faster RMSNorm, fused LoRA) |
| **Training** | 4 Tiers | LoRA, LISA, MeZO, LOMO — auto-selected by available RAM |
| **Optimizer** | FigPipeline | Async GPU-CPU training with CPU-resident optimizer states |
| **Memory** | Ember | Cognitive memory tokens + training data for embedded memory |

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
│   ├── linear.py            # FigLinear: FigQuant base + LoRA
│   ├── model.py             # FigModel: streaming loader
│   ├── trainer.py           # FigTrainer: unified training loop
│   ├── ember_integration.py # Ember memory tokens + training data
│   ├── tier.py              # Tier selection + memory estimation
│   ├── lisa.py              # LISA scheduler
│   ├── mezo.py              # MeZO optimizer
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

## Research Paper

See [`paper/fig_engine.md`](paper/fig_engine.md) for the full technical paper.

---

## License

MIT
