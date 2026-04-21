# 🍐 Little Fig

**CPU-native LLM inference and fine-tuning engine.**  
Built for machines with no GPU. Honest about what's possible.

---

## What this is

Little Fig is a self-contained toolkit for running and fine-tuning language models entirely on CPU. It's not Unsloth — Unsloth is built on CUDA kernels. Little Fig is built for the opposite constraint: no GPU, no cloud, full local control.

## What's actually possible on CPU

| Task | Model | Reality |
|------|-------|---------|
| Inference (HF float32) | TinyLlama 1.1B | ~2–5s per response ✓ |
| Inference (HF float32) | Gemma 4B | ~20–40s per response ⚠ |
| Inference (GGUF Q4) | Gemma 4B | ~3–8s per response ✓ |
| Fine-tuning (LoRA) | TinyLlama 1.1B | ~5–15 min/epoch ✓ |
| Fine-tuning (LoRA) | Gemma 4B | Hours per step ✗ |

**Recommendation:** Use GGUF for Gemma 4B inference. Use TinyLlama for fine-tuning experiments.

---

## Quick start

### 1. Install (inference only)

```bash
pip install -e .
```

### 2. Install with fine-tuning support

```bash
pip install -e ".[train]"
```

### 3. Install with GGUF support (recommended for Gemma 4B)

```bash
pip install -e ".[gguf]"
```

### 4. Launch the studio

```bash
little-fig
# → http://localhost:8888
```

---

## GGUF setup (fast CPU inference)

Download a quantized model:

```bash
pip install huggingface_hub
huggingface-cli download bartowski/gemma-3-4b-it-GGUF \
    gemma-3-4b-it-Q4_K_M.gguf \
    --local-dir ./models
```

Then in `studio/app.py`, uncomment the GGUF_MODELS entry and restart.

---

## Fine-tuning

```python
from little_fig.trainer import FigTrainer, FigTrainingConfig

config = FigTrainingConfig(
    lora_r=8,
    num_epochs=3,
    max_seq_length=512,  # Keep low on CPU
)

trainer = FigTrainer.from_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", config)
trainer.load_dataset("./data/my_data.jsonl")
trainer.train(output_dir="./checkpoints/run_01")
```

Dataset format (JSONL):

```json
{"instruction": "Summarize this text.", "input": "Long text here...", "output": "Summary here."}
{"instruction": "Write a haiku about rain.", "input": "", "output": "Rain on the window..."}
```

---

## Architecture

```
src/little_fig/
├── __init__.py          # Entry point
├── model.py             # FigLanguageModel (HF) + FigGGUFModel (llama.cpp)
├── trainer.py           # FigTrainer — LoRA fine-tuning
└── studio/
    └── app.py           # Gradio UI with streaming
```

---

## When you get a GPU later

This codebase is GPU-ready. Change two things:

1. `model.py`: `device_map="cpu"` → `device_map="auto"`
2. `trainer.py`: `no_cuda=True` → `no_cuda=False`, enable `fp16=True`

That's it. Everything else works unchanged.
