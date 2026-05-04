#!/usr/bin/env python3
"""Experiment: Standard MeZO vs FigMeZO on GPT-2 + Alpaca."""
import sys, os, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import numpy as np
from little_fig.engine import FigModel
from little_fig.engine.tier import TrainingTier
from little_fig.engine.mezo import MeZOOptimizer, MeZOConfig
from little_fig.engine.figmezo import FigMeZO, FigMeZOConfig

MODEL = "gpt2"
LR = 1e-5
EPSILON = 1e-3
STEPS = 100
SEED = 42

def log(msg): print(f"[EXP] {msg}", flush=True)

def prepare_data():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    log("Loading Alpaca (200 samples)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(200))
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    batches = []
    for row in ds:
        inst = row.get("instruction", ""); out = row.get("output", "")
        text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        enc = tok(text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        batches.append({"input_ids": enc["input_ids"], "labels": enc["input_ids"].clone()})
    return batches

def run_exp(model, optimizer, batches, steps, label):
    model.model.eval()
    losses = []
    t0 = time.time()
    for step in range(steps):
        batch = batches[step % len(batches)]
        def forward_fn():
            return model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
        loss = optimizer.step(forward_fn)
        losses.append(loss)
        if (step+1) % 20 == 0:
            log(f"  [{label}] step={step+1:3d}  loss={loss:.4f}")
    log(f"  [{label}] {time.time()-t0:.1f}s, final={losses[-1]:.4f}")
    return losses

if __name__ == "__main__":
    log("="*60)
    log("  EXPERIMENT: Standard MeZO vs FigMeZO")
    log("="*60)
    batches = prepare_data()

    log("\n--- Standard MeZO ---")
    m1 = FigModel.from_pretrained(MODEL, lora_r=16, lora_alpha=32, tier=TrainingTier.STREAMING_LORA)
    opt1 = MeZOOptimizer(m1.model, MeZOConfig(learning_rate=LR, epsilon=EPSILON, seed=SEED))
    losses_std = run_exp(m1, opt1, batches, STEPS, "MeZO")
    del m1, opt1; gc.collect()

    log("\n--- FigMeZO (0.7) ---")
    m2 = FigModel.from_pretrained(MODEL, lora_r=16, lora_alpha=32, tier=TrainingTier.STREAMING_LORA)
    opt2 = FigMeZO(m2.model, FigMeZOConfig(learning_rate=LR, epsilon=EPSILON, seed=SEED, shaping_strength=0.7))
    losses_07 = run_exp(m2, opt2, batches, STEPS, "FigMeZO.7")
    del m2, opt2; gc.collect()

    log("\n--- FigMeZO (1.0) ---")
    m3 = FigModel.from_pretrained(MODEL, lora_r=16, lora_alpha=32, tier=TrainingTier.STREAMING_LORA)
    opt3 = FigMeZO(m3.model, FigMeZOConfig(learning_rate=LR, epsilon=EPSILON, seed=SEED, shaping_strength=1.0))
    losses_10 = run_exp(m3, opt3, batches, STEPS, "FigMeZO1.0")
    del m3, opt3; gc.collect()

    log("\n" + "="*60)
    log("  RESULTS")
    log("="*60)
    log(f"\n  {'Step':>5}  {'MeZO':>9}  {'Fig0.7':>9}  {'Fig1.0':>9}")
    log(f"  {'─'*38}")
    for i in [0,9,19,29,49,69,99]:
        if i < min(len(losses_std), len(losses_07), len(losses_10)):
            log(f"  {i+1:>5}  {losses_std[i]:>9.4f}  {losses_07[i]:>9.4f}  {losses_10[i]:>9.4f}")
    avg_s = np.mean(losses_std[-20:])
    avg_7 = np.mean(losses_07[-20:])
    avg_1 = np.mean(losses_10[-20:])
    log(f"\n  Avg last 20: MeZO={avg_s:.4f}  Fig0.7={avg_7:.4f} ({(avg_7-avg_s)/avg_s*100:+.1f}%)  Fig1.0={avg_1:.4f} ({(avg_1-avg_s)/avg_s*100:+.1f}%)")
    best = min(avg_s, avg_7, avg_1)
    if best < avg_s:
        log("\n  ✅ FigMeZO WINS")
    else:
        log("\n  ❌ Standard MeZO wins")
