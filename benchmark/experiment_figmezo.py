#!/usr/bin/env python3
"""
FigMeZO Experiment — Error-Shaped vs Standard MeZO
====================================================

Controlled head-to-head on GPT-2 (124M) with Alpaca 200 examples.
All hyperparameters identical; only the perturbation distribution differs.

FigMeZO hypothesis: shaping z ~ N(0, diag(σ²)) by q_scales concentrates the
single zeroth-order probe direction where LoRA needs to compensate most,
yielding faster convergence vs isotropic MeZO.

Run:
    python benchmark/experiment_figmezo.py

Requires: transformers, datasets, little_fig installed.
"""

import sys, os, time, json, gc
import numpy as np
import torch
import torch.nn as nn

# ── Deps ──────────────────────────────────────────────────────────────────────
print("[SETUP] Installing dependencies...", flush=True)
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "accelerate"])

# Install little_fig from repo root if not already installed
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "src"))
try:
    import little_fig
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", repo_root])
    import little_fig

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from little_fig.engine.figquant import figquant_quantize, figquant_dequantize
from little_fig.engine.linear import FigLinear
from little_fig.engine.mezo import MeZOOptimizer, MeZOConfig
from little_fig.engine.figmezo import FigMeZO, FigMeZOConfig

# ── Experiment config (IDENTICAL for both methods) ────────────────────────────
MODEL_NAME    = "gpt2"
DATASET_NAME  = "tatsu-lab/alpaca"
N_EXAMPLES    = 200
MAX_SEQ       = 128
LR            = 1e-5
EPSILON       = 1e-3
SEED          = 42
STEPS         = 100
GROUP_SIZE    = 128
LORA_R        = 8
LORA_ALPHA    = 16
LORA_TARGETS  = ["c_attn", "c_proj"]   # GPT-2 attention layers
SHAPING_STR   = 0.7                     # FigMeZO shaping_strength

torch.manual_seed(SEED)
device = torch.device("cpu")
print(f"\n[CONFIG] Model={MODEL_NAME} | Steps={STEPS} | LR={LR} | ε={EPSILON} | "
      f"shaping_strength={SHAPING_STR}\n", flush=True)


# ── Data preparation ──────────────────────────────────────────────────────────

def build_dataset(tokenizer):
    print("[DATA] Loading Alpaca...", flush=True)
    ds = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    ds = ds.select(range(N_EXAMPLES))

    def fmt(ex):
        prompt = ex["instruction"]
        if ex.get("input"):
            prompt += f"\n\n{ex['input']}"
        prompt += f"\n\n{ex['output']}"
        return {"text": prompt}

    ds = ds.map(fmt)
    tokenizer.pad_token = tokenizer.eos_token

    def tok(ex):
        enc = tokenizer(
            ex["text"], truncation=True, max_length=MAX_SEQ,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        labels    = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}

    ds = ds.map(tok, remove_columns=ds.column_names)
    ds.set_format("torch")
    print(f"[DATA] {len(ds)} examples ready.", flush=True)
    return ds


# ── Model construction (FigQuant + LoRA) ─────────────────────────────────────

def build_fig_model(base_model):
    """
    Wrap target linear layers in FigLinear (FigQuant INT4 + LoRA).
    Returns the wrapped model with only LoRA params requiring grad.
    """
    print("[MODEL] Quantizing and wrapping target layers...", flush=True)
    target_count = 0

    for name, module in base_model.named_modules():
        for target in LORA_TARGETS:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Get parent module to replace
                parts = name.split(".")
                parent = base_model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                child_name = parts[-1]

                W = module.weight.data.float()
                fq = figquant_quantize(W, group_size=GROUP_SIZE, n_iters=8)

                bias = module.bias.data.float() if module.bias is not None else None
                fig_layer = FigLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    fq=fq,
                    lora_r=LORA_R,
                    lora_alpha=LORA_ALPHA,
                    bias=bias,
                    mode="fast",
                )
                setattr(parent, child_name, fig_layer)
                target_count += 1
                break

    # Freeze everything except LoRA
    for name, param in base_model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in base_model.parameters())
    print(f"[MODEL] {target_count} layers wrapped | "
          f"trainable={trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)
    return base_model


# ── Training loop ─────────────────────────────────────────────────────────────

def run_experiment(method_name: str, optimizer) -> dict:
    """
    Run STEPS training steps and record loss curve + timing.
    Returns dict with losses, final_loss, time_per_step.
    """
    dataset = optimizer._dataset
    n = len(dataset)
    rng = np.random.RandomState(SEED)
    indices = list(range(n))
    rng.shuffle(indices)

    losses = []
    t_start = time.time()

    print(f"\n[{method_name}] Starting {STEPS} steps...", flush=True)
    print(f"  {'Step':>5}  {'Loss':>10}", flush=True)
    print(f"  {'─'*5}  {'─'*10}", flush=True)

    for step in range(STEPS):
        idx = indices[step % n]
        batch = dataset[idx]
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        labels    = batch["labels"].unsqueeze(0).to(device)

        def forward_fn():
            out = optimizer.model(input_ids=input_ids, labels=labels)
            return out.loss

        loss = optimizer.step(forward_fn)
        losses.append(loss)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  {step+1:>5}  {loss:>10.4f}", flush=True)

    elapsed = time.time() - t_start
    sps = elapsed / STEPS

    print(f"\n[{method_name}] Done. "
          f"Final loss={losses[-1]:.4f} | "
          f"Min loss={min(losses):.4f} | "
          f"{sps:.2f}s/step", flush=True)

    return {
        "losses": losses,
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "loss_at_10": losses[9],
        "loss_at_50": losses[49],
        "time_total": elapsed,
        "time_per_step": sps,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset   = build_dataset(tokenizer)

    results = {}

    # ── Experiment A: Standard MeZO ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  EXPERIMENT A: Standard MeZO (isotropic z ~ N(0, I))")
    print("="*60, flush=True)
    gc.collect()

    base_a = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model_a = build_fig_model(base_a).to(device)

    mezo_cfg = MeZOConfig(learning_rate=LR, epsilon=EPSILON, seed=SEED)
    mezo_opt = MeZOOptimizer(model_a, mezo_cfg)
    mezo_opt._dataset = dataset   # attach for run_experiment

    results["MeZO"] = run_experiment("MeZO", mezo_opt)
    del model_a, mezo_opt, base_a
    gc.collect()

    # ── Experiment B: FigMeZO ────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  EXPERIMENT B: FigMeZO (shaped z, strength={SHAPING_STR})")
    print("="*60, flush=True)
    gc.collect()

    base_b = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model_b = build_fig_model(base_b).to(device)

    figmezo_cfg = FigMeZOConfig(
        learning_rate=LR, epsilon=EPSILON, seed=SEED,
        shaping_strength=SHAPING_STR,
    )
    figmezo_opt = FigMeZO(model_b, figmezo_cfg)
    figmezo_opt._dataset = dataset   # attach for run_experiment

    results["FigMeZO"] = run_experiment("FigMeZO", figmezo_opt)
    del model_b, figmezo_opt, base_b
    gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)

    m  = results["MeZO"]
    fm = results["FigMeZO"]

    delta_final = fm["final_loss"] - m["final_loss"]
    delta_10    = fm["loss_at_10"] - m["loss_at_10"]
    delta_50    = fm["loss_at_50"] - m["loss_at_50"]
    winner_final = "FigMeZO ✅" if delta_final < 0 else "MeZO"
    winner_10    = "FigMeZO ✅" if delta_10 < 0 else "MeZO"
    winner_50    = "FigMeZO ✅" if delta_50 < 0 else "MeZO"

    print(f"\n  {'Metric':<28}  {'MeZO':>10}  {'FigMeZO':>10}  {'Δ':>10}  Winner")
    print(f"  {'─'*28}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")
    print(f"  {'Loss at step 10':<28}  {m['loss_at_10']:>10.4f}  {fm['loss_at_10']:>10.4f}  {delta_10:>+10.4f}  {winner_10}")
    print(f"  {'Loss at step 50':<28}  {m['loss_at_50']:>10.4f}  {fm['loss_at_50']:>10.4f}  {delta_50:>+10.4f}  {winner_50}")
    print(f"  {'Final loss (step 100)':<28}  {m['final_loss']:>10.4f}  {fm['final_loss']:>10.4f}  {delta_final:>+10.4f}  {winner_final}")
    print(f"  {'Min loss':<28}  {m['min_loss']:>10.4f}  {fm['min_loss']:>10.4f}")
    print(f"  {'Time/step (s)':<28}  {m['time_per_step']:>10.3f}  {fm['time_per_step']:>10.3f}")

    overhead_pct = 100 * (fm["time_per_step"] - m["time_per_step"]) / m["time_per_step"]
    print(f"\n  FigMeZO overhead: {overhead_pct:+.1f}% vs standard MeZO")

    if delta_final < 0:
        print(f"\n  🎯 FigMeZO wins at step 100: {abs(delta_final):.4f} lower loss "
              f"({100*abs(delta_final)/m['final_loss']:.1f}% improvement)")
    else:
        print(f"\n  ⚠️  Standard MeZO wins — revisit shaping_strength or target layers")

    # Save results
    out = {
        "config": {
            "model": MODEL_NAME, "steps": STEPS, "lr": LR,
            "epsilon": EPSILON, "seed": SEED,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_targets": LORA_TARGETS, "shaping_strength": SHAPING_STR,
        },
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "losses"}
                    for k, v in results.items()},
        "loss_curves": {k: v["losses"] for k, v in results.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "figmezo_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
