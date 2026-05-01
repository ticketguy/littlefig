#!/usr/bin/env python3
"""
Little Fig — GPU Benchmark (corrected)
=======================================
Self-contained: clones repo, installs deps, runs everything.
Fixes from investigation:
  - Equal batch sizes across all methods
  - Autocast for FigQuant GPU path
  - Same target modules for fair LoRA comparison
"""

import os, sys, subprocess, json, time, gc, traceback
import numpy as np

# ── Setup ─────────────────────────────────────────────────────────────────────
print("[SETUP] Installing...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers", "accelerate", "peft", "bitsandbytes",
    "datasets", "sentencepiece", "protobuf", "psutil", "numpy"])

if not os.path.exists("/app/littlefig"):
    subprocess.check_call(["git", "clone", "https://github.com/ticketguy/littlefig.git", "/app/littlefig"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "/app/littlefig[train]"])
sys.path.insert(0, "/app/littlefig/src")

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config (IDENTICAL across all methods) ─────────────────────────────────────
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET = "tatsu-lab/alpaca"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS_PEFT = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_TARGETS_FIG = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_SEQ = 512
TRAIN_STEPS = 200
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
GROUP_SIZE = 128

RESULTS = {}
def log(msg): print(f"[BENCH] {msg}", flush=True)
def gpu_mb(): return torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0.0
def reset():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
def safe_run(name, fn):
    log(f"\n{'='*70}\n  {name}\n{'='*70}")
    try:
        r = fn(); RESULTS[name] = r; log(f"  ✅ {name}"); return r
    except Exception as e:
        log(f"  ❌ {name}: {e}"); traceback.print_exc(); RESULTS[name] = {"error": str(e)}; return None

# ═══════════════════════════════════════════════════════════════════════════════
# A. QUANTIZATION QUALITY — FigQuant vs NF4 vs Absmax on TinyLlama
# ═══════════════════════════════════════════════════════════════════════════════

def meas(o, d):
    o, d = o.reshape(-1).float(), d.reshape(-1).float()
    mse = F.mse_loss(d, o).item()
    cos = F.cosine_similarity(o.unsqueeze(0), d.unsqueeze(0)).item()
    snr = 10*np.log10(o.pow(2).mean().item()/max(mse,1e-20))
    return {"mse": mse, "cos": cos, "snr": snr}

def nf4_qd(W, gs=128):
    s, n = W.shape, W.numel(); f = W.reshape(-1).float()
    p = (gs-n%gs)%gs
    if p>0: f = torch.cat([f, torch.zeros(p)])
    g = f.reshape(-1,gs); sc = g.abs().amax(1).clamp(min=1e-10)
    cb = torch.tensor([-1.0,-0.6962,-0.5251,-0.3949,-0.2844,-0.1848,-0.0911,0.0,
                        0.0796,0.1609,0.2461,0.3379,0.4407,0.5626,0.7230,1.0])
    idx = ((g/sc.unsqueeze(1)).reshape(-1).unsqueeze(1)-cb.unsqueeze(0)).abs().argmin(1).reshape(-1,gs)
    return (torch.gather(cb.unsqueeze(0).expand(idx.shape[0],-1),1,idx.long())*sc.unsqueeze(1)).reshape(-1)[:n].reshape(s)

def absmax_qd(W, gs=128):
    s, n = W.shape, W.numel(); f = W.reshape(-1).float()
    p = (gs-n%gs)%gs
    if p>0: f = torch.cat([f, torch.zeros(p)])
    g = f.reshape(-1,gs); sc = g.abs().amax(1).clamp(min=1e-10)
    cb = torch.linspace(-1.0,1.0,16)
    idx = ((g/sc.unsqueeze(1)).reshape(-1).unsqueeze(1)-cb.unsqueeze(0)).abs().argmin(1).reshape(-1,gs)
    return (torch.gather(cb.unsqueeze(0).expand(idx.shape[0],-1),1,idx.long())*sc.unsqueeze(1)).reshape(-1)[:n].reshape(s)

def bench_quant():
    from transformers import AutoModelForCausalLM
    from little_fig.engine.figquant import figquant_quantize, figquant_dequantize
    log("Loading TinyLlama FP32..."); reset()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    methods = {"figquant":{},"nf4":{},"absmax":{}}; n=0; fw_n=0; fw_a=0
    for name, param in model.named_parameters():
        if param.ndim!=2 or param.numel()<1024: continue
        W = param.data.float()
        q = figquant_quantize(W, group_size=GROUP_SIZE, n_iters=8)
        ef = meas(W, figquant_dequantize(q)); en = meas(W, nf4_qd(W,GROUP_SIZE)); ea = meas(W, absmax_qd(W,GROUP_SIZE))
        for m,e in [("figquant",ef),("nf4",en),("absmax",ea)]:
            for k,v in e.items(): methods[m].setdefault(k,[]).append(v)
        if ef["mse"]<en["mse"]: fw_n+=1
        if ef["mse"]<ea["mse"]: fw_a+=1
        n+=1
        if n%20==0: log(f"  {n} layers...")
    avgs = {m:{k:float(np.mean(v)) for k,v in d.items()} for m,d in methods.items()}
    mvn = (avgs["nf4"]["mse"]-avgs["figquant"]["mse"])/avgs["nf4"]["mse"]*100
    mva = (avgs["absmax"]["mse"]-avgs["figquant"]["mse"])/avgs["absmax"]["mse"]*100
    del model; gc.collect()
    return {"avgs":avgs,"n":n,"fw_nf4":fw_n,"fw_abs":fw_a,"mvn":mvn,"mva":mva}

# ═══════════════════════════════════════════════════════════════════════════════
# B. BnB NF4 actual quality via dequantize
# ═══════════════════════════════════════════════════════════════════════════════

def bench_bnb_actual():
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    reset(); log("BnB NF4 actual quality...")
    bc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    m4 = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bc, device_map="auto")
    m16 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cpu")
    p4, p16 = dict(m4.named_parameters()), dict(m16.named_parameters())
    tot_mse=0; tot_cos=0; n=0
    for name in p16:
        if name not in p4: continue
        w16 = p16[name].data.float().cpu()
        pr = p4[name]
        if hasattr(pr,'quant_state') and pr.quant_state is not None:
            try: w4 = bnb.functional.dequantize_4bit(pr.data, pr.quant_state).float().cpu()
            except: continue
        else: w4 = pr.data.float().cpu()
        if w16.shape!=w4.shape or w16.ndim!=2 or w16.numel()<1024: continue
        tot_mse += F.mse_loss(w4.flatten(), w16.flatten()).item()
        tot_cos += F.cosine_similarity(w16.flatten().unsqueeze(0), w4.flatten().unsqueeze(0)).item()
        n+=1
    del m4, m16; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"n":n, "avg_mse":tot_mse/max(n,1), "avg_cos":tot_cos/max(n,1)}

# ═══════════════════════════════════════════════════════════════════════════════
# C. PERPLEXITY
# ═══════════════════════════════════════════════════════════════════════════════

def ppl_eval(model, tok, texts, dev, stride=512, maxl=1024):
    model.eval()
    enc = tok("\n\n".join(texts), return_tensors="pt")
    sl = enc.input_ids.size(1); nlls=[]; pe=0
    for b in range(0, sl, stride):
        e = min(b+maxl, sl); tl = e-pe
        ids = enc.input_ids[:,b:e].to(dev)
        tgt = ids.clone(); tgt[:,:-tl]=-100
        with torch.no_grad(): nlls.append(model(ids, labels=tgt).loss.item())
        pe=e
        if e==sl: break
    return float(np.exp(np.mean(nlls)))

def bench_ppl():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    from little_fig.engine import FigModel
    from little_fig.engine.tier import TrainingTier
    log("Loading wikitext-2...")
    wt = load_dataset("wikitext","wikitext-2-raw-v1",split="test")
    texts = [t for t in wt["text"] if len(t.strip())>50][:200]
    tok = AutoTokenizer.from_pretrained(MODEL)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = {}
    for lbl, load_fn in [
        ("fp16", lambda: AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")),
        ("bnb_nf4", lambda: AutoModelForCausalLM.from_pretrained(MODEL,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16), device_map="auto")),
        ("bnb_int8", lambda: AutoModelForCausalLM.from_pretrained(MODEL,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")),
    ]:
        log(f"  PPL: {lbl}..."); reset()
        m = load_fn(); R[lbl] = {"ppl": ppl_eval(m, tok, texts, dev), "gpu_mb": gpu_mb()}
        log(f"    {lbl}: {R[lbl]['ppl']:.2f}"); del m; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    log("  PPL: FigQuant..."); reset()
    fm = FigModel.from_pretrained(MODEL, lora_r=0, tier=TrainingTier.STREAMING_LORA, group_size=GROUP_SIZE)
    if torch.cuda.is_available(): fm = fm.to(dev)
    R["figquant"] = {"ppl": ppl_eval(fm, fm.tokenizer, texts, dev), "gpu_mb": gpu_mb()}
    log(f"    FigQuant: {R['figquant']['ppl']:.2f}"); del fm; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return R

# ═══════════════════════════════════════════════════════════════════════════════
# D. TRAINING — all methods with IDENTICAL setup
# ═══════════════════════════════════════════════════════════════════════════════

def load_alpaca():
    from datasets import load_dataset
    return load_dataset(DATASET, split="train").select(range(2000))

def _hf_loop(model, tokenizer, dataset, name):
    dev = next(model.parameters()).device
    def tok_fn(ex):
        inst=ex.get("instruction",""); inp=ex.get("input","").strip(); out=ex.get("output","")
        txt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}" if inp else \
              f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        e = tokenizer(txt, truncation=True, max_length=MAX_SEQ, padding="max_length")
        e["labels"] = e["input_ids"].copy(); return e
    td = dataset.map(tok_fn, remove_columns=dataset.column_names); td.set_format("torch")
    from torch.utils.data import DataLoader
    dl = DataLoader(td, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: {k:torch.stack([x[k] for x in b]) for k in b[0] if isinstance(b[0][k],torch.Tensor)},
        drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    model.train(); losses=[]; times=[]; gs=0; al=0.0; reset(); t0=time.time()
    for batch in dl:
        if gs>=TRAIN_STEPS*GRAD_ACCUM: break
        batch = {k:v.to(dev) for k,v in batch.items()}
        ts = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            loss = model(**batch).loss / GRAD_ACCUM
        loss.backward(); al+=loss.item(); gs+=1
        if gs%GRAD_ACCUM==0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            s=gs//GRAD_ACCUM; losses.append(al); times.append(time.time()-ts); al=0.0
            if s%20==0: log(f"  [{name}] step={s} loss={losses[-1]:.4f}")
    tt=time.time()-t0; pm=gpu_mb()
    del model,opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"method":name,"losses":[float(l) for l in losses],"final":float(losses[-1]) if losses else None,
            "time_s":tt,"steps":len(losses),"ms_step":float(np.mean(times)*1000) if times else None,"gpu_mb":pm}

def train_fp16(ds):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    reset(); log("Training FP16 LoRA...")
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
    t = AutoTokenizer.from_pretrained(MODEL); t.pad_token=t.eos_token
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    m = get_peft_model(m, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS_PEFT, bias="none", task_type="CAUSAL_LM"))
    return _hf_loop(m, t, ds, "fp16_lora")

def train_nf4(ds):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    reset(); log("Training BnB NF4...")
    m = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16), device_map="auto")
    t = AutoTokenizer.from_pretrained(MODEL); t.pad_token=t.eos_token
    m = prepare_model_for_kbit_training(m)
    m = get_peft_model(m, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS_PEFT, bias="none", task_type="CAUSAL_LM"))
    return _hf_loop(m, t, ds, "bnb_nf4")

def train_int8(ds):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    reset(); log("Training BnB INT8...")
    m = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
    t = AutoTokenizer.from_pretrained(MODEL); t.pad_token=t.eos_token
    m = prepare_model_for_kbit_training(m)
    m = get_peft_model(m, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS_PEFT, bias="none", task_type="CAUSAL_LM"))
    return _hf_loop(m, t, ds, "bnb_int8")

def train_figquant(ds):
    from little_fig.engine import FigModel, FigTrainer, FigTrainingConfig
    from little_fig.engine.tier import TrainingTier
    from torch.utils.data import DataLoader
    reset(); log("Training FigQuant LoRA...")
    model = FigModel.from_pretrained(MODEL, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
        tier=TrainingTier.STREAMING_LORA, group_size=GROUP_SIZE, target_modules=LORA_TARGETS_FIG)
    tok = model.tokenizer
    cfg = FigTrainingConfig(tier="streaming_lora", lora_r=LORA_R, lora_alpha=LORA_ALPHA,
        learning_rate=LR, max_seq_length=MAX_SEQ, use_packing=False, activation_checkpointing=True)
    trainer = FigTrainer(model, cfg)
    examples = [dict(r) for r in ds]
    def tok_fn(ex):
        inst=ex.get("instruction",""); inp=ex.get("input","").strip(); out=ex.get("output","")
        txt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}" if inp else \
              f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        e = tok(txt, truncation=True, max_length=MAX_SEQ, padding="max_length")
        return {"input_ids": e["input_ids"], "labels": e["input_ids"].copy(), "attention_mask": e["attention_mask"]}
    tokenized = [tok_fn(ex) for ex in examples]
    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            return {k: torch.tensor(v, dtype=torch.long) for k, v in self.data[i].items()}
    dl = DataLoader(SimpleDS(tokenized), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type=="cuda": model = model.to(dev)
    params = model.get_trainable_parameters()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    model.model.train(); losses=[]; times=[]; gs=0; al=0.0; reset(); t0=time.time()
    for batch in dl:
        if gs>=TRAIN_STEPS*GRAD_ACCUM: break
        if dev.type=="cuda": batch = {k:v.to(dev) for k,v in batch.items()}
        ts = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type=="cuda"):
            loss = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         labels=batch["labels"]).loss / GRAD_ACCUM
        loss.backward(); al+=loss.item(); gs+=1
        if gs%GRAD_ACCUM==0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            s=gs//GRAD_ACCUM; losses.append(al); times.append(time.time()-ts); al=0.0
            if s%20==0: log(f"  [figquant] step={s} loss={losses[-1]:.4f}")
    tt=time.time()-t0; pm=gpu_mb()
    del model,opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"method":"figquant_lora","losses":[float(l) for l in losses],"final":float(losses[-1]) if losses else None,
            "time_s":tt,"steps":len(losses),"ms_step":float(np.mean(times)*1000) if times else None,"gpu_mb":pm}

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def summary():
    log("\n"+"="*80)
    log("  🍐 BENCHMARK: FigQuant vs Industry Quantization (GPU)")
    log("="*80)
    if "quant" in RESULTS and "error" not in RESULTS["quant"]:
        q=RESULTS["quant"]
        log(f"\n📊 A. WEIGHT QUALITY (TinyLlama 1.1B, {q['n']} layers)")
        log(f"   {'Method':>12} {'Cosine':>9} {'MSE':>12} {'SNR':>7}")
        log(f"   {'─'*44}")
        for m in ["figquant","nf4","absmax"]:
            a=q["avgs"][m]; lb={"figquant":"FigQuant★","nf4":"NF4(QLoRA)","absmax":"AbsmaxI4"}[m]
            log(f"   {lb:>12} {a['cos']:.6f} {a['mse']:.6e} {a['snr']:.1f}")
        log(f"   FigQuant vs NF4: {q['mvn']:+.1f}% MSE ({q['fw_nf4']}/{q['n']})")
    if "bnb" in RESULTS and "error" not in RESULTS["bnb"]:
        b=RESULTS["bnb"]
        log(f"   BnB NF4 actual: MSE={b['avg_mse']:.6e}, cos={b['avg_cos']:.6f}")
    if "ppl" in RESULTS and "error" not in RESULTS["ppl"]:
        p=RESULTS["ppl"]
        log(f"\n📊 B. PERPLEXITY (wikitext-2)")
        log(f"   {'Method':>12} {'PPL':>8} {'GPU MB':>8}")
        for m in ["fp16","bnb_nf4","bnb_int8","figquant"]:
            if m in p: log(f"   {m:>12} {p[m]['ppl']:>8.2f} {p[m]['gpu_mb']:>8.0f}")
    log(f"\n📊 C. TRAINING (Alpaca, {TRAIN_STEPS} steps, batch={BATCH_SIZE}x{GRAD_ACCUM})")
    log(f"   {'Method':>12} {'Loss':>8} {'Time':>7} {'ms/s':>6} {'GPU MB':>8}")
    log(f"   {'─'*48}")
    for k in ["t_fp16","t_nf4","t_int8","t_fig"]:
        if k in RESULTS and "error" not in RESULTS[k]:
            r=RESULTS[k]
            log(f"   {r['method']:>12} {r['final']:.4f} {r['time_s']:.0f}s {r['ms_step']:.0f} {r['gpu_mb']:.0f}")
    log("="*80)

if __name__ == "__main__":
    log(f"🍐 Little Fig GPU Benchmark")
    log(f"   PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"   GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB)")
    safe_run("quant", bench_quant)
    if torch.cuda.is_available(): safe_run("bnb", bench_bnb_actual)
    safe_run("ppl", bench_ppl)
    ds = load_alpaca()
    safe_run("t_fp16", lambda: train_fp16(ds))
    safe_run("t_nf4", lambda: train_nf4(ds))
    safe_run("t_int8", lambda: train_int8(ds))
    safe_run("t_fig", lambda: train_figquant(ds))
    summary()
    with open("/app/benchmark_results.json","w") as f: json.dump(RESULTS, f, indent=2, default=str)
    log("📁 Saved /app/benchmark_results.json")
