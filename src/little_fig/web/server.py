"""
Little Fig — FastAPI Backend v3

Endpoints:
  GET  /api/health              — detailed hardware + system info
  POST /api/model/load          — load HF model or local path
  POST /api/model/unload        — unload current model
  GET  /api/model/status        — current model info
  GET  /api/files/browse        — browse local dirs for model files
  GET  /api/chats               — list saved chat sessions
  POST /api/chats               — create new chat session
  GET  /api/chats/{id}          — get a chat session
  DELETE /api/chats/{id}        — delete a chat session
  WS   /api/chat                — streaming chat via WebSocket
  WS   /api/terminal            — live bidirectional terminal
  POST /api/upload              — file upload
"""

import os, sys, json, time, asyncio, uuid, platform, collections, subprocess, threading
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ── State ─────────────────────────────────────────────────────────────────────

_model = None
_model_id = None
_hw = None
_loading = False
_log_buffer = collections.deque(maxlen=500)

UPLOAD_DIR = Path(os.getcwd()) / ".fig_uploads"
CHATS_DIR = Path(os.getcwd()) / ".fig_chats"
UPLOAD_DIR.mkdir(exist_ok=True)
CHATS_DIR.mkdir(exist_ok=True)


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _log_buffer.append(line)
    print(line, flush=True)


def _get_hw():
    global _hw
    if _hw is None:
        try:
            from little_fig import detect_hardware
            _hw = detect_hardware()
        except Exception:
            import psutil
            _hw = {"device": "cpu", "gpu_available": False,
                   "cpu_cores": os.cpu_count(),
                   "ram_total_gb": round(psutil.virtual_memory().total/(1024**3),1),
                   "ram_available_gb": round(psutil.virtual_memory().available/(1024**3),1)}
    return _hw


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_hw(); _log("Server started"); yield

app = FastAPI(title="Little Fig", version="0.6.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent / "static"
if (STATIC_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")


@app.get("/")
async def index():
    p = STATIC_DIR / "index.html"
    return FileResponse(str(p)) if p.exists() else JSONResponse({"error": "No frontend"}, 500)


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={}, status_code=204)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    import psutil
    hw = _get_hw()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())
    return {
        "status": "ok", "version": "0.6.0",
        "platform": {"os": platform.system(), "os_version": platform.version()[:60],
                      "python": sys.version.split()[0], "arch": platform.machine()},
        "hardware": {
            "device": hw.get("device","cpu"),
            "gpu": hw.get("gpu_name") if hw.get("gpu_available") else None,
            "gpu_vram_gb": hw.get("gpu_vram_gb"),
            "cpu_cores": hw.get("cpu_cores"),
            "cpu_name": platform.processor()[:40] or "CPU",
            "ram_total_gb": round(mem.total/(1024**3),1),
            "ram_available_gb": round(mem.available/(1024**3),1),
            "ram_used_pct": mem.percent,
            "disk_total_gb": round(disk.total/(1024**3),1),
            "disk_free_gb": round(disk.free/(1024**3),1),
            "torch_version": hw.get("torch_version","N/A"),
        },
        "model_loaded": _model_id,
    }


# ── Model ─────────────────────────────────────────────────────────────────────

@app.post("/api/model/load")
async def load_model(body: dict):
    global _model, _model_id, _loading
    mid = body.get("model_id","").strip()
    if not mid: raise HTTPException(400, "model_id required")
    if _loading: raise HTTPException(409, "Already loading")
    _loading = True; _log(f"Loading: {mid}")
    try:
        from little_fig.model import FigLanguageModel
        _model = FigLanguageModel.from_pretrained(mid, hw=_get_hw())
        _model_id = mid; _log(f"✓ Loaded: {mid}")
        return {"status": "loaded", "model_id": mid}
    except Exception as e:
        _log(f"✗ Load failed: {e}"); raise HTTPException(500, str(e))
    finally: _loading = False


@app.post("/api/model/unload")
async def unload_model():
    global _model, _model_id
    old = _model_id; _model = None; _model_id = None; _log(f"Unloaded: {old}")
    return {"status": "unloaded"}


@app.get("/api/model/status")
async def model_status():
    return {"loaded": _model is not None, "model_id": _model_id, "loading": _loading}


# ── File Browser ──────────────────────────────────────────────────────────────

@app.get("/api/files/browse")
async def browse_files(path: str = Query(default=".")):
    try:
        target = Path(path).resolve()
        if not target.exists(): raise HTTPException(404, "Not found")
        entries = []
        if target.is_dir():
            for item in sorted(target.iterdir()):
                try:
                    is_dir = item.is_dir()
                    ext = item.suffix.lower()
                    is_model = ext in {'.gguf','.safetensors','.bin','.pt','.pth','.json','.yaml','.yml','.txt'}
                    if is_dir or is_model:
                        size = item.stat().st_size if not is_dir else 0
                        entries.append({"name":item.name,"path":str(item),"is_dir":is_dir,
                                        "size":size,"size_human":_human_size(size) if not is_dir else ""})
                except PermissionError: continue
        return {"current":str(target),"parent":str(target.parent),"entries":entries}
    except PermissionError: raise HTTPException(403,"Permission denied")


def _human_size(s):
    for u in ["B","KB","MB","GB"]:
        if s<1024: return f"{s:.1f} {u}" if u!="B" else f"{s} B"
        s/=1024
    return f"{s:.1f} TB"


# ── Chat History ──────────────────────────────────────────────────────────────

@app.get("/api/chats")
async def list_chats():
    chats = []
    for f in sorted(CHATS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            chats.append({"id": f.stem, "title": data.get("title","Untitled"),
                          "created": data.get("created",""), "message_count": len(data.get("messages",[]))})
        except: continue
    return {"chats": chats}


@app.post("/api/chats")
async def create_chat(body: dict = {}):
    chat_id = uuid.uuid4().hex[:12]
    data = {"id": chat_id, "title": body.get("title","New Chat"),
            "created": datetime.now().isoformat(), "messages": [], "model_id": _model_id}
    (CHATS_DIR / f"{chat_id}.json").write_text(json.dumps(data, indent=2))
    return data


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    p = CHATS_DIR / f"{chat_id}.json"
    if not p.exists(): raise HTTPException(404, "Chat not found")
    return json.loads(p.read_text())


@app.put("/api/chats/{chat_id}")
async def save_chat(chat_id: str, body: dict):
    p = CHATS_DIR / f"{chat_id}.json"
    body["id"] = chat_id
    p.write_text(json.dumps(body, indent=2))
    return {"status": "saved"}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    p = CHATS_DIR / f"{chat_id}.json"
    if p.exists(): p.unlink()
    return {"status": "deleted"}


# ── Ember Memory ──────────────────────────────────────────────────────────────

@app.post("/api/ember/generate-training-data")
async def generate_ember_training_data(body: dict = {}):
    """Generate memory-operation training data from Ember's cognitive modules."""
    n = body.get("n_examples", 500)
    _log(f"Generating {n} Ember memory training examples…")
    try:
        from little_fig.engine.ember_integration import EmberTrainingDataGenerator
        gen = EmberTrainingDataGenerator()
        output_path = str(Path(os.getcwd()) / "data" / "ember_memory_train.jsonl")
        Path(output_path).parent.mkdir(exist_ok=True)
        gen.generate_jsonl(n_examples=n, path=output_path)
        _log(f"✓ Generated {n} examples → {output_path}")
        return {"status": "generated", "path": output_path, "n_examples": n}
    except Exception as e:
        _log(f"✗ Generation failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/ember/status")
async def ember_status():
    """Check if Ember's Diaries is available."""
    try:
        from embers import EmberDB
        return {"available": True, "version": "installed"}
    except ImportError:
        return {"available": False, "install": "pip install embers-diaries"}


# ── Push to Hub ───────────────────────────────────────────────────────────────

@app.post("/api/model/push")
async def push_to_hub(body: dict):
    """Push the latest checkpoint or loaded model to HuggingFace Hub."""
    repo_id = body.get("repo_id", "").strip()
    if not repo_id:
        raise HTTPException(400, "repo_id required (e.g. 'username/my-model')")

    private = body.get("private", False)
    checkpoint_path = body.get("checkpoint_path", "").strip()
    merge = body.get("merge", True)

    _log(f"🚀 Pushing to Hub: {repo_id}")

    try:
        if checkpoint_path:
            # Push a specific checkpoint adapter
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id, exist_ok=True, private=private)
            api.upload_folder(
                folder_path=checkpoint_path,
                repo_id=repo_id,
                commit_message=f"Upload adapter from {os.path.basename(checkpoint_path)}",
            )
            _log(f"✓ Adapter pushed to https://huggingface.co/{repo_id}")
            return {"status": "pushed", "url": f"https://huggingface.co/{repo_id}", "type": "adapter"}
        else:
            raise HTTPException(400, "checkpoint_path required")

    except Exception as e:
        _log(f"✗ Push failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/model/export")
async def export_model(body: dict):
    """Merge LoRA adapter into base model and save locally."""
    base_model = body.get("base_model", "").strip()
    adapter_path = body.get("adapter_path", "").strip()
    output_name = body.get("output_name", "merged_model").strip()

    if not base_model or not adapter_path:
        raise HTTPException(400, "base_model and adapter_path required")

    output_dir = str(Path(os.getcwd()) / "merged_models" / output_name)

    _log(f"🔀 Merging adapter into {base_model}...")

    try:
        from little_fig.engine.model import FigModel
        from little_fig.engine.tier import TrainingTier

        fig = FigModel.from_pretrained(base_model, lora_r=16, lora_alpha=32)
        fig.load_adapter(adapter_path)
        fig.merge_and_export(output_dir)

        size_mb = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ) / (1024 ** 2)

        _log(f"✓ Merged model saved: {output_dir} ({size_mb:.0f} MB)")
        return {"status": "exported", "path": output_dir, "size_mb": round(size_mb)}

    except Exception as e:
        _log(f"✗ Export failed: {e}")
        raise HTTPException(500, str(e))


# ── Checkpoints ───────────────────────────────────────────────────────────────

@app.get("/api/checkpoints")
async def list_checkpoints():
    """List saved training checkpoints."""
    ckpt_dir = Path(os.getcwd()) / "checkpoints"
    if not ckpt_dir.exists():
        return {"checkpoints": []}
    checkpoints = []
    for d in sorted(ckpt_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir():
            meta_path = d / "training_state.json"
            meta = {}
            if meta_path.exists():
                try: meta = json.loads(meta_path.read_text())
                except: pass
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            checkpoints.append({
                "name": d.name, "path": str(d),
                "size_human": _human_size(size),
                "step": meta.get("step"), "tier": meta.get("tier"),
                "created": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
            })
    return {"checkpoints": checkpoints}


# ── Training ──────────────────────────────────────────────────────────────────

_training = False
_train_thread = None

@app.post("/api/train/start")
async def start_training(body: dict):
    """Start a training run in a background thread."""
    global _training, _train_thread
    if _training:
        raise HTTPException(409, "Training already in progress")

    model_id = body.get("model_id", "").strip()
    if not model_id:
        raise HTTPException(400, "model_id required")

    _training = True
    _log(f"🏋️ Starting training: {model_id}")

    def _run():
        global _training
        try:
            from little_fig.engine.model import FigModel
            from little_fig.engine.trainer import FigTrainer, FigTrainingConfig

            tier = body.get("tier", "auto")
            ember = body.get("ember_mode", False)
            dataset = body.get("dataset", "").strip()
            local_file = body.get("local_file", "").strip()

            config = FigTrainingConfig(
                tier=None if tier == "auto" else tier,
                num_epochs=int(body.get("epochs", 3)),
                batch_size=int(body.get("batch_size", 1)),
                learning_rate=float(body.get("learning_rate", "2e-4")),
                max_seq_length=int(body.get("max_seq_length", 512)),
                lora_r=int(body.get("lora_r", 16)),
                lora_alpha=int(body.get("lora_alpha", 32)),
            )

            _log(f"   Loading model with FigQuant...")
            fig_model = FigModel.from_pretrained(
                model_id,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                ember_mode=ember,
            )

            trainer = FigTrainer(fig_model, config)

            if ember:
                n = int(body.get("ember_examples", 500))
                _log(f"   Loading Ember dataset ({n} examples)...")
                trainer.load_ember_dataset(n_examples=n)
            elif local_file:
                _log(f"   Loading local dataset: {local_file}")
                trainer.load_dataset(local_file)
            elif dataset:
                _log(f"   Loading HF dataset: {dataset}")
                trainer.load_dataset(dataset)
            else:
                _log("✗ No dataset specified")
                _training = False
                return

            _log("   Training started...")
            trainer.train()
            _log("✓ Training complete!")

        except Exception as e:
            _log(f"✗ Training error: {e}")
        finally:
            _training = False

    _train_thread = threading.Thread(target=_run, daemon=True)
    _train_thread.start()
    return {"status": "started", "model_id": model_id}


@app.get("/api/train/status")
async def train_status():
    return {"training": _training}


@app.post("/api/train/stop")
async def stop_training():
    global _training
    _training = False
    _log("⚠ Training stop requested")
    return {"status": "stop_requested"}


# ── Benchmarks ────────────────────────────────────────────────────────────────

@app.post("/api/bench/run")
async def run_benchmarks():
    """Run comprehensive benchmarks: FigQuant, all FigKernels, inference, training, memory."""
    _log("📊 Running full benchmark suite...")
    results = {}

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import time as _t
        import psutil
        import gc
        from little_fig.engine.figquant import figquant_quantize, figquant_dequantize, measure_quality
        from little_fig.engine.figkernel import (
            FigRMSNorm, FigSwiGLU, FigCrossEntropy,
            fig_fused_linear_lora, fig_chunked_cross_entropy,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ─── 1. FigQuant Quality ─────────────────────────────────────────────
        _log("   [1/5] FigQuant quality...")
        W = torch.randn(2048, 768)
        fq = figquant_quantize(W, group_size=128, n_iters=8)
        qual = measure_quality(W, fq)
        results["figquant"] = {
            "cosine_similarity": round(qual["cosine_similarity"], 6),
            "mse": round(qual["mse"], 6),
            "snr_db": round(qual["snr_db"], 1),
            "bits_per_param": round(qual["bits_per_param"], 2),
            "compression_ratio": round(qual["compression_ratio"], 1),
        }

        # ─── 2. All FigKernels ───────────────────────────────────────────────
        _log("   [2/5] FigKernel benchmarks...")
        hidden = 2048
        inter = 5504
        batch, seq = 4, 256

        # 2a. RMSNorm
        x = torch.randn(batch, seq, hidden, device=device)
        weight = torch.ones(hidden, device=device)
        norm = FigRMSNorm(hidden).to(device)

        times = []
        for _ in range(5):
            t0 = _t.time()
            for _ in range(20):
                _ = x.pow(2).mean(-1, keepdim=True)
                _ = x * torch.rsqrt(_ + 1e-6) * weight
            times.append((_t.time() - t0) / 20 * 1000)
        rmsnorm_std_ms = sum(times) / len(times)

        times = []
        for _ in range(5):
            t0 = _t.time()
            for _ in range(20):
                _ = norm(x)
            times.append((_t.time() - t0) / 20 * 1000)
        rmsnorm_fig_ms = sum(times) / len(times)

        # 2b. SwiGLU
        swiglu = FigSwiGLU(hidden, inter).to(device)
        x_mlp = torch.randn(batch, seq, hidden, device=device)

        # Standard SwiGLU
        gate_w = torch.randn(inter, hidden, device=device)
        up_w = torch.randn(inter, hidden, device=device)
        down_w = torch.randn(hidden, inter, device=device)
        times = []
        for _ in range(5):
            t0 = _t.time()
            for _ in range(10):
                g = F.silu(F.linear(x_mlp, gate_w))
                u = F.linear(x_mlp, up_w)
                _ = F.linear(g * u, down_w)
            times.append((_t.time() - t0) / 10 * 1000)
        swiglu_std_ms = sum(times) / len(times)

        times = []
        for _ in range(5):
            t0 = _t.time()
            for _ in range(10):
                _ = swiglu(x_mlp)
            times.append((_t.time() - t0) / 10 * 1000)
        swiglu_fig_ms = sum(times) / len(times)

        # 2c. CrossEntropy (chunked vs standard)
        vocab_size = 32000
        n_tokens = batch * seq
        hidden_flat = torch.randn(n_tokens, hidden, device=device)
        lm_head = torch.randn(vocab_size, hidden, device=device)
        targets = torch.randint(0, vocab_size, (n_tokens,), device=device)

        times = []
        for _ in range(3):
            t0 = _t.time()
            logits = F.linear(hidden_flat, lm_head)
            _ = F.cross_entropy(logits, targets)
            times.append((_t.time() - t0) * 1000)
        ce_std_ms = sum(times) / len(times)

        times = []
        for _ in range(3):
            t0 = _t.time()
            _ = fig_chunked_cross_entropy(hidden_flat, lm_head, targets, chunk_size=8192)
            times.append((_t.time() - t0) * 1000)
        ce_fig_ms = sum(times) / len(times)

        # Peak memory for standard vs chunked CE
        gc.collect()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            logits = F.linear(hidden_flat, lm_head)
            _ = F.cross_entropy(logits, targets)
            ce_std_mem_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
            _ = fig_chunked_cross_entropy(hidden_flat, lm_head, targets, chunk_size=8192)
            ce_fig_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            # Estimate: standard materializes [n_tokens, vocab] float32
            ce_std_mem_mb = n_tokens * vocab_size * 4 / 1e6
            ce_fig_mem_mb = n_tokens * 8192 * 4 / 1e6  # only one chunk at a time

        # 2d. Fused Linear+LoRA
        W_base = torch.randn(hidden, hidden, device=device)
        lora_A = torch.randn(hidden, 16, device=device)
        lora_B = torch.randn(16, hidden, device=device)
        x_lin = torch.randn(batch, seq, hidden, device=device)

        times = []
        for _ in range(5):
            t0 = _t.time()
            for _ in range(20):
                h = F.linear(x_lin, W_base)
                h = h + (x_lin @ lora_A) @ lora_B * (32 / 16)
            times.append((_t.time() - t0) / 20 * 1000)
        lora_std_ms = sum(times) / len(times)

        times = []
        for _ in range(5):
            t0 = _t.time()
            for _ in range(20):
                _ = fig_fused_linear_lora(x_lin, W_base, lora_A, lora_B, 32 / 16)
            times.append((_t.time() - t0) / 20 * 1000)
        lora_fig_ms = sum(times) / len(times)

        results["figkernel"] = {
            "rmsnorm_standard_ms": round(rmsnorm_std_ms, 3),
            "rmsnorm_fig_ms": round(rmsnorm_fig_ms, 3),
            "rmsnorm_speedup": round(rmsnorm_std_ms / max(rmsnorm_fig_ms, 0.001), 2),
            "swiglu_standard_ms": round(swiglu_std_ms, 3),
            "swiglu_fig_ms": round(swiglu_fig_ms, 3),
            "swiglu_speedup": round(swiglu_std_ms / max(swiglu_fig_ms, 0.001), 2),
            "crossentropy_standard_ms": round(ce_std_ms, 2),
            "crossentropy_fig_ms": round(ce_fig_ms, 2),
            "crossentropy_speedup": round(ce_std_ms / max(ce_fig_ms, 0.01), 2),
            "crossentropy_std_mem_mb": round(ce_std_mem_mb, 1),
            "crossentropy_fig_mem_mb": round(ce_fig_mem_mb, 1),
            "crossentropy_mem_savings": round(ce_std_mem_mb / max(ce_fig_mem_mb, 0.01), 1),
            "linear_lora_standard_ms": round(lora_std_ms, 3),
            "linear_lora_fig_ms": round(lora_fig_ms, 3),
            "linear_lora_speedup": round(lora_std_ms / max(lora_fig_ms, 0.001), 2),
        }

        # ─── 3. Inference Speed ──────────────────────────────────────────────
        _log("   [3/5] Inference speed...")
        inference_results = {}
        if _model is not None:
            try:
                tokenizer = _model.tokenizer
                test_prompt = "The meaning of life is"
                inputs = tokenizer(test_prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)

                # Warmup
                with torch.no_grad():
                    _model.model.eval()
                    _ = _model.model.generate(input_ids, max_new_tokens=5, do_sample=False)

                # Timed generation
                gen_lengths = [32, 64, 128]
                for length in gen_lengths:
                    t0 = _t.time()
                    with torch.no_grad():
                        out = _model.model.generate(input_ids, max_new_tokens=length, do_sample=False)
                    elapsed = _t.time() - t0
                    n_generated = out.shape[1] - input_ids.shape[1]
                    tok_per_sec = n_generated / max(elapsed, 0.001)
                    inference_results[f"gen_{length}_tok_per_sec"] = round(tok_per_sec, 1)
                    inference_results[f"gen_{length}_time_s"] = round(elapsed, 3)

                # Time to first token (TTFT)
                t0 = _t.time()
                with torch.no_grad():
                    _ = _model.model.generate(input_ids, max_new_tokens=1, do_sample=False)
                inference_results["ttft_ms"] = round((_t.time() - t0) * 1000, 1)

            except Exception as e:
                inference_results["error"] = str(e)
        else:
            inference_results["error"] = "No model loaded — load a model to benchmark inference"

        results["inference"] = inference_results

        # ─── 4. Training Throughput ──────────────────────────────────────────
        _log("   [4/5] Training throughput...")
        training_results = {}
        if _model is not None:
            try:
                tokenizer = _model.tokenizer
                # Create a small batch of training data
                texts = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning models require large amounts of data.",
                    "GPU acceleration significantly speeds up neural network training.",
                    "Little Fig trains LLMs on CPU with minimal memory overhead.",
                ]
                batch_enc = tokenizer(
                    texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=128
                )
                input_ids = batch_enc["input_ids"].to(device)
                attention_mask = batch_enc["attention_mask"].to(device)

                _model.model.train()
                # Enable grad for LoRA params
                optimizer = torch.optim.AdamW(
                    [p for p in _model.parameters() if p.requires_grad],
                    lr=2e-4
                )

                # Warmup step
                outputs = _model.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                outputs.loss.backward()
                optimizer.zero_grad()

                # Timed training steps
                n_steps = 5
                t0 = _t.time()
                total_tokens = 0
                for step in range(n_steps):
                    outputs = _model.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    outputs.loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_tokens += attention_mask.sum().item()
                elapsed = _t.time() - t0

                training_results["steps"] = n_steps
                training_results["total_time_s"] = round(elapsed, 2)
                training_results["step_time_ms"] = round(elapsed / n_steps * 1000, 1)
                training_results["tokens_per_sec"] = round(total_tokens / elapsed, 1)
                training_results["samples_per_sec"] = round(n_steps * len(texts) / elapsed, 2)
                training_results["last_loss"] = round(outputs.loss.item(), 4)

                _model.model.eval()
            except Exception as e:
                training_results["error"] = str(e)
        else:
            training_results["error"] = "No model loaded — load a model to benchmark training"

        results["training"] = training_results

        # ─── 5. Memory Usage (actual measurement) ────────────────────────────
        _log("   [5/5] Memory profiling...")
        mem = psutil.virtual_memory()
        memory_results = {
            "system_ram_total_gb": round(mem.total / 1e9, 1),
            "system_ram_used_gb": round(mem.used / 1e9, 1),
            "system_ram_available_gb": round(mem.available / 1e9, 1),
            "system_ram_percent": mem.percent,
        }

        if device == "cuda":
            memory_results["gpu_vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
            memory_results["gpu_vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            memory_results["gpu_vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
            memory_results["gpu_vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

        if _model is not None:
            # Measure actual model memory footprint
            model_params_bytes = sum(
                p.numel() * p.element_size() for p in _model.parameters()
            )
            trainable_bytes = sum(
                p.numel() * p.element_size() for p in _model.parameters() if p.requires_grad
            )
            frozen_bytes = model_params_bytes - trainable_bytes

            memory_results["model_total_mb"] = round(model_params_bytes / 1e6, 1)
            memory_results["model_trainable_mb"] = round(trainable_bytes / 1e6, 1)
            memory_results["model_frozen_mb"] = round(frozen_bytes / 1e6, 1)
            memory_results["model_name"] = _model_id or "unknown"
            memory_results["trainable_params"] = sum(p.numel() for p in _model.parameters() if p.requires_grad)
            memory_results["total_params"] = sum(p.numel() for p in _model.parameters())

            # Estimate optimizer states (AdamW = 2x param size for momentum + variance)
            optimizer_est_mb = trainable_bytes * 2 / 1e6
            memory_results["optimizer_est_mb"] = round(optimizer_est_mb, 1)
            memory_results["total_training_est_mb"] = round(
                model_params_bytes / 1e6 + optimizer_est_mb, 1
            )
        else:
            memory_results["model_loaded"] = False

        results["memory"] = memory_results

        _log(f"✓ Full benchmark suite complete")

    except Exception as e:
        _log(f"✗ Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

    return results


# ── CogMemBench ──────────────────────────────────────────────────────────────

_cogmem_running = False
_cogmem_results = None

@app.post("/api/bench/cogmem")
async def run_cogmembench(body: dict = {}):
    """Run CogMemBench against the currently loaded model."""
    global _cogmem_running, _cogmem_results
    if _cogmem_running:
        raise HTTPException(409, "CogMemBench already running")
    if _model is None:
        raise HTTPException(400, "Load a model first")

    per_axis = int(body.get("per_axis", 20))
    max_cases = int(body.get("max_cases", 100))

    _cogmem_running = True
    _log(f"🧠 Running CogMemBench ({per_axis}/axis, max {max_cases})...")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "cogmembench"))
        from cogmembench import CogMemRunner

        runner = CogMemRunner(per_axis=per_axis)

        def model_fn(prompt):
            return _model.generate(prompt, max_new_tokens=200)

        results = runner.run(model_fn=model_fn, max_cases=max_cases, verbose=False)
        _cogmem_results = results
        _log(f"✓ CogMemBench complete: Score={results['cogmem_score']}/100")
        return results
    except Exception as e:
        _log(f"✗ CogMemBench error: {e}")
        raise HTTPException(500, str(e))
    finally:
        _cogmem_running = False


@app.get("/api/bench/cogmem/status")
async def cogmem_status():
    return {"running": _cogmem_running, "results": _cogmem_results}
    return {"logs": list(_log_buffer)[-n:]}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    name = f"{uuid.uuid4().hex[:8]}{Path(file.filename).suffix}"
    fpath = UPLOAD_DIR / name
    fpath.write_bytes(await file.read())
    return {"filename": file.filename, "path": str(fpath), "size": fpath.stat().st_size}


# ── Chat WebSocket ────────────────────────────────────────────────────────────

@app.websocket("/api/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            msg = data.get("message","").strip()
            history = data.get("history",[])
            sys_prompt = data.get("system_prompt","")
            temp = float(data.get("temperature",0.7))
            max_tok = int(data.get("max_tokens",512))

            if not msg:
                await ws.send_json({"type":"error","content":"Empty message"}); continue
            if _model is None:
                await ws.send_json({"type":"error","content":"No model loaded"}); continue

            _model.config.max_new_tokens = max_tok
            _model.config.temperature = temp
            norm = [{"role":m.get("role","user"),"content":m.get("content","")} for m in history if isinstance(m.get("content"),str)]
            prompt = _model.apply_chat_template(msg, norm)
            if sys_prompt.strip(): prompt = f"System: {sys_prompt}\n\n" + prompt

            await ws.send_json({"type":"start"})
            full=""; t0=time.time()
            try:
                for chunk in _model.stream(prompt):
                    full += chunk
                    await ws.send_json({"type":"chunk","content":full})
                    await asyncio.sleep(0)
                elapsed = time.time()-t0
                await ws.send_json({"type":"done","content":full,"stats":{"words":len(full.split()),"time_s":round(elapsed,1),"tok_per_s":round(len(full.split())/max(elapsed,0.1),1)}})
                _log(f"Chat: {len(full.split())}w in {elapsed:.1f}s")
            except Exception as e:
                await ws.send_json({"type":"error","content":str(e)}); _log(f"Chat error: {e}")
    except WebSocketDisconnect: pass


# ── Terminal WebSocket ────────────────────────────────────────────────────────

@app.websocket("/api/terminal")
async def terminal_ws(ws: WebSocket):
    """Bidirectional terminal: sends log stream, accepts commands."""
    await ws.accept()
    last_sent = 0
    try:
        while True:
            # Send new logs
            logs = list(_log_buffer)
            if len(logs) > last_sent:
                new_logs = logs[last_sent:]
                for line in new_logs:
                    await ws.send_json({"type":"log","content":line})
                last_sent = len(logs)

            # Check for incoming commands (non-blocking)
            try:
                data = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                cmd = data.get("command","").strip()
                if cmd:
                    _log(f"$ {cmd}")
                    try:
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd())
                        for line in (result.stdout + result.stderr).strip().split("\n"):
                            if line: _log(line)
                    except subprocess.TimeoutExpired:
                        _log("Command timed out (30s)")
                    except Exception as e:
                        _log(f"Error: {e}")
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect: pass


# ── Runner ────────────────────────────────────────────────────────────────────

def run_server(host="0.0.0.0", port=8888):
    import uvicorn
    print(f"🍐 Little Fig Web → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
