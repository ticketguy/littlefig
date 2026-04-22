"""
Little Fig — FastAPI Backend v2

REST + WebSocket API:
  GET  /api/health          — hardware info (detailed)
  POST /api/model/load      — load HF model or local path
  POST /api/model/unload    — unload current model
  GET  /api/model/status    — current model info
  GET  /api/files/browse    — browse local directories for models
  GET  /api/logs            — server log tail
  WS   /api/chat            — streaming chat via WebSocket
  POST /api/upload          — file upload
"""

import os
import sys
import json
import time
import asyncio
import uuid
import glob
import platform
import collections
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ── Global state ──────────────────────────────────────────────────────────────

_model = None
_model_id = None
_hw = None
_loading = False
_log_buffer = collections.deque(maxlen=200)  # Ring buffer for logs

UPLOAD_DIR = os.path.join(os.getcwd(), ".fig_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _log(msg: str):
    """Add to log buffer + print."""
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
            _hw = {
                "device": "cpu",
                "gpu_available": False,
                "cpu_cores": os.cpu_count(),
                "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
            }
    return _hw


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_hw()
    _log("Little Fig server started")
    yield

app = FastAPI(title="Little Fig", version="0.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
STATIC_DIR = Path(__file__).parent / "static"
ASSETS_DIR = STATIC_DIR / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"error": "Frontend not found"}, status_code=500)


@app.get("/favicon.ico")
async def favicon():
    # Return a simple empty response (no 404 spam in logs)
    return JSONResponse(content={}, status_code=204)


# ── Health (detailed) ─────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    import psutil
    hw = _get_hw()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())
    return {
        "status": "ok",
        "version": "0.5.0",
        "platform": {
            "os": platform.system(),
            "os_version": platform.version(),
            "python": sys.version.split()[0],
            "arch": platform.machine(),
        },
        "hardware": {
            "device": hw.get("device", "cpu"),
            "gpu": hw.get("gpu_name") if hw.get("gpu_available") else None,
            "gpu_vram_gb": hw.get("gpu_vram_gb"),
            "cpu_cores": hw.get("cpu_cores"),
            "cpu_name": platform.processor() or "Unknown",
            "ram_total_gb": round(mem.total / (1024**3), 1),
            "ram_available_gb": round(mem.available / (1024**3), 1),
            "ram_used_pct": mem.percent,
            "disk_total_gb": round(disk.total / (1024**3), 1),
            "disk_free_gb": round(disk.free / (1024**3), 1),
            "torch_version": hw.get("torch_version", "N/A"),
        },
        "model_loaded": _model_id,
    }


# ── Model ─────────────────────────────────────────────────────────────────────

@app.post("/api/model/load")
async def load_model(body: dict):
    global _model, _model_id, _loading
    model_id = body.get("model_id", "").strip()
    if not model_id:
        raise HTTPException(400, "model_id required")
    if _loading:
        raise HTTPException(409, "A model is already loading")

    _loading = True
    _log(f"Loading model: {model_id}")
    try:
        from little_fig.model import FigLanguageModel
        hw = _get_hw()
        _model = FigLanguageModel.from_pretrained(model_id, hw=hw)
        _model_id = model_id
        _log(f"✓ Model loaded: {model_id}")
        return {"status": "loaded", "model_id": model_id}
    except Exception as e:
        _log(f"✗ Model load failed: {e}")
        raise HTTPException(500, str(e))
    finally:
        _loading = False


@app.post("/api/model/unload")
async def unload_model():
    global _model, _model_id
    old = _model_id
    _model = None
    _model_id = None
    _log(f"Model unloaded: {old}")
    return {"status": "unloaded"}


@app.get("/api/model/status")
async def model_status():
    return {
        "loaded": _model is not None,
        "model_id": _model_id,
        "loading": _loading,
    }


# ── File Browser ──────────────────────────────────────────────────────────────

@app.get("/api/files/browse")
async def browse_files(path: str = Query(default=".")):
    """Browse local filesystem for model files (GGUF, safetensors, etc.)."""
    try:
        target = Path(path).resolve()
        if not target.exists():
            raise HTTPException(404, f"Path not found: {path}")

        entries = []
        if target.is_dir():
            for item in sorted(target.iterdir()):
                try:
                    is_dir = item.is_dir()
                    size = item.stat().st_size if not is_dir else 0
                    name = item.name
                    # Filter: show dirs + model-related files
                    is_model = any(name.endswith(ext) for ext in [
                        ".gguf", ".safetensors", ".bin", ".pt", ".pth",
                        ".json", ".txt", ".yaml", ".yml",
                    ])
                    if is_dir or is_model:
                        entries.append({
                            "name": name,
                            "path": str(item),
                            "is_dir": is_dir,
                            "size": size,
                            "size_human": _human_size(size) if not is_dir else "",
                        })
                except PermissionError:
                    continue

        return {
            "current": str(target),
            "parent": str(target.parent),
            "entries": entries,
        }
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    except Exception as e:
        raise HTTPException(500, str(e))


def _human_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} TB"


# ── Logs ──────────────────────────────────────────────────────────────────────

@app.get("/api/logs")
async def get_logs(n: int = Query(default=50, le=200)):
    """Return last N log lines."""
    return {"logs": list(_log_buffer)[-n:]}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix
    name = f"{uuid.uuid4().hex[:8]}{ext}"
    fpath = os.path.join(UPLOAD_DIR, name)
    content = await file.read()
    with open(fpath, "wb") as f:
        f.write(content)
    return {"filename": file.filename, "path": fpath, "size": len(content)}


# ── WebSocket Chat ────────────────────────────────────────────────────────────

@app.websocket("/api/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            message = data.get("message", "").strip()
            history = data.get("history", [])
            system_prompt = data.get("system_prompt", "")
            temperature = float(data.get("temperature", 0.7))
            max_tokens = int(data.get("max_tokens", 512))

            if not message:
                await ws.send_json({"type": "error", "content": "Empty message"})
                continue

            if _model is None:
                await ws.send_json({"type": "error", "content": "No model loaded. Load a model first."})
                continue

            _model.config.max_new_tokens = max_tokens
            _model.config.temperature = temperature

            normalised = []
            for m in history:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, str):
                    normalised.append({"role": role, "content": content})

            prompt = _model.apply_chat_template(message, normalised)
            if system_prompt.strip():
                prompt = f"System: {system_prompt}\n\n" + prompt

            await ws.send_json({"type": "start"})
            full = ""
            t0 = time.time()
            try:
                for chunk in _model.stream(prompt):
                    full += chunk
                    await ws.send_json({"type": "chunk", "content": full})
                    await asyncio.sleep(0)
                elapsed = time.time() - t0
                words = len(full.split())
                await ws.send_json({
                    "type": "done",
                    "content": full,
                    "stats": {"words": words, "time_s": round(elapsed, 1)},
                })
                _log(f"Chat: {words} words in {elapsed:.1f}s")
            except Exception as e:
                await ws.send_json({"type": "error", "content": str(e)})
                _log(f"Chat error: {e}")

    except WebSocketDisconnect:
        pass


# ── Runner ────────────────────────────────────────────────────────────────────

def run_server(host="0.0.0.0", port=8888):
    import uvicorn
    print(f"🍐 Little Fig Web → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
