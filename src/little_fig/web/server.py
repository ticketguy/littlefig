"""
Little Fig — FastAPI Backend

REST + WebSocket API for the custom frontend.
Endpoints:
  GET  /api/health          — server status + hardware info
  POST /api/model/load      — load a model (HF ID or local path)
  POST /api/model/unload    — unload current model
  GET  /api/model/status    — current model info
  WS   /api/chat            — streaming chat via WebSocket
  POST /api/upload           — file upload for multimodal
"""

import os
import sys
import json
import time
import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ── Global state ──────────────────────────────────────────────────────────────

_model = None
_model_id = None
_hw = None
_loading = False

UPLOAD_DIR = os.path.join(os.getcwd(), ".fig_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
    return JSONResponse({"error": "Frontend not built. Run: cd web && npm run build"})


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    hw = _get_hw()
    return {
        "status": "ok",
        "version": "0.5.0",
        "hardware": {
            "device": hw.get("device", "cpu"),
            "gpu": hw.get("gpu_name") if hw.get("gpu_available") else None,
            "gpu_vram_gb": hw.get("gpu_vram_gb"),
            "cpu_cores": hw.get("cpu_cores"),
            "ram_total_gb": hw.get("ram_total_gb"),
            "ram_available_gb": hw.get("ram_available_gb"),
        },
        "model_loaded": _model_id,
    }


@app.post("/api/model/load")
async def load_model(body: dict):
    global _model, _model_id, _loading

    model_id = body.get("model_id", "").strip()
    if not model_id:
        raise HTTPException(400, "model_id required")

    if _loading:
        raise HTTPException(409, "A model is already loading")

    _loading = True
    try:
        from little_fig.model import FigLanguageModel
        hw = _get_hw()
        _model = FigLanguageModel.from_pretrained(model_id, hw=hw)
        _model_id = model_id
        return {"status": "loaded", "model_id": model_id}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        _loading = False


@app.post("/api/model/unload")
async def unload_model():
    global _model, _model_id
    _model = None
    _model_id = None
    return {"status": "unloaded"}


@app.get("/api/model/status")
async def model_status():
    return {
        "loaded": _model is not None,
        "model_id": _model_id,
        "loading": _loading,
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix
    name = f"{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    return {"filename": file.filename, "path": path, "size": len(content)}


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

            # Configure
            _model.config.max_new_tokens = max_tokens
            _model.config.temperature = temperature

            # Build prompt
            normalised = []
            for m in history:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, str):
                    normalised.append({"role": role, "content": content})

            prompt = _model.apply_chat_template(message, normalised)
            if system_prompt.strip():
                prompt = f"System: {system_prompt}\n\n" + prompt

            # Stream response
            await ws.send_json({"type": "start"})
            full = ""
            t0 = time.time()
            try:
                for chunk in _model.stream(prompt):
                    full += chunk
                    await ws.send_json({"type": "chunk", "content": full})
                    await asyncio.sleep(0)  # Yield to event loop
                elapsed = time.time() - t0
                words = len(full.split())
                await ws.send_json({
                    "type": "done",
                    "content": full,
                    "stats": {"words": words, "time_s": round(elapsed, 1)},
                })
            except Exception as e:
                await ws.send_json({"type": "error", "content": str(e)})

    except WebSocketDisconnect:
        pass


# ── Runner ────────────────────────────────────────────────────────────────────

def run_server(host="0.0.0.0", port=8888):
    """Run the Little Fig web server."""
    import uvicorn
    print(f"🍐 Little Fig Web → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
