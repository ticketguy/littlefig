"""
Little Fig — Unified Model Engine (v0.4.1)

Loads models from ANY source:
    1. Local GGUF files    → via llama-cpp-python (if installed) or llama-server
    2. Local HF folders    → downloaded model directories
    3. HuggingFace Hub     → auto-downloads from hub
    4. Fig Engine INT4     → 7.1× less RAM, pure PyTorch, always works

Loading priority for local .gguf files:
    Try llama-cpp-python → if fails, tell user how to fix → offer Fig Engine alternative

Loading modes for HF models:
    FP32  → standard HuggingFace (small models / enough RAM)
    INT4  → Fig Engine streaming (large models / limited RAM)
"""

import os
import glob
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from threading import Thread
from typing import Iterator, Optional
from dataclasses import dataclass


# ── Generation config ─────────────────────────────────────────────────────────

@dataclass
class FigInferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


# ── Source detection ──────────────────────────────────────────────────────────

def _detect_source(model_name_or_path: str) -> str:
    """
    Detect the model source type:
        "gguf"       → local .gguf file
        "gguf_dir"   → directory containing .gguf files
        "local_hf"   → local directory with HF model files (config.json etc.)
        "hub"        → HuggingFace Hub model ID
    """
    if os.path.isfile(model_name_or_path):
        if model_name_or_path.lower().endswith(".gguf"):
            return "gguf"
        return "local_hf"  # Assume it's a single-file model

    if os.path.isdir(model_name_or_path):
        # Check for GGUF files in directory
        gguf_files = glob.glob(os.path.join(model_name_or_path, "*.gguf"))
        if gguf_files:
            return "gguf_dir"
        # Check for HF model files
        if os.path.exists(os.path.join(model_name_or_path, "config.json")):
            return "local_hf"
        # Could be a GGUF directory without config.json
        if gguf_files:
            return "gguf_dir"

    # Default: treat as HuggingFace Hub model ID
    return "hub"


def _find_gguf_in_dir(directory: str) -> Optional[str]:
    """Find the best GGUF file in a directory (prefer Q4_K_M, then largest)."""
    gguf_files = glob.glob(os.path.join(directory, "*.gguf"))
    if not gguf_files:
        return None

    # Prefer Q4_K_M (best quality/size tradeoff)
    for f in gguf_files:
        if "Q4_K_M" in f.upper():
            return f

    # Otherwise return the largest file (usually the highest quality)
    return max(gguf_files, key=os.path.getsize)


# ── Unified Model ─────────────────────────────────────────────────────────────

class FigLanguageModel:
    """
    Unified inference model for Little Fig.

    Loads from any source:
        model = FigLanguageModel.from_pretrained("gpt2")                        # Hub
        model = FigLanguageModel.from_pretrained("./models/tinyllama/")         # Local HF folder
        model = FigLanguageModel.from_pretrained("./models/gemma-4b-Q4.gguf")  # Local GGUF
        model = FigLanguageModel.from_pretrained("./models/")                   # Dir with GGUF files

    Modes:
        use_int4=True   → Fig Engine INT4 (7.1× less RAM)
        use_int4=False  → Full FP32 HuggingFace
        use_int4=None   → Auto-detect based on available RAM
    """

    def __init__(self, model, tokenizer, model_name: str = "",
                 backend: str = "hf"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = FigInferenceConfig()
        self.backend = backend          # "hf" or "fig_engine"

        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        hw: Optional[dict] = None,
        use_int4: Optional[bool] = None,
    ) -> "FigLanguageModel":
        """
        Load a model from any source.

        Args:
            model_name_or_path: One of:
                - HuggingFace model ID: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                - Local path to .gguf file: "./models/gemma-4b-Q4_K_M.gguf"
                - Local directory with .gguf files: "./models/"
                - Local HF model folder: "./my_model/"
            hw: Hardware config (auto-detected if None)
            use_int4: Force INT4 mode for HF models (None = auto)
        """
        if hw is None:
            from little_fig import HW
            hw = HW

        source = _detect_source(model_name_or_path)

        # ── GGUF file ────────────────────────────────────────────────────
        if source == "gguf":
            return FigLanguageModel._load_gguf(model_name_or_path, hw)

        # ── Directory with GGUF files ────────────────────────────────────
        if source == "gguf_dir":
            gguf_path = _find_gguf_in_dir(model_name_or_path)
            if gguf_path:
                return FigLanguageModel._load_gguf(gguf_path, hw)

        # ── Local HF folder or Hub model ─────────────────────────────────
        gpu = hw.get("gpu_available", False)

        # GPU: always use standard HF loading (FP16/BF16, leverage GPU VRAM)
        # CPU: use Fig Engine INT4 for large models, FP32 for small ones
        if use_int4 is None:
            if gpu:
                use_int4 = False  # GPU has its own optimizations
            else:
                ram_gb = hw.get("ram_available_gb", 16)
                large_indicators = ["4b", "7b", "8b", "13b", "70b", "gemma-3", "gemma-4"]
                name_lower = model_name_or_path.lower()
                large_model = any(x in name_lower for x in large_indicators)
                use_int4 = ram_gb < 12 or large_model

        if use_int4:
            return FigLanguageModel._load_int4(model_name_or_path, hw)
        else:
            return FigLanguageModel._load_fp32(model_name_or_path, hw)

    @staticmethod
    def _load_gguf(path: str, hw: dict) -> "FigLanguageModel":
        """
        Load from GGUF file using Fig Engine's native GGUF reader.
        No llama-cpp-python needed. No model substitution.
        Loads exactly the model in the file — whatever architecture it is.
        """
        from little_fig.engine.gguf_loader import load_gguf_as_fig_model

        print(f"🍐 Loading GGUF: {os.path.basename(path)}")
        print(f"   Using universal GGUF loader (auto-detects any architecture)")

        fig_model = load_gguf_as_fig_model(path, lora_r=0, lora_alpha=0)

        model = fig_model.model
        if model is not None:
            model.eval()

        return FigLanguageModel(
            model=model,
            tokenizer=fig_model.tokenizer,
            model_name=os.path.basename(path),
            backend="fig_engine",
        )

    @staticmethod
    def _load_fp32(model_name: str, hw: dict) -> "FigLanguageModel":
        """
        Load model in native precision.
        GPU: FP16 or BF16 (fast, leverages GPU VRAM).
        CPU: FP32 (most compatible).
        """
        gpu = hw.get("gpu_available", False)

        if gpu:
            # Prefer BF16 on Ampere+ (compute capability 8.0+), else FP16
            try:
                props = torch.cuda.get_device_properties(0)
                dtype = torch.bfloat16 if props.major >= 8 else torch.float16
            except Exception:
                dtype = torch.float16
            dtype_label = "BF16" if dtype == torch.bfloat16 else "FP16"
        else:
            dtype = torch.float32
            dtype_label = "FP32"

        print(f"🍐 Loading: {model_name} ({dtype_label})")
        print(f"   Device : {'GPU (' + hw.get('gpu_name', '') + ')' if gpu else 'CPU'}")

        load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        try:
            load_kwargs["device_map"] = "auto" if gpu else "cpu"
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except (ImportError, ValueError):
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"   ✓ {param_count:.2f}B parameters loaded ({dtype_label})")

        return FigLanguageModel(model, tokenizer, model_name, backend="hf")

    @staticmethod
    def _load_int4(model_name: str, hw: dict) -> "FigLanguageModel":
        """Load model with Fig Engine INT4 quantization."""
        from little_fig.engine.model import FigModel

        print(f"🍐 Loading: {model_name} (Fig Engine INT4)")

        fig_model = FigModel.from_pretrained(
            model_name,
            lora_r=0,
            lora_alpha=0,
        )
        fig_model.model.eval()

        return FigLanguageModel(
            fig_model.model, fig_model.tokenizer, model_name, backend="fig_engine"
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Generate a complete response."""
        inputs = self._encode(prompt)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        input_length = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    def stream(self, prompt: str) -> Iterator[str]:
        """Generate response with streaming."""
        inputs = self._encode(prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join()

    def _encode(self, prompt: str) -> dict:
        device = next(self.model.parameters()).device
        return self.tokenizer(
            prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(device)

    def apply_chat_template(self, message: str, history: list) -> str:
        """
        Build a chat prompt. Handles:
            - Messages format: [{"role": "user", "content": "..."}]
            - Legacy tuple format: [("user msg", "bot msg")]
            - GGUF models (no tokenizer — uses generic template)
        """
        messages = []
        for msg in history:
            if isinstance(msg, dict):
                messages.append(msg)
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                user_msg, bot_msg = msg
                if user_msg:
                    messages.append({"role": "user", "content": str(user_msg)})
                if bot_msg:
                    messages.append({"role": "assistant", "content": str(bot_msg)})
        messages.append({"role": "user", "content": message})

        # Use tokenizer's chat template if available
        if self.tokenizer and hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass

        # GGUF or fallback: detect model family from filename
        name_lower = self.model_name.lower()

        if "gemma" in name_lower:
            prompt = ""
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                prompt += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"
            return prompt

        if any(x in name_lower for x in ["llama", "mistral"]):
            # Llama 3 / Mistral chat format
            prompt = "<|begin_of_text|>"
            for msg in messages:
                prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return prompt

        if "chatml" in name_lower or "tinyllama" in name_lower or "qwen" in name_lower:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            for msg in messages:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt

        # Generic fallback
        prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            prompt += f"{role}: {msg['content']}\n"
        prompt += "Assistant:"
        return prompt

