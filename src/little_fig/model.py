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
from typing import Iterator, Optional, Union
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


# ── GGUF Backend (optional, graceful) ─────────────────────────────────────────

class _GGUFBackend:
    """
    Wraps llama-cpp-python for GGUF inference.
    Gracefully handles missing library or unsupported architectures.
    """

    def __init__(self, llm, model_path: str):
        self._llm = llm
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)

    @staticmethod
    def load(path: str, n_ctx: int = 4096, hw: Optional[dict] = None) -> "_GGUFBackend":
        """
        Load a GGUF file. Raises ImportError if llama-cpp-python not installed,
        or RuntimeError if the model architecture is not supported.
        """
        from llama_cpp import Llama

        if hw is None:
            hw = {}

        n_gpu_layers = -1 if hw.get("gpu_available") else 0
        n_threads = os.cpu_count() or 4

        print(f"🍐 Loading GGUF: {os.path.basename(path)}")
        print(f"   Context: {n_ctx} tokens, Threads: {n_threads}")

        llm = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        print(f"   ✓ GGUF model ready")
        return _GGUFBackend(llm, path)

    def generate(self, prompt: str, config: FigInferenceConfig) -> str:
        output = self._llm(
            prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repetition_penalty,
            echo=False,
            stop=["<end_of_turn>", "<|im_end|>", "<|eot_id|>",
                  "\nUser:", "\nHuman:", "<|end|>"],
        )
        return output["choices"][0]["text"].strip()

    def stream(self, prompt: str, config: FigInferenceConfig) -> Iterator[str]:
        stream = self._llm(
            prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repeat_penalty=config.repetition_penalty,
            echo=False,
            stream=True,
            stop=["<end_of_turn>", "<|im_end|>", "<|eot_id|>",
                  "\nUser:", "\nHuman:", "<|end|>"],
        )
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                yield token


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
                 backend: str = "hf", gguf_backend: Optional[_GGUFBackend] = None):
        self.model = model              # HF model (or None for GGUF)
        self.tokenizer = tokenizer      # HF tokenizer (or None for GGUF)
        self.model_name = model_name
        self.config = FigInferenceConfig()
        self.backend = backend          # "hf", "fig_engine", "gguf"
        self._gguf = gguf_backend

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

        if use_int4 is None:
            ram_gb = hw.get("ram_available_gb", 16)
            large_indicators = ["4b", "7b", "8b", "13b", "70b", "gemma-3", "gemma-4"]
            name_lower = model_name_or_path.lower()
            large_model = any(x in name_lower for x in large_indicators)
            use_int4 = (not gpu) and (ram_gb < 12 or large_model)

        if use_int4:
            return FigLanguageModel._load_int4(model_name_or_path, hw)
        else:
            return FigLanguageModel._load_fp32(model_name_or_path, hw)

    @staticmethod
    def _load_gguf(path: str, hw: dict) -> "FigLanguageModel":
        """Load from GGUF file with graceful fallback."""
        filename = os.path.basename(path)

        # Try llama-cpp-python first
        try:
            gguf_backend = _GGUFBackend.load(path, hw=hw)
            return FigLanguageModel(
                model=None, tokenizer=None,
                model_name=filename, backend="gguf",
                gguf_backend=gguf_backend,
            )
        except ImportError:
            print(f"⚠ llama-cpp-python not installed.")
            print(f"   Install: pip install llama-cpp-python")
            print(f"   For Gemma 4 support, build from source:")
            print(f"     git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git")
            print(f"     cd llama-cpp-python/vendor/llama.cpp && git checkout b8828 && cd ../..")
            print(f"     pip install -e .")
            print(f"")
            print(f"   Falling back to Fig Engine INT4...")
        except Exception as e:
            error_msg = str(e).lower()
            if "unsupported model architecture" in error_msg or "failed to load" in error_msg:
                print(f"⚠ GGUF load failed: {e}")
                print(f"   This usually means your llama-cpp-python version doesn't")
                print(f"   support this model's architecture (e.g., Gemma 4).")
                print(f"   Fix: pip install llama-cpp-python --upgrade")
                print(f"   Or build from source with llama.cpp b8828+.")
                print(f"")
                print(f"   Falling back to Fig Engine INT4...")
            else:
                print(f"⚠ GGUF load error: {e}")
                print(f"   Falling back to Fig Engine INT4...")

        # Fallback: Try to infer the HF model name from the GGUF filename
        # e.g., "gemma-3-4b-it-Q4_K_M.gguf" → try "google/gemma-3-4b-it"
        hf_name = _guess_hf_name_from_gguf(filename)
        if hf_name:
            print(f"   Attempting to load from Hub: {hf_name}")
            try:
                return FigLanguageModel._load_int4(hf_name, hw)
            except Exception as e2:
                print(f"   ⚠ Hub load also failed: {e2}")

        raise RuntimeError(
            f"Could not load GGUF file: {path}\n"
            f"Install llama-cpp-python with Gemma 4 support, or download the HF version."
        )

    @staticmethod
    def _load_fp32(model_name: str, hw: dict) -> "FigLanguageModel":
        """Load model in full FP32."""
        gpu = hw.get("gpu_available", False)
        dtype = torch.float16 if gpu else torch.float32

        print(f"🍐 Loading: {model_name} (FP32)")
        print(f"   Device : {'GPU' if gpu else 'CPU'}")

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
        print(f"   ✓ {param_count:.2f}B parameters loaded")

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
        if self.backend == "gguf":
            return self._gguf.generate(prompt, self.config)

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
        if self.backend == "gguf":
            yield from self._gguf.stream(prompt, self.config)
            return

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _guess_hf_name_from_gguf(filename: str) -> Optional[str]:
    """
    Try to guess the HuggingFace model name from a GGUF filename.
    e.g., "gemma-3-4b-it-Q4_K_M.gguf" → "google/gemma-3-4b-it"
    """
    name = filename.lower().replace(".gguf", "")

    # Remove quantization suffixes
    for suffix in ["-q4_k_m", "-q4_k_s", "-q5_k_m", "-q5_k_s", "-q8_0",
                   "-q6_k", "-q3_k_m", "-q3_k_s", "-q2_k", "-iq4_xs",
                   "_q4_k_m", "_q4_k_s", "_q5_k_m", "_q8_0"]:
        if name.endswith(suffix):
            name = name[:len(name) - len(suffix)]
            break

    # Common model name → HF repo mappings
    mappings = {
        "gemma-3-4b-it": "google/gemma-3-4b-it",
        "gemma-3-12b-it": "google/gemma-3-12b-it",
        "gemma-3-1b-it": "google/gemma-3-1b-it",
        "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "phi-2": "microsoft/phi-2",
        "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "tinyllama-1.1b-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    }

    for pattern, hf_name in mappings.items():
        if pattern in name:
            return hf_name

    return None
