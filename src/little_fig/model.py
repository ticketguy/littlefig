"""
Little Fig — Model Engine
Handles both CPU and GPU. Gemma-aware (4B-IT and others).
Two backends: HuggingFace (transformers) and GGUF (llama-cpp-python).

Gemma 4B-IT on CPU:
  - HF float32: ~16GB RAM, ~30-60s per response. Feasible if you have RAM.
  - GGUF Q4_K_M: ~3GB RAM, ~5-10s per response. Recommended.

Gemma 4B-IT on GPU (when you get one):
  - HF float16: ~8GB VRAM, ~1-2s per response.
  - Auto-detected at startup via __init__.HW
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from threading import Thread
from typing import Iterator, Optional, Tuple
from dataclasses import dataclass, field


# ── Generation config ─────────────────────────────────────────────────────────

@dataclass
class FigInferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


# ── Model identity helpers ────────────────────────────────────────────────────

KNOWN_MODELS = {
    # model_id substring → display info
    "gemma": {
        "chat_format": "gemma",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "tinyllama": {
        "chat_format": "chatml",
        "target_modules": ["q_proj", "v_proj"],
    },
    "phi": {
        "chat_format": "phi",
        "target_modules": ["q_proj", "v_proj", "fc1", "fc2"],
    },
    "llama": {
        "chat_format": "llama",
        "target_modules": ["q_proj", "v_proj"],
    },
    "mistral": {
        "chat_format": "llama",
        "target_modules": ["q_proj", "v_proj"],
    },
}


def _get_model_info(model_name: str) -> dict:
    name_lower = model_name.lower()
    for key, info in KNOWN_MODELS.items():
        if key in name_lower:
            return info
    return {"chat_format": "default", "target_modules": ["q_proj", "v_proj"]}


def _get_lora_target_modules(model_name: str) -> list:
    return _get_model_info(model_name)["target_modules"]


# ── HuggingFace loader ────────────────────────────────────────────────────────

class FigLanguageModel:
    """
    HuggingFace transformers backend.
    Auto-selects dtype and device_map from hardware detection.

    For Gemma 4B on CPU: needs ~16GB RAM in float32.
    For Gemma 4B on GPU: needs ~8GB VRAM in float16.
    """

    def __init__(self, model, tokenizer, model_name: str = ""):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = FigInferenceConfig()
        self._model_info = _get_model_info(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def from_pretrained(
        model_name: str,
        hw: Optional[dict] = None,
    ) -> "FigLanguageModel":
        """
        Load a model. Pass hw=HW from __init__ for auto GPU/CPU config,
        or leave None to auto-detect.
        """
        if hw is None:
            from little_fig import HW
            hw = HW

        gpu = hw.get("gpu_available", False)
        dtype = hw.get("recommended_dtype", torch.float32)

        print(f"🍐 Loading: {model_name}")
        print(f"   Device : {'GPU (' + hw.get('gpu_name', '') + ')' if gpu else 'CPU'}")
        print(f"   dtype  : {dtype}")

        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        if gpu:
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "cpu"
            # On CPU, warn about RAM for large models
            if "gemma" in model_name.lower() and "4b" in model_name.lower():
                print("   ⚠  Gemma 4B float32 needs ~16GB RAM.")
                print("   ⚠  If you hit OOM, use GGUF instead (FigGGUFModel).")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"   ✓ {param_count:.2f}B parameters loaded")

        return FigLanguageModel(model, tokenizer, model_name)

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
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
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

    def apply_chat_template(self, message: str, history: list) -> str:
        """
        Build a proper chat prompt. Uses the model's built-in chat template
        if available (Gemma, TinyLlama, etc. all have one).
        """
        messages = []
        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        # Use the tokenizer's own chat template (most modern models have this)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass  # Fall through to manual format

        # Manual fallback by model family
        fmt = self._model_info.get("chat_format", "default")

        if fmt == "gemma":
            prompt = ""
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                prompt += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"
            return prompt

        if fmt == "chatml":
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


# ── GGUF loader ───────────────────────────────────────────────────────────────

class FigGGUFModel:
    """
    GGUF backend via llama-cpp-python.
    Recommended for Gemma 4B on CPU — 4-8x faster than HF float32.

    Install:
        pip install llama-cpp-python

    Download Gemma 4B Q4 (~2.5GB):
        huggingface-cli download bartowski/gemma-3-4b-it-GGUF \\
            gemma-3-4b-it-Q4_K_M.gguf --local-dir ./models
    """

    def __init__(self, llm, model_path: str = ""):
        self._llm = llm
        self.model_name = os.path.basename(model_path)
        self.config = FigInferenceConfig()
        self._model_info = _get_model_info(model_path)

    @staticmethod
    def from_gguf(
        path: str,
        context_length: int = 4096,
        hw: Optional[dict] = None,
    ) -> Tuple["FigGGUFModel", None]:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "\n🍐 llama-cpp-python not installed.\n"
                "   pip install llama-cpp-python\n"
            )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n🍐 GGUF file not found: {path}\n"
                "   Download from HuggingFace. See README.\n"
            )

        if hw is None:
            from little_fig import HW
            hw = HW

        n_gpu_layers = 0
        if hw.get("gpu_available"):
            # Offload all layers to GPU if available
            n_gpu_layers = -1
            print(f"🍐 GGUF: GPU offload enabled ({hw.get('gpu_name', 'GPU')})")

        print(f"🍐 Loading GGUF: {os.path.basename(path)}")
        print(f"   Context : {context_length} tokens")
        print(f"   Threads : {os.cpu_count()} CPU cores")

        llm = Llama(
            model_path=path,
            n_ctx=context_length,
            n_threads=os.cpu_count(),
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        print(f"   ✓ GGUF model ready")
        return FigGGUFModel(llm, path), None

    def generate(self, prompt: str) -> str:
        output = self._llm(
            prompt,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repetition_penalty,
            echo=False,
            stop=["<end_of_turn>", "<|im_end|>", "\nUser:", "\nHuman:"],
        )
        return output["choices"][0]["text"].strip()

    def stream(self, prompt: str) -> Iterator[str]:
        stream = self._llm(
            prompt,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repetition_penalty,
            echo=False,
            stream=True,
            stop=["<end_of_turn>", "<|im_end|>", "\nUser:", "\nHuman:"],
        )
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def apply_chat_template(self, message: str, history: list) -> str:
        """Gemma-format prompt. Works for Gemma GGUF models."""
        fmt = self._model_info.get("chat_format", "gemma")

        if fmt == "gemma":
            prompt = ""
            for user_msg, bot_msg in history:
                if user_msg:
                    prompt += f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
                if bot_msg:
                    prompt += f"<start_of_turn>model\n{bot_msg}<end_of_turn>\n"
            prompt += f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"
            return prompt

        # chatml fallback
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for user_msg, bot_msg in history:
            if user_msg:
                prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            if bot_msg:
                prompt += f"<|im_start|>assistant\n{bot_msg}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        return prompt