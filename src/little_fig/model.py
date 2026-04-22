"""
Little Fig — Model Engine (v0.4)

Unified inference backend. Loads any HuggingFace model, with two modes:
    1. Full FP32 (for small models or machines with enough RAM)
    2. Fig Engine INT4 (for large models on limited RAM)

No more GGUF dependency — Fig Engine INT4 replaces it with pure PyTorch.
Works with any model on HuggingFace Hub, any architecture.
"""

import os
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


# ── Unified Model ─────────────────────────────────────────────────────────────

class FigLanguageModel:
    """
    Unified inference model for Little Fig.

    Supports any HuggingFace causal LM. Two loading modes:
        - FP32: standard HuggingFace loading
        - INT4 (Fig Engine): 7.1× less RAM, works on 8GB machines

    Usage:
        # Auto-detect best loading strategy
        model = FigLanguageModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # Force INT4 (for low-RAM machines)
        model = FigLanguageModel.from_pretrained("google/gemma-3-4b-it", use_int4=True)

        # Chat
        response = model.generate("Hello, how are you?")

        # Streaming
        for chunk in model.stream("Tell me a story"):
            print(chunk, end="")
    """

    def __init__(self, model, tokenizer, model_name: str = "", is_fig_engine: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = FigInferenceConfig()
        self.is_fig_engine = is_fig_engine

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def from_pretrained(
        model_name: str,
        hw: Optional[dict] = None,
        use_int4: Optional[bool] = None,
    ) -> "FigLanguageModel":
        """
        Load any HuggingFace model for inference.

        Args:
            model_name: HuggingFace model ID (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            hw: Hardware config dict (auto-detected if None)
            use_int4: Force INT4 mode (None = auto-detect based on RAM)
        """
        if hw is None:
            from little_fig import HW
            hw = HW

        gpu = hw.get("gpu_available", False)

        # Auto-detect whether to use INT4
        if use_int4 is None:
            ram_gb = hw.get("ram_available_gb", 16)
            # Use INT4 if less than 8GB available, or if model name suggests large
            large_model = any(x in model_name.lower() for x in ["4b", "7b", "8b", "13b", "70b", "gemma-3"])
            use_int4 = (not gpu) and (ram_gb < 12 or large_model)

        if use_int4:
            return FigLanguageModel._load_int4(model_name, hw)
        else:
            return FigLanguageModel._load_fp32(model_name, hw)

    @staticmethod
    def _load_fp32(model_name: str, hw: dict) -> "FigLanguageModel":
        """Load model in full FP32 (standard HuggingFace)."""
        gpu = hw.get("gpu_available", False)
        dtype = torch.float16 if gpu else torch.float32

        print(f"🍐 Loading: {model_name} (FP32)")
        print(f"   Device : {'GPU' if gpu else 'CPU'}")

        load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}

        # Try with device_map first (needs accelerate), fall back without
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

        return FigLanguageModel(model, tokenizer, model_name, is_fig_engine=False)

    @staticmethod
    def _load_int4(model_name: str, hw: dict) -> "FigLanguageModel":
        """Load model with Fig Engine INT4 quantization (7.1× less RAM)."""
        from little_fig.engine.model import FigModel

        print(f"🍐 Loading: {model_name} (Fig Engine INT4)")

        fig_model = FigModel.from_pretrained(
            model_name,
            lora_r=0,       # No LoRA for inference-only
            lora_alpha=0,
        )
        fig_model.model.eval()

        return FigLanguageModel(
            fig_model.model, fig_model.tokenizer, model_name, is_fig_engine=True
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
        """Tokenize and move to model device."""
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
        Build a chat prompt from message + history.

        History format: list of {"role": ..., "content": ...} dicts
        (Gradio messages format, compatible with Gradio 4.x+)
        """
        messages = []

        # History is a list of message dicts (Gradio messages format)
        for msg in history:
            if isinstance(msg, dict):
                messages.append(msg)
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                # Legacy tuple format: (user_msg, bot_msg)
                user_msg, bot_msg = msg
                if user_msg:
                    messages.append({"role": "user", "content": str(user_msg)})
                if bot_msg:
                    messages.append({"role": "assistant", "content": str(bot_msg)})

        messages.append({"role": "user", "content": message})

        # Use the tokenizer's built-in chat template (most modern models have this)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass  # Fall through to generic fallback

        # Generic fallback — works for any model
        prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            prompt += f"{role}: {msg['content']}\n"
        prompt += "Assistant:"
        return prompt
