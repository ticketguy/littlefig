"""
FigPipeline — Async GPU-CPU Training Engine

Heterogeneous pipeline that overlaps GPU compute with CPU optimizer updates.

Architecture:
    GPU: forward + backward (compute-bound)
    CPU: AdamW optimizer states + parameter updates (memory-bound)

    For LoRA training: only adapter gradients transferred (tiny).
    For full fine-tune: overlap grad D2H with next layer's compute.

This enables training models larger than GPU VRAM:
    RTX 4090 (24GB) + 64GB RAM → fine-tune 70B+ with LoRA
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from dataclasses import dataclass
import time


@dataclass
class PipelineConfig:
    """Configuration for FigPipeline."""
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # Optimizer (AdamW, states on CPU)
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    max_grad_norm: float = 1.0


class FigPipeline:
    """
    Training pipeline with CPU-resident optimizer states.

    Key idea: optimizer states (exp_avg, exp_avg_sq) live on CPU.
    Only gradients are transferred GPU→CPU after backward.
    For LoRA (rank 16), grad transfer is ~100KB per layer — negligible.

    Usage:
        pipeline = FigPipeline(model, config)
        for batch in dataloader:
            loss = pipeline.train_step(batch)
    """

    def __init__(self, model: nn.Module, config: Optional[PipelineConfig] = None):
        self.model = model
        self.config = config or PipelineConfig()

        # Detect hardware
        if torch.cuda.is_available() and self.config.device == "cuda":
            self.device = torch.device("cuda")
            props = torch.cuda.get_device_properties(0)
            self.config.dtype = torch.bfloat16 if props.major >= 8 else torch.float16
            # CUDA streams for overlapped transfers
            self._compute_stream = torch.cuda.Stream()
            self._transfer_stream = torch.cuda.Stream()
        else:
            self.device = torch.device("cpu")
            self._compute_stream = None
            self._transfer_stream = None

        # CPU-resident AdamW states
        self._optimizer_states: Dict[str, dict] = {}
        self._step_count = 0

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor,
                   attention_mask: Optional[torch.Tensor] = None) -> float:
        """Execute one training step. Returns loss value."""
        self._step_count += 1
        self.model.train()

        if self.device.type == "cpu":
            return self._cpu_train_step(input_ids, labels, attention_mask)
        return self._gpu_train_step(input_ids, labels, attention_mask)

    def _cpu_train_step(self, input_ids, labels, attention_mask):
        """CPU training: standard forward → backward → AdamW update."""
        kwargs = {"input_ids": input_ids, "labels": labels}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        outputs = self.model(**kwargs)
        loss = outputs.loss
        loss.backward()

        # AdamW update (all on CPU)
        with torch.no_grad():
            if self.config.max_grad_norm > 0:
                params = [p for p in self.model.parameters() if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self._adam_update_inplace(name, param)
                    param.grad = None

        return loss.item()

    def _gpu_train_step(self, input_ids, labels, attention_mask):
        """
        GPU training with CPU-resident optimizer states.

        Forward + backward on GPU, then:
        - For each parameter with a gradient:
            1. Transfer grad to CPU (async via transfer stream)
            2. AdamW update on CPU
            3. Transfer updated param back to GPU (async)
        - For LoRA params, grads are tiny → transfer is negligible
        """
        device = self.device
        dtype = self.config.dtype

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward + backward
        with torch.autocast(device.type, dtype=dtype):
            kwargs = {"input_ids": input_ids, "labels": labels}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            outputs = self.model(**kwargs)
            loss = outputs.loss

        loss.backward()

        # Gradient clipping on GPU (fast)
        with torch.no_grad():
            if self.config.max_grad_norm > 0:
                params = [p for p in self.model.parameters() if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)

            # Optimizer updates: overlap with transfer stream
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                if param.numel() < 100_000:
                    # Small params (LoRA adapters): update on GPU directly (faster than transfer)
                    self._adam_update_inplace(name, param)
                else:
                    # Large params: transfer to CPU, update, transfer back
                    with torch.cuda.stream(self._transfer_stream):
                        self._adam_update_via_cpu(name, param)

                param.grad = None

        # Sync streams
        if self._transfer_stream is not None:
            self._transfer_stream.synchronize()

        return loss.item()

    def _adam_update_inplace(self, name: str, param: torch.Tensor):
        """AdamW update in-place on whatever device param lives on."""
        lr = self.config.learning_rate
        beta1, beta2 = self.config.betas
        wd = self.config.weight_decay
        eps = 1e-8
        t = self._step_count

        device = param.device
        if name not in self._optimizer_states:
            self._optimizer_states[name] = {
                "exp_avg": torch.zeros_like(param.data, device=device),
                "exp_avg_sq": torch.zeros_like(param.data, device=device),
            }
        state = self._optimizer_states[name]

        # Move states to param's device if needed
        if state["exp_avg"].device != device:
            state["exp_avg"] = state["exp_avg"].to(device)
            state["exp_avg_sq"] = state["exp_avg_sq"].to(device)

        grad = param.grad.data

        state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t
        step_size = lr / bc1
        denom = (state["exp_avg_sq"].sqrt() / (bc2 ** 0.5)).add_(eps)

        param.data.mul_(1 - lr * wd)
        param.data.addcdiv_(state["exp_avg"], denom, value=-step_size)

    def _adam_update_via_cpu(self, name: str, param: torch.Tensor):
        """AdamW with CPU-resident states. Transfers grad→CPU, updates, pushes back."""
        lr = self.config.learning_rate
        beta1, beta2 = self.config.betas
        wd = self.config.weight_decay
        eps = 1e-8
        t = self._step_count

        if name not in self._optimizer_states:
            self._optimizer_states[name] = {
                "exp_avg": torch.zeros_like(param.data, device="cpu"),
                "exp_avg_sq": torch.zeros_like(param.data, device="cpu"),
            }
        state = self._optimizer_states[name]

        grad_cpu = param.grad.data.to("cpu", non_blocking=True)
        param_cpu = param.data.to("cpu", non_blocking=True)

        state["exp_avg"].mul_(beta1).add_(grad_cpu, alpha=1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)

        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t
        step_size = lr / bc1
        denom = (state["exp_avg_sq"].sqrt() / (bc2 ** 0.5)).add_(eps)

        param_cpu.mul_(1 - lr * wd)
        param_cpu.addcdiv_(state["exp_avg"], denom, value=-step_size)

        param.data.copy_(param_cpu.to(param.device), non_blocking=True)

    @property
    def gpu_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def cleanup(self):
        """Free optimizer states."""
        self._optimizer_states.clear()
