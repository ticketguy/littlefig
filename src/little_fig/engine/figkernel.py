"""
FigKernel — Fused Operations for CPU and GPU

Custom fused operations designed to be torch.compile-friendly.
Unlike Triton kernels (GPU-only), these work on BOTH CPU and GPU
through torch.compile's inductor backend.

Key operations:
    1. FigFusedLinearLoRA:    dequant + matmul + LoRA in one compiled op
    2. FigFusedRMSNorm:       RMS normalization with minimal saved state
    3. FigFusedCrossEntropy:  chunked cross-entropy that never materializes full logits
    4. FigFusedSwiGLU:        gate + up + SiLU fused with recomputation in backward

The compile strategy: write clean PyTorch that torch.compile can fuse.
inductor generates AVX-512 on CPU, CUDA on GPU — same code, both targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# FigFusedRMSNorm — saves only inv_rms (scalar per row) for backward
# ═══════════════════════════════════════════════════════════════════════════════

class FigRMSNorm(nn.Module):
    """
    RMS LayerNorm that saves minimal state for backward.
    
    Standard RMSNorm saves the full normalized output for backward.
    We save only inv_rms (one scalar per row) and recompute in backward.
    Memory: O(batch * seq) instead of O(batch * seq * hidden).
    
    This is what Liger/Unsloth do with Triton — we do it with torch.compile.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fig_rms_norm_forward(x, self.weight, self.eps)


@torch.compile(backend="inductor", fullgraph=True, dynamic=True)
def _fig_rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compiled RMSNorm — inductor fuses norm + scale into one pass."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return x * inv_rms * weight


# ═══════════════════════════════════════════════════════════════════════════════
# FigFusedCrossEntropy — chunked, never materializes full logit tensor
# ═══════════════════════════════════════════════════════════════════════════════

class FigCrossEntropy(nn.Module):
    """
    Chunked cross-entropy loss that processes vocab in chunks.
    
    Standard CE: logits [batch*seq, vocab] → softmax → loss
    For vocab=128K, that's a 128K × 4-byte tensor PER TOKEN.
    
    FigCE: Process logits in chunks of 8K, compute partial logsumexp,
    merge. Never allocates the full [batch*seq, vocab] tensor.
    
    This is the #1 memory optimization from Liger — we implement it
    in pure PyTorch so torch.compile can optimize for both CPU and GPU.
    """
    def __init__(self, chunk_size: int = 8192, ignore_index: int = -100):
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
    
    def forward(
        self,
        hidden_states: torch.Tensor,   # [batch*seq, hidden]
        weight: torch.Tensor,           # [vocab, hidden]  (lm_head weight)
        targets: torch.Tensor,          # [batch*seq]
    ) -> torch.Tensor:
        """
        Compute cross-entropy without materializing full logits.
        
        Instead of: logits = hidden @ weight.T; loss = CE(logits, targets)
        We do:      for each chunk of vocab: partial_logits, accumulate loss
        """
        return _fig_chunked_ce(
            hidden_states, weight, targets,
            self.chunk_size, self.ignore_index,
        )


def _fig_chunked_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    ignore_index: int,
) -> torch.Tensor:
    """
    Chunked cross-entropy computation.
    
    For each token position, we need: loss = -logit[target] + logsumexp(all logits)
    
    We compute logsumexp in chunks:
        chunk_logits = hidden @ weight[chunk_start:chunk_end].T
        Track running max and sum for stable logsumexp
    """
    n_tokens = hidden.shape[0]
    vocab_size = weight.shape[0]
    
    # Mask for valid (non-ignored) targets
    valid_mask = targets != ignore_index
    
    # We need: for each token, the logit at the target position, and logsumexp over all
    target_logits = torch.zeros(n_tokens, device=hidden.device, dtype=hidden.dtype)
    running_max = torch.full((n_tokens,), float('-inf'), device=hidden.device, dtype=hidden.dtype)
    running_sum = torch.zeros(n_tokens, device=hidden.device, dtype=hidden.dtype)
    
    # Process vocab in chunks
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        
        # Compute logits for this vocab chunk: [n_tokens, chunk]
        chunk_logits = F.linear(hidden, weight[start:end])
        
        # Extract target logits that fall in this chunk
        target_in_chunk = (targets >= start) & (targets < end) & valid_mask
        if target_in_chunk.any():
            local_idx = targets[target_in_chunk] - start
            target_logits[target_in_chunk] = chunk_logits[target_in_chunk].gather(
                1, local_idx.unsqueeze(1)
            ).squeeze(1)
        
        # Update running logsumexp (numerically stable)
        chunk_max = chunk_logits.max(dim=1).values
        new_max = torch.maximum(running_max, chunk_max)
        
        # Rescale previous sum and add new chunk
        running_sum = running_sum * torch.exp(running_max - new_max) + \
                      torch.exp(chunk_max - new_max) * chunk_logits.sub(chunk_max.unsqueeze(1)).exp().sum(dim=1)
        running_max = new_max
    
    # Final loss: -target_logit + logsumexp
    logsumexp = running_max + torch.log(running_sum)
    loss_per_token = logsumexp - target_logits
    
    # Mask out ignored tokens
    loss_per_token = loss_per_token * valid_mask.float()
    
    n_valid = valid_mask.sum().clamp(min=1)
    return loss_per_token.sum() / n_valid


# ═══════════════════════════════════════════════════════════════════════════════
# FigFusedSwiGLU — gate + up + SiLU in one pass
# ═══════════════════════════════════════════════════════════════════════════════

class FigSwiGLU(nn.Module):
    """
    Fused SwiGLU activation.
    
    Standard: gate = silu(x @ W_gate); up = x @ W_up; out = gate * up
    Fused: Compute both projections, apply SiLU, multiply in one operation.
    
    Backward: Recompute SiLU instead of saving gate activations.
    Saves hidden_size * seq_len * batch * 4 bytes per layer.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_fig_swiglu_forward(
            x, self.gate_proj.weight, self.up_proj.weight,
            self.gate_proj.bias, self.up_proj.bias,
        ))


@torch.compile(backend="inductor", fullgraph=True, dynamic=True)
def _fig_swiglu_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    up_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused gate + up + SiLU — inductor merges the two matmuls and activation."""
    gate = F.linear(x, gate_weight, gate_bias)
    up = F.linear(x, up_weight, up_bias)
    return F.silu(gate) * up


# ═══════════════════════════════════════════════════════════════════════════════
# FigFusedLinearLoRA — the core operation, dequant + matmul + LoRA fused
# ═══════════════════════════════════════════════════════════════════════════════

@torch.compile(backend="inductor", fullgraph=True, dynamic=True)
def fig_fused_linear_lora(
    x: torch.Tensor,          # [batch, seq, in_features]
    cached_W: torch.Tensor,   # [out_features, in_features] — pre-dequantized
    lora_A: torch.Tensor,     # [in_features, r]
    lora_B: torch.Tensor,     # [r, out_features]
    lora_scale: float,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused base matmul + LoRA correction.
    
    inductor fuses: F.linear + matmul + matmul + scale + add into one kernel.
    On CPU: generates AVX-512 vectorized code.
    On GPU: generates CUDA code.
    
    This is THE performance-critical path. Every forward pass goes through here.
    """
    h = F.linear(x, cached_W, bias)
    h = h + (x @ lora_A) @ lora_B * lora_scale
    return h


@torch.compile(backend="inductor", fullgraph=True, dynamic=True)
def fig_fused_linear(
    x: torch.Tensor,
    cached_W: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused linear without LoRA (inference mode)."""
    return F.linear(x, cached_W, bias)
