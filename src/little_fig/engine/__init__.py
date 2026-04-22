"""
Fig Engine — CPU-Native LLM Training System

A novel architecture for training large language models on CPU with minimal RAM.
Combines: INT4 mmap-streamed weights, fused dequant-matmul, LoRA/LISA/MeZO/LOMO
training tiers, torch.compile acceleration, and automatic tier selection.

Key innovation: Layer-streaming with on-the-fly INT4 dequantization eliminates
the need to hold the full model in RAM. A 4B parameter model can be fine-tuned
in under 3GB of RAM.

New in v0.5:
  - FigQuant: Adaptive codebook INT4 (NF4-refined + sensitivity weighting)
  - FigKernel: torch.compile fused ops (RMSNorm, SwiGLU, LinearLoRA, CrossEntropy)
  - FigPipeline: Async GPU-CPU training with CPU-resident optimizer states

References:
    - LISA: arxiv 2403.17919 (Layerwise Importance Sampled AdamW)
    - LOMO: arxiv 2306.09782 (Low-Memory Optimization)
    - MeZO: arxiv 2305.17333 (Memory-efficient Zeroth-Order)
    - GaLore: arxiv 2403.03507 (Gradient Low-Rank Projection)
    - BitNet: arxiv 2402.17764 (1.58-bit LLMs)
"""

__version__ = "0.5.0"

# Core (v0.4)
from .quantize import FigQuantizer, FIG4Tensor
from .linear import FigLinear
from .model import FigModel
from .trainer import FigTrainer, FigTrainingConfig
from .tier import TrainingTier, select_tier

# New (v0.5)
from .figquant import FigQuantTensor, figquant_quantize, figquant_dequantize, measure_quality
from .figkernel import (
    FigRMSNorm, FigCrossEntropy, FigSwiGLU,
    fig_fused_linear_lora, fig_fused_linear,
    fig_chunked_cross_entropy,
)
from .figpipeline import FigPipeline, PipelineConfig

__all__ = [
    # Core
    "FigQuantizer", "FIG4Tensor",
    "FigLinear",
    "FigModel",
    "FigTrainer", "FigTrainingConfig",
    "TrainingTier", "select_tier",
    # FigQuant
    "FigQuantTensor", "figquant_quantize", "figquant_dequantize", "measure_quality",
    # FigKernel
    "FigRMSNorm", "FigCrossEntropy", "FigSwiGLU",
    "fig_fused_linear_lora", "fig_fused_linear",
    "fig_chunked_cross_entropy",
    # FigPipeline
    "FigPipeline", "PipelineConfig",
]
