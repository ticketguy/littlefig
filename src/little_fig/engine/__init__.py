"""
Fig Engine — CPU-Native LLM Training System (v0.6)

A novel architecture for training large language models on CPU with minimal RAM.
Powered by FigQuant adaptive codebook quantization, FigKernel fused ops,
FigPipeline async GPU-CPU training, and 4 automatic training tiers.

Stack:
  FigQuant   → Adaptive codebook INT4 (9.7% less MSE than standard INT4)
  FigKernel  → torch.compile fused ops (RMSNorm, SwiGLU, CrossEntropy, LinearLoRA)
  FigPipeline → Async GPU-CPU training with CPU-resident optimizer states
  4 Tiers    → LoRA, LISA, MeZO, LOMO (auto-selected by available RAM)
  Ember      → Cognitive memory token injection + training data generation

References:
    - LISA: arxiv 2403.17919 (Layerwise Importance Sampled AdamW)
    - LOMO: arxiv 2306.09782 (Low-Memory Optimization)
    - MeZO: arxiv 2305.17333 (Memory-efficient Zeroth-Order)
    - GaLore: arxiv 2403.03507 (Gradient Low-Rank Projection)
    - BitNet: arxiv 2402.17764 (1.58-bit LLMs)
"""

__version__ = "0.6.0"

# Core — FigQuant-powered
from .linear import FigLinear
from .model import FigModel
from .trainer import FigTrainer, FigTrainingConfig
from .tier import TrainingTier, select_tier

# FigQuant — Adaptive codebook quantization (primary quantization engine)
from .figquant import FigQuantTensor, figquant_quantize, figquant_dequantize, measure_quality

# FigKernel — Fused operations via torch.compile
from .figkernel import (
    FigRMSNorm, FigCrossEntropy, FigSwiGLU,
    fig_fused_linear_lora, fig_fused_linear,
    fig_chunked_cross_entropy,
)

# FigPipeline — Async GPU-CPU training
from .figpipeline import FigPipeline, PipelineConfig

# Ember — Cognitive memory integration
from .ember_integration import (
    MEMORY_TOKENS,
    EmberTrainingDataGenerator,
    EmberChatManager,
)

__all__ = [
    # Core
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
    # Ember
    "MEMORY_TOKENS", "EmberTrainingDataGenerator", "EmberChatManager",
]
