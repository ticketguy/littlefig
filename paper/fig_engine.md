# Fig Engine: CPU-Native LLM Training via Adaptive Codebook Quantization and Cognitive Memory Embedding

**Authors:** Sammie (@ticketguy)  
**Repository:** https://github.com/ticketguy/littlefig  
**Version:** 0.6

---

## Abstract

We present **Fig Engine**, a system for fine-tuning large language models entirely on CPU with minimal RAM. Fig Engine combines five components: (1) **FigQuant**, an adaptive codebook INT4 quantization that refines NF4 quantiles via k-means — measured at 5.3% lower MSE than fixed NF4 and 57.0% lower MSE than uniform absmax INT4 across all 50 weight matrices in GPT-2, winning every layer; (2) **FigCache**, a three-tier caching strategy that trades between memory and speed by caching unpacked codebook indices instead of full FP32 weights — 75% less memory at 1.3× the speed of no-cache; (3) **FigKernel**, torch.compile fused operations for RMSNorm (2.95× speedup), SwiGLU, cross-entropy, and linear+LoRA; (4) **FigSweep**, a rolling layer-window strategy that dequantizes only a subset of layers at a time during sequential forward passes; and (5) **Ember integration**, which trains cognitive memory operations directly into model weights via special tokens.

Fig Engine fine-tunes GPT-2 (124M) using 45.8 MB for base weights and projects TinyLlama (1.1B) at ~400 MB — an order of magnitude below the 26.6 GB required by standard FP32+AdamW.

---

## 1. Introduction

Current LLM fine-tuning tools (Unsloth, LLaMA-Factory, TRL) assume GPU with high-bandwidth memory. This excludes consumer hardware, CPU cloud instances, and edge devices. While these tools technically support CPU via PyTorch's device abstraction, they do not optimize for the CPU memory hierarchy.

The fundamental bottleneck on CPU is **memory bandwidth** (5-8 GB/s), not compute. A 1.1B model in FP32 requires 4.4 GB for weights alone, plus 4.4 GB for gradients and 8.8 GB for AdamW states — 26.6 GB total, exceeding most consumer machines.

### Contributions

1. **FigQuant**: Adaptive codebook INT4 quantization with k-means refinement and double quantization. 0.9948 cosine similarity on GPT-2 real weights (50 layers), 5.3% less MSE than NF4 on every layer, 7.4× compression.

2. **FigCache**: A three-mode caching strategy (fast/figcache/lowram) where the middle mode caches unpacked uint8 codebook indices — 75% less memory than full FP32 cache, 1.3× faster than full dequant. This exploits FigQuant's codebook structure: bit-unpacking is 60% of dequant cost and can be amortized.

3. **FigSweep**: Rolling layer-window dequantization. Since transformer layers execute sequentially, only a window of W layers needs to be in fast mode at any time. For GPT-2 (48 layers), window=4 uses 25 MB vs 302 MB for full cache.

4. **FigKernel**: torch.compile fused operations that generate AVX-512 on CPU and CUDA on GPU from the same source. FigRMSNorm (2.95×), chunked cross-entropy (8× less peak memory), fused linear+LoRA.

5. **Four Training Tiers**: Automatic selection of LoRA, LISA, MeZO, or LOMO based on available RAM. Each tier uses FigQuant weights and benefits from FigKernel acceleration.

6. **Ember Memory Embedding**: 9 special tokens (`<|mem_store|>`, `<|mem_recall|>`, etc.) injected into the model vocabulary, with synthetic training data generation for cognitive memory operations.

---

## 2. System Architecture

### 2.1 FigQuant

Adaptive codebook INT4 quantization with three techniques:

1. **Adaptive codebook**: Initialize from NF4 quantiles (16 values from N(0,1)), then refine via k-means on the actual weight distribution. This captures heavy tails and layer-specific distributions that a fixed codebook misses.

2. **Double quantization**: Per-group scale factors quantized to FP8, saving ~0.37 bits/param.

Note: sensitivity weighting (upweighting high-magnitude values during k-means) was tested and found to hurt quality — it pulls the codebook toward outliers at the expense of the dense center. Uniform weighting during k-means produces consistently better results on real model weights.

**Dequantization**: `codebook[indices] × per_group_scale`. The codebook lookup uses `torch.gather` for vectorized operation, compatible with `torch.compile`.

Measured on all 50 weight matrices in GPT-2 (124M), group_size=128:

| Method | Cosine Sim | MSE | SNR (dB) |
|--------|-----------|-----|----------|
| **FigQuant** | **0.9948** | **1.768e-4** | **19.8** |
| NF4 (fixed codebook) | 0.9946 | 1.866e-4 | 19.6 |
| Absmax INT4 (uniform) | 0.9883 | 4.114e-4 | 17.1 |

FigQuant vs NF4: **5.3% lower MSE**, +0.2 dB SNR — FigQuant wins on **50/50 layers**. FigQuant vs Absmax INT4: **57.0% lower MSE**, +2.7 dB SNR. The adaptive codebook captures layer-specific weight distributions (skew, heavy tails, outliers) that a fixed NF4 codebook misses.

### 2.2 FigCache

FigQuant's dequantization has three stages: (1) bit-unpacking packed indices, (2) codebook lookup, (3) scale multiplication. Profiling reveals bit-unpacking is 60% of total dequant time. FigCache exploits this by offering three modes:

| Mode | What's cached | Memory (768→2048) | Speed vs fast |
|------|--------------|-------------------|---------------|
| **fast** | Full FP32 weight | 6144 KB (100%) | 1.0× |
| **figcache** | Unpacked uint8 indices | 1536 KB (25%) | 2.2× |
| **lowram** | Nothing (packed INT4) | 828 KB (13%) | 2.9× |

FigCache mode is specific to FigQuant's architecture — it only works because the codebook is small (16 values, 64 bytes) and shared globally, so the per-layer cache is just the pre-unpacked index array.

For GPT-2 (48 quantized layers):
- fast: 302 MB cached
- figcache: 75 MB cached (75% savings)
- lowram: 41 MB (INT4 only)

### 2.3 FigSweep

Transformer layers execute sequentially during forward and backward passes. FigSweep maintains a sliding window of W layers in fast mode, switching layers to lowram as the window moves past them.

For GPT-2 with window=4: 25 MB total cache instead of 302 MB. Layers inside the window run at fast-mode speed; layers outside are lowram but are also not actively computing.

### 2.4 FigKernel

Fused operations via `torch.compile(backend="inductor")`, generating AVX-512 on CPU and CUDA on GPU:

- **FigRMSNorm**: Fuses variance, rsqrt, and scale. Saves only inv_rms (scalar per row) for backward. 2.95× speedup. Auto-swapped into models at load time.
- **FigCrossEntropy**: Processes vocabulary in 8K chunks with numerically stable running logsumexp. Never materializes full [seq_len, vocab] tensor. ~8× less peak memory.
- **FigSwiGLU**: Fuses gate + up + SiLU into one compiled pass.
- **fig_fused_linear_lora**: `F.linear(x, W) + (x @ A) @ B * scale` compiled into one kernel.

### 2.5 Training Tiers

Auto-selected by available RAM (70% budget, 30% OS headroom):

| Tier | Method | Memory (1.1B) | Quality |
|------|--------|---------------|---------|
| 1 | Streaming LoRA | ~400 MB | Good |
| 2 | LISA | ~900 MB | Better |
| 3 | MeZO | ~600 MB | Acceptable |
| 4 | LOMO | ~800 MB | Best |

### 2.6 Ember Memory Integration

9 special tokens injected into the model vocabulary via `ember_mode=True`:

```
<|mem_store|>  <|mem_recall|>  <|mem_consolidate|>
<|mem_forget|>  <|mem_conflict|>  <|mem_episode|>
<|mem_reflect|>  <|memory_start|>  <|memory_end|>
```

The training data generator produces synthetic examples across 7 memory operation types (store, recall, consolidate, forget, conflict detection, episode segmentation, reflection). The trained model learns to emit memory operations as part of its text generation, enabling it to operate an Ember's Diaries instance.

---

## 3. Experimental Results

### 3.1 FigQuant vs Baselines (GPT-2 Real Weights)

All three methods measured on every 2D weight matrix in GPT-2 (50 layers), group_size=128. Real NF4 uses the same fixed codebook that FigQuant initializes from but with zero refinement. Absmax INT4 uses 16 uniformly-spaced levels.

| Layer type | FigQuant MSE | NF4 MSE | FQ wins |
|-----------|-------------|---------|---------|
| Embeddings (wte, wpe) | 1.57e-4 | 1.75e-4 | 2/2 |
| Attention (c_attn, c_proj) | 1.83e-4 | 1.94e-4 | 24/24 |
| MLP (c_fc, c_proj) | 1.68e-4 | 1.74e-4 | 24/24 |
| **All layers** | **1.768e-4** | **1.866e-4** | **50/50** |

FigQuant vs NF4: **5.3% less MSE**, +0.2 dB SNR, higher cosine on every layer.
FigQuant vs Absmax INT4: **57.0% less MSE**, +2.7 dB SNR.

### 3.2 FigCache Benchmark (768→2048, seq=128)

| Mode | Forward (ms) | Cache memory | vs fast |
|------|-------------|-------------|---------|
| nn.Linear | 1.70 | 6144 KB (FP32) | baseline |
| fast | 2.18 | 6144 KB | 1.0× |
| figcache | 4.86 | 1536 KB | 2.2× |
| lowram | 6.39 | 828 KB | 2.9× |

FigCache produces **zero numerical error** vs fast mode — the output is bit-identical.

### 3.3 GPT-2 End-to-End

- 48 linear layers quantized with FigQuant
- Base weights: 339.7 MB → 45.8 MB (7.4× compression)
- 1,179,648 trainable LoRA parameters (2.9% of total)
- Forward + backward + adapter save verified

### 3.4 Memory Projections

| Model | Standard | Fig Tier 1 (LoRA) | Fits 8GB? |
|-------|---------|-------------------|-----------|
| GPT-2 (124M) | 3.48 GB | ~350 MB | ✓ |
| TinyLlama (1.1B) | 26.6 GB | ~400 MB | ✓ |
| Gemma 4B | 96.9 GB | ~1.5 GB | ✓ |
| Llama 3.1 8B | 193.7 GB | ~3 GB | ✓ |

### 3.5 FigKernel Benchmarks (CPU, 2048 hidden, seq=256)

| Operation | Standard | FigKernel | Speedup |
|-----------|---------|-----------|---------|
| RMSNorm | 4.72 ms | 1.60 ms | 2.95× |
| Cross-entropy (32K vocab) | Full alloc | 8K chunks | ~8× less memory |

---

## 4. Conclusion

Fig Engine demonstrates that CPU-native LLM fine-tuning with 8GB RAM is practical. The key architectural decisions are: (1) FigQuant's adaptive codebook reduces quantization error by 5.3% vs NF4 on real model weights (50/50 layers), by fitting the codebook to each layer's actual distribution via k-means; (2) FigCache exploits the codebook structure to cache at an intermediate level — cheaper than FP32 weights, faster than full dequant; (3) FigSweep reduces whole-model cache to a layer window by exploiting sequential execution; and (4) Ember integration embeds cognitive memory directly into model weights rather than bolting it on externally.

---

## References

1. Hu, E., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. Pan, T., et al. (2024). "LISA: Layerwise Importance Sampling for Memory-Efficient LLM Fine-Tuning." arxiv 2403.17919.
3. Malladi, S., et al. (2023). "Fine-Tuning Language Models with Just Forward Passes." NeurIPS 2023. arxiv 2305.17333.
4. Lv, K., et al. (2023). "Full Parameter Fine-tuning for Large Language Models with Limited Resources." arxiv 2306.09782.
5. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023.
6. Lin, J., et al. (2024). "AWQ: Activation-aware Weight Quantization." MLSys 2024.
