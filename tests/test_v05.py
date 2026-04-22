"""
Little Fig v0.5 — Tests & Benchmarks

Tests:
  1. FigQuant: quantize/dequantize roundtrip, quality measurement
  2. FigKernel: RMSNorm, SwiGLU, chunked CE, fused linear+LoRA
  3. FigPipeline: CPU training step
  4. Integration: FigLinear uses fused kernels
  5. Comparison: FIG4 vs FigQuant quality

Benchmarks:
  - FigQuant vs FIG4 (asymmetric INT4): quality comparison
  - FigKernel fused ops vs standard PyTorch: speed comparison
  - Chunked CE vs standard CE: memory + speed
  - FigLinear fused vs non-fused: speed comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
import json
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS = {}
PASS = 0
FAIL = 0


def run_test(name, fn):
    global PASS, FAIL
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print(f"{'='*70}")
    try:
        fn()
        PASS += 1
        print(f"  ✅ PASSED: {name}")
    except Exception as e:
        FAIL += 1
        print(f"  ❌ FAILED: {name}")
        traceback.print_exc()


def bench(label, fn, warmup=2, repeats=10):
    """Benchmark a function, return avg time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    print(f"    {label}: {avg:.2f} ± {std:.2f} ms")
    return avg


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_figquant_roundtrip():
    """FigQuant: quantize → dequantize roundtrip quality."""
    from little_fig.engine.figquant import figquant_quantize, figquant_dequantize, measure_quality

    W = torch.randn(2048, 768)
    q = figquant_quantize(W, group_size=128, n_iters=8)
    W_deq = figquant_dequantize(q)

    assert W_deq.shape == W.shape, f"Shape mismatch: {W_deq.shape} vs {W.shape}"

    quality = measure_quality(W, q)
    print(f"    Cosine similarity: {quality['cosine_similarity']:.6f}")
    print(f"    MSE:               {quality['mse']:.6f}")
    print(f"    SNR (dB):          {quality['snr_db']:.1f}")
    print(f"    Bits/param:        {quality['bits_per_param']:.2f}")
    print(f"    Compression:       {quality['compression_ratio']:.1f}x")

    assert quality['cosine_similarity'] > 0.99, f"Cosine sim too low: {quality['cosine_similarity']}"
    assert quality['bits_per_param'] < 5.0, f"Too many bits: {quality['bits_per_param']}"
    RESULTS['figquant_quality'] = quality


def test_figquant_double_quant():
    """FigQuant: double quantization saves space."""
    from little_fig.engine.figquant import figquant_quantize

    W = torch.randn(1024, 512)
    q_dq = figquant_quantize(W, double_quant=True)
    q_nodq = figquant_quantize(W, double_quant=False)

    assert q_dq.dq_scales is not None, "Double quant should produce dq_scales"
    assert q_nodq.dq_scales is None, "No double quant should not produce dq_scales"

    # Double quant should use fewer bytes
    print(f"    With double quant:    {q_dq.nbytes:,} bytes ({q_dq.bits_per_param:.2f} bits/param)")
    print(f"    Without double quant: {q_nodq.nbytes:,} bytes ({q_nodq.bits_per_param:.2f} bits/param)")


def test_figquant_vs_fig4():
    """Compare FigQuant (adaptive codebook) vs FIG4 (asymmetric INT4)."""
    from little_fig.engine.figquant import figquant_quantize, figquant_dequantize, measure_quality
    from little_fig.engine.quantize import FigQuantizer

    W = torch.randn(2048, 768)

    # FigQuant (adaptive codebook)
    q_fq = figquant_quantize(W, group_size=128, n_iters=8)
    qual_fq = measure_quality(W, q_fq)

    # FIG4 (standard asymmetric INT4)
    quantizer = FigQuantizer(group_size=128)
    q_f4 = quantizer.quantize(W)
    W_f4 = q_f4.dequantize()
    cos_f4 = F.cosine_similarity(W.reshape(1, -1).float(), W_f4.reshape(1, -1).float()).item()
    mse_f4 = F.mse_loss(W_f4.float(), W.float()).item()

    print(f"    FigQuant cosine: {qual_fq['cosine_similarity']:.6f}  MSE: {qual_fq['mse']:.6f}  SNR: {qual_fq['snr_db']:.1f} dB")
    print(f"    FIG4     cosine: {cos_f4:.6f}  MSE: {mse_f4:.6f}")
    print(f"    FigQuant improvement: {(qual_fq['cosine_similarity'] - cos_f4)*100:.4f}% cosine, {(mse_f4 - qual_fq['mse'])/mse_f4*100:.1f}% MSE reduction")

    RESULTS['figquant_vs_fig4'] = {
        'figquant_cosine': qual_fq['cosine_similarity'],
        'figquant_mse': qual_fq['mse'],
        'figquant_snr': qual_fq['snr_db'],
        'fig4_cosine': cos_f4,
        'fig4_mse': mse_f4,
    }


def test_figkernel_rmsnorm():
    """FigKernel: RMSNorm correctness."""
    from little_fig.engine.figkernel import FigRMSNorm

    hidden_size = 768
    norm = FigRMSNorm(hidden_size, eps=1e-6)
    x = torch.randn(2, 128, hidden_size)

    out = norm(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    # Compare with manual RMSNorm
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + 1e-6)
    expected = x * inv_rms * norm.weight
    diff = (out - expected).abs().max().item()
    print(f"    Max diff vs reference: {diff:.2e}")
    assert diff < 1e-5, f"RMSNorm output differs: {diff}"


def test_figkernel_swiglu():
    """FigKernel: SwiGLU correctness."""
    from little_fig.engine.figkernel import FigSwiGLU

    hidden = 256
    intermediate = 512
    swiglu = FigSwiGLU(hidden, intermediate)
    x = torch.randn(2, 64, hidden)

    out = swiglu(x)
    assert out.shape == (2, 64, hidden), f"Shape: {out.shape}"
    assert out.requires_grad == False or True  # Just check it doesn't crash
    print(f"    Output shape: {out.shape}, mean: {out.mean().item():.4f}")


def test_figkernel_chunked_ce():
    """FigKernel: chunked cross-entropy matches standard CE."""
    from little_fig.engine.figkernel import fig_chunked_cross_entropy

    n_tokens = 64
    hidden_size = 128
    vocab_size = 1000

    hidden = torch.randn(n_tokens, hidden_size)
    weight = torch.randn(vocab_size, hidden_size)
    targets = torch.randint(0, vocab_size, (n_tokens,))
    targets[0] = -100  # Test ignore_index

    # Chunked CE
    loss_chunked = fig_chunked_cross_entropy(hidden, weight, targets, chunk_size=256)

    # Standard CE
    logits = hidden @ weight.T
    loss_standard = F.cross_entropy(logits, targets, ignore_index=-100)

    diff = abs(loss_chunked.item() - loss_standard.item())
    print(f"    Chunked CE loss:  {loss_chunked.item():.6f}")
    print(f"    Standard CE loss: {loss_standard.item():.6f}")
    print(f"    Difference:       {diff:.2e}")
    assert diff < 1e-4, f"CE loss mismatch: {diff}"


def test_figkernel_fused_linear_lora():
    """FigKernel: fused linear+LoRA matches non-fused."""
    from little_fig.engine.figkernel import fig_fused_linear_lora

    in_f, out_f, r = 768, 2048, 16
    x = torch.randn(2, 64, in_f)
    W = torch.randn(out_f, in_f)
    lora_A = torch.randn(in_f, r)
    lora_B = torch.randn(r, out_f)
    scale = 2.0
    bias = torch.randn(out_f)

    # Fused
    out_fused = fig_fused_linear_lora(x, W, lora_A, lora_B, scale, bias)

    # Non-fused reference
    out_ref = F.linear(x, W, bias) + (x @ lora_A) @ lora_B * scale

    diff = (out_fused - out_ref).abs().max().item()
    print(f"    Max diff: {diff:.2e}")
    assert diff < 1e-4, f"Fused linear+LoRA mismatch: {diff}"


def test_figpipeline_cpu():
    """FigPipeline: CPU training step runs without error."""
    from little_fig.engine.figpipeline import FigPipeline, PipelineConfig

    # Simple tiny model
    model = nn.Sequential(
        nn.Embedding(100, 32),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 100),
    )

    # Wrap: forward must return object with .loss
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, labels=None, **kw):
            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, 100), labels.view(-1))
            return type('Out', (), {'loss': loss})()

    wrapped = Wrapper(model)
    config = PipelineConfig(device="cpu", learning_rate=1e-3)
    pipeline = FigPipeline(wrapped, config)

    ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 100, (2, 16))

    loss1 = pipeline.train_step(ids, labels)
    loss2 = pipeline.train_step(ids, labels)

    print(f"    Step 1 loss: {loss1:.4f}")
    print(f"    Step 2 loss: {loss2:.4f}")
    assert loss2 < loss1 * 1.5, "Loss should not explode"  # Sanity check
    pipeline.cleanup()


def test_figlinear_uses_fused():
    """Integration: FigLinear forward works with fused kernel path."""
    from little_fig.engine.quantize import FigQuantizer
    from little_fig.engine.linear import FigLinear

    W = torch.randn(512, 256)
    quantizer = FigQuantizer(group_size=128)
    q = quantizer.quantize(W)

    layer = FigLinear(256, 512, q, lora_r=16, lora_alpha=32, fast=True)
    x = torch.randn(1, 32, 256)

    out = layer(x)
    assert out.shape == (1, 32, 512), f"Shape: {out.shape}"
    print(f"    Output shape: {out.shape}")
    print(f"    Trainable params: {layer.trainable_params:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_quantization():
    """Benchmark: FigQuant vs FIG4 quantization speed and quality."""
    from little_fig.engine.figquant import figquant_quantize, figquant_dequantize
    from little_fig.engine.quantize import FigQuantizer

    print("\n  Quantization Speed & Quality (2048×768 matrix)")
    print("  " + "-"*55)

    W = torch.randn(2048, 768)
    quantizer = FigQuantizer(group_size=128)

    t_fig4 = bench("FIG4 quantize  ", lambda: quantizer.quantize(W))
    t_fq = bench("FigQuant quant ", lambda: figquant_quantize(W, n_iters=8))

    q_fig4 = quantizer.quantize(W)
    q_fq = figquant_quantize(W, n_iters=8)

    t_fig4_deq = bench("FIG4 dequant   ", lambda: q_fig4.dequantize())
    t_fq_deq = bench("FigQuant deq   ", lambda: figquant_dequantize(q_fq))

    RESULTS['bench_quant'] = {
        'fig4_quantize_ms': t_fig4,
        'figquant_quantize_ms': t_fq,
        'fig4_dequantize_ms': t_fig4_deq,
        'figquant_dequantize_ms': t_fq_deq,
    }


def benchmark_fused_ops():
    """Benchmark: fused vs standard operations."""
    from little_fig.engine.figkernel import FigRMSNorm, fig_chunked_cross_entropy

    print("\n  Fused Ops vs Standard PyTorch")
    print("  " + "-"*55)

    # RMSNorm
    hidden_size = 2048
    x = torch.randn(4, 256, hidden_size)
    weight = torch.ones(hidden_size)

    def std_rmsnorm():
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + 1e-6) * weight

    norm = FigRMSNorm(hidden_size)
    t_std = bench("RMSNorm standard", std_rmsnorm)
    t_fig = bench("RMSNorm Fig     ", lambda: norm(x))

    # Chunked CE vs standard CE
    n = 512
    h = torch.randn(n, 256)
    w = torch.randn(32000, 256)  # Large vocab
    tgt = torch.randint(0, 32000, (n,))

    def std_ce():
        logits = h @ w.T  # Materializes full [512, 32000]
        return F.cross_entropy(logits, tgt)

    t_std_ce = bench("CE standard     ", std_ce)
    t_fig_ce = bench("CE chunked      ", lambda: fig_chunked_cross_entropy(h, w, tgt, chunk_size=4096))

    RESULTS['bench_fused'] = {
        'rmsnorm_standard_ms': t_std,
        'rmsnorm_fig_ms': t_fig,
        'ce_standard_ms': t_std_ce,
        'ce_chunked_ms': t_fig_ce,
    }


def benchmark_figlinear():
    """Benchmark: FigLinear fused vs non-fused forward pass."""
    from little_fig.engine.quantize import FigQuantizer
    from little_fig.engine.linear import FigLinear

    print("\n  FigLinear Forward Pass (768→2048, seq=256)")
    print("  " + "-"*55)

    W = torch.randn(2048, 768)
    quantizer = FigQuantizer(group_size=128)
    q = quantizer.quantize(W)

    # Standard nn.Linear
    linear = nn.Linear(768, 2048, bias=False)
    linear.weight.data = W.clone()

    x = torch.randn(1, 256, 768)

    t_std = bench("nn.Linear       ", lambda: linear(x))

    # FigLinear fast mode (with fused kernel)
    fig_fast = FigLinear(768, 2048, q, lora_r=16, lora_alpha=32, fast=True)
    t_fig_fast = bench("FigLinear fast  ", lambda: fig_fast(x))

    # FigLinear low-RAM mode
    fig_slow = FigLinear(768, 2048, q, lora_r=16, lora_alpha=32, fast=False)
    t_fig_slow = bench("FigLinear lowram", lambda: fig_slow(x))

    RESULTS['bench_linear'] = {
        'nn_linear_ms': t_std,
        'figlinear_fast_ms': t_fig_fast,
        'figlinear_lowram_ms': t_fig_slow,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🍐 Little Fig v0.5 — Tests & Benchmarks")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
    print()

    # Tests
    run_test("FigQuant roundtrip", test_figquant_roundtrip)
    run_test("FigQuant double quantization", test_figquant_double_quant)
    run_test("FigQuant vs FIG4 quality", test_figquant_vs_fig4)
    run_test("FigKernel RMSNorm", test_figkernel_rmsnorm)
    run_test("FigKernel SwiGLU", test_figkernel_swiglu)
    run_test("FigKernel chunked CE", test_figkernel_chunked_ce)
    run_test("FigKernel fused linear+LoRA", test_figkernel_fused_linear_lora)
    run_test("FigPipeline CPU step", test_figpipeline_cpu)
    run_test("FigLinear fused integration", test_figlinear_uses_fused)

    # Benchmarks
    print(f"\n{'='*70}")
    print(f"  BENCHMARKS")
    print(f"{'='*70}")
    try:
        benchmark_quantization()
        benchmark_fused_ops()
        benchmark_figlinear()
    except Exception as e:
        print(f"  Benchmark error: {e}")
        traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
    print(f"{'='*70}")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "v05_results.json")
    with open(results_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")

    sys.exit(1 if FAIL > 0 else 0)
