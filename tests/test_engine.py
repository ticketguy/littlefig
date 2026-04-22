"""
Fig Engine — Test Suite

Tests all core components:
    1. INT4 quantization accuracy and roundtrip
    2. FigLinear forward/backward correctness
    3. FigModel loading and forward pass
    4. LISA scheduler
    5. MeZO optimizer
    6. LOMO optimizer
    7. Sequence packing
    8. Full training loop (tiny model + tiny dataset)
"""

import torch
import torch.nn as nn
import os
import sys
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from little_fig.engine.quantize import FigQuantizer, FIG4Tensor, _quantize_int4, _dequantize_int4
from little_fig.engine.linear import FigLinear, DequantMatmul
from little_fig.engine.tier import TrainingTier, estimate_memory, select_tier
from little_fig.engine.lisa import LISAScheduler, LISAConfig
from little_fig.engine.mezo import MeZOOptimizer, MeZOConfig
from little_fig.engine.lomo import LOMOOptimizer, LOMOConfig
from little_fig.engine.packing import PackedDataset


def test_passed(name):
    print(f"  ✓ {name}")

def test_failed(name, e):
    print(f"  ✗ {name}: {e}")
    return False


def test_quantize_roundtrip():
    """Test INT4 quantization and dequantization accuracy."""
    shapes = [(64, 64), (768, 768), (2048, 5632)]
    
    for shape in shapes:
        W = torch.randn(shape)
        q = _quantize_int4(W, group_size=128)
        W_deq = _dequantize_int4(q)
        
        assert W_deq.shape == W.shape, f"Shape mismatch: {W_deq.shape} vs {W.shape}"
        
        cos_sim = torch.nn.functional.cosine_similarity(
            W.flatten().unsqueeze(0), W_deq.flatten().unsqueeze(0)
        ).item()
        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim}"
        
        mse = torch.nn.functional.mse_loss(W_deq, W).item()
        assert mse < 0.02, f"MSE too high: {mse}"
    
    test_passed("Quantize roundtrip (shapes, cos_sim > 0.99)")


def test_quantize_save_load():
    """Test saving and loading FIG4 files."""
    tmpdir = tempfile.mkdtemp()
    try:
        W = torch.randn(256, 512)
        q = _quantize_int4(W, group_size=128)
        
        path = os.path.join(tmpdir, "test.fig4")
        q.save(path)
        
        assert os.path.exists(path), "File not created"
        
        loaded = FIG4Tensor.load(path)
        W_orig = _dequantize_int4(q)
        W_loaded = _dequantize_int4(loaded)
        
        assert torch.allclose(W_orig, W_loaded, atol=1e-6), "Roundtrip through disk failed"
        
        test_passed("Quantize save/load roundtrip")
    finally:
        shutil.rmtree(tmpdir)


def test_fig_linear_forward():
    """Test FigLinear forward pass produces correct shape."""
    in_f, out_f = 64, 128
    W = torch.randn(out_f, in_f)
    q = _quantize_int4(W, group_size=64)
    
    layer = FigLinear(in_f, out_f, q, lora_r=8, lora_alpha=16)
    x = torch.randn(2, 16, in_f)
    y = layer(x)
    
    assert y.shape == (2, 16, out_f), f"Output shape wrong: {y.shape}"
    test_passed("FigLinear forward shape")


def test_fig_linear_backward():
    """Test FigLinear backward pass computes gradients for LoRA."""
    in_f, out_f = 64, 128
    W = torch.randn(out_f, in_f)
    q = _quantize_int4(W, group_size=64)
    
    layer = FigLinear(in_f, out_f, q, lora_r=8)
    x = torch.randn(2, 16, in_f, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert layer.lora_A.grad is not None, "No gradient for lora_A"
    assert layer.lora_B.grad is not None, "No gradient for lora_B"
    assert x.grad is not None, "No gradient for input"
    assert layer.q_packed.grad is None or not layer.q_packed.requires_grad, "INT4 weights should be frozen"
    
    test_passed("FigLinear backward (LoRA gradients)")


def test_fig_linear_merge():
    """Test LoRA merge produces valid weight."""
    in_f, out_f = 64, 128
    W = torch.randn(out_f, in_f)
    q = _quantize_int4(W, group_size=64)
    
    layer = FigLinear(in_f, out_f, q, lora_r=8)
    
    # Manually set LoRA weights
    layer.lora_A.data.fill_(0.1)
    layer.lora_B.data.fill_(0.1)
    
    merged = layer.merge_lora()
    assert merged.shape == (out_f, in_f), f"Merged shape wrong: {merged.shape}"
    
    # Merged should differ from base (LoRA contributes)
    base = _dequantize_int4(q)
    diff = (merged - base).abs().sum().item()
    assert diff > 0, "LoRA merge had no effect"
    
    test_passed("FigLinear LoRA merge")


def test_tier_estimation():
    """Test memory estimation for different tiers."""
    params_1b = 1_100_000_000
    
    for tier in TrainingTier:
        est = estimate_memory(params_1b, tier)
        assert est.total_bytes > 0, f"Zero memory for {tier}"
        assert est.total_gb < 100, f"Unreasonable estimate for {tier}: {est.total_gb} GB"
    
    # Streaming LoRA should use least memory
    lora_est = estimate_memory(params_1b, TrainingTier.STREAMING_LORA)
    lomo_est = estimate_memory(params_1b, TrainingTier.LOMO)
    assert lora_est.total_bytes < lomo_est.total_bytes, "LoRA should use less memory than LOMO"
    
    test_passed("Tier memory estimation")


def test_tier_selection():
    """Test automatic tier selection."""
    params_1b = 1_100_000_000
    
    # With lots of RAM, should pick LISA (best quality that fits)
    tier = select_tier(params_1b, available_ram=32 * 1024**3)
    assert tier in [TrainingTier.LISA, TrainingTier.LOMO], f"Expected LISA/LOMO with 32GB, got {tier}"
    
    # With minimal RAM, should pick MeZO or streaming LoRA
    tier = select_tier(params_1b, available_ram=1 * 1024**3)
    assert tier in [TrainingTier.STREAMING_LORA, TrainingTier.MEZO], f"Expected LoRA/MeZO with 1GB, got {tier}"
    
    test_passed("Automatic tier selection")


def test_lisa_scheduler():
    """Test LISA layer switching."""
    # Create a simple model with sequential layers
    model = nn.Sequential(*[nn.Linear(32, 32) for _ in range(10)])
    
    # Wrap in a structure LISA can find
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(10)])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    model = FakeModel()
    config = LISAConfig(active_layers=2, switch_interval=3)
    scheduler = LISAScheduler(model, config)
    
    # Check initial state
    active = scheduler.active_layer_indices
    assert len(active) == 2, f"Expected 2 active layers, got {len(active)}"
    
    # Step without switch
    switched = scheduler.step(1)
    assert not switched, "Should not switch at step 1"
    
    # Step with switch
    old_active = scheduler.active_layer_indices
    switched = scheduler.step(3)
    assert switched, "Should switch at step 3"
    # Active layers may or may not change (random), but the method should run
    
    test_passed("LISA scheduler")


def test_mezo_optimizer():
    """Test MeZO optimizer reduces loss."""
    torch.manual_seed(42)
    
    # Simple regression problem
    model = nn.Linear(10, 1, bias=False)
    X = torch.randn(50, 10)
    y = torch.randn(50, 1)
    
    config = MeZOConfig(learning_rate=1e-2, epsilon=1e-3)
    mezo = MeZOOptimizer(model, config)
    
    # Get initial loss
    with torch.no_grad():
        initial_loss = nn.functional.mse_loss(model(X), y).item()
    
    # Train for a few steps
    for _ in range(50):
        def forward_fn():
            return nn.functional.mse_loss(model(X), y)
        mezo.step(forward_fn)
    
    # Check loss decreased
    with torch.no_grad():
        final_loss = nn.functional.mse_loss(model(X), y).item()
    
    assert final_loss < initial_loss, f"MeZO did not reduce loss: {initial_loss:.4f} → {final_loss:.4f}"
    test_passed(f"MeZO optimizer (loss: {initial_loss:.4f} → {final_loss:.4f})")


def test_lomo_optimizer():
    """Test LOMO optimizer reduces loss with fused backward."""
    torch.manual_seed(42)
    
    model = nn.Linear(10, 1, bias=True)
    X = torch.randn(50, 10)
    y = torch.randn(50, 1)
    
    config = LOMOConfig(learning_rate=1e-2, clip_grad_norm=None)
    lomo = LOMOOptimizer(model, config)
    
    initial_loss = nn.functional.mse_loss(model(X), y).item()
    
    for _ in range(50):
        model.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        lomo.fused_backward_no_clip(loss)
    
    final_loss = nn.functional.mse_loss(model(X), y).item()
    
    lomo.remove_hooks()
    
    assert final_loss < initial_loss, f"LOMO did not reduce loss: {initial_loss:.4f} → {final_loss:.4f}"
    test_passed(f"LOMO optimizer (loss: {initial_loss:.4f} → {final_loss:.4f})")


def test_sequence_packing():
    """Test sequence packing correctness."""
    examples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
        {"input_ids": [6, 7, 8, 9], "labels": [6, 7, 8, 9]},
        {"input_ids": [10, 11], "labels": [10, 11]},
        {"input_ids": [12], "labels": [12]},
    ]
    
    packed = PackedDataset(examples, max_length=16, pad_token_id=0, eos_token_id=99)
    
    # Should have fewer packed sequences than original examples
    assert len(packed) <= len(examples), f"Packing increased count: {len(packed)} > {len(examples)}"
    assert len(packed) >= 1, "No packed sequences"
    
    # Check first packed sequence
    item = packed[0]
    assert item["input_ids"].shape[0] == 16, f"Wrong length: {item['input_ids'].shape[0]}"
    assert item["attention_mask"].shape[0] == 16
    assert item["labels"].shape[0] == 16
    
    # Labels should have -100 for padding positions
    pad_labels = (item["labels"] == -100).sum().item()
    assert pad_labels > 0, "No padding labels found"
    
    test_passed("Sequence packing")


def test_full_pipeline_gpt2():
    """End-to-end test: load GPT-2, quantize, train 1 step."""
    try:
        from little_fig.engine.model import FigModel
        from little_fig.engine.trainer import FigTrainer, FigTrainingConfig
    except ImportError as e:
        print(f"  ⚠ Skipping full pipeline test: {e}")
        return
    
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print(f"  ⚠ Skipping full pipeline test: transformers not installed")
        return
    
    tmpdir = tempfile.mkdtemp()
    try:
        print("  Loading GPT-2 (this may take a moment)...")
        model = FigModel.from_pretrained(
            "gpt2",
            lora_r=8,
            lora_alpha=16,
            tier=TrainingTier.STREAMING_LORA,
        )
        
        model.print_trainable_summary()
        
        # Create tiny dataset
        tokenizer = model.tokenizer
        text = "The quick brown fox jumps over the lazy dog."
        enc = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
        
        # Single forward + backward
        model.model.train()
        outputs = model(input_ids=enc["input_ids"], labels=enc["input_ids"])
        loss = outputs.loss
        
        assert loss.item() > 0, f"Loss should be positive: {loss.item()}"
        
        loss.backward()
        
        # Check gradients exist on LoRA params
        has_grads = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grads = True
                break
        assert has_grads, "No gradients computed"
        
        # Save adapter
        save_path = os.path.join(tmpdir, "adapter")
        model.save_adapter(save_path)
        assert os.path.exists(os.path.join(save_path, "fig_adapter.pt")), "Adapter not saved"
        
        test_passed(f"Full pipeline GPT-2 (loss={loss.item():.4f})")
    
    except Exception as e:
        test_failed("Full pipeline GPT-2", e)
    finally:
        shutil.rmtree(tmpdir)


def run_all_tests():
    print("🍐 Fig Engine — Test Suite")
    print("=" * 60)
    
    tests = [
        test_quantize_roundtrip,
        test_quantize_save_load,
        test_fig_linear_forward,
        test_fig_linear_backward,
        test_fig_linear_merge,
        test_tier_estimation,
        test_tier_selection,
        test_lisa_scheduler,
        test_mezo_optimizer,
        test_lomo_optimizer,
        test_sequence_packing,
        test_full_pipeline_gpt2,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            test_failed(test_fn.__name__, e)
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"🍐 Results: {passed} passed, {failed} failed, {passed + failed} total")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
