# Theria Examples

This directory contains example scripts demonstrating how to use theria's attention operators.

## Available Examples

### [minimal_decoder_transformer.py](minimal_decoder_transformer.py)

A complete, runnable example of a decoder-only transformer using theria's fully fused attention.

**What it demonstrates:**
- How to call `sdpa_custom` with `backend="triton_full_fused"`
- Proper multi-head attention reshaping: (B,T,D) → (B,H,T,D//H)
- Complete transformer architecture with residuals and LayerNorm
- Constraint validation (D ≤ 64, CUDA requirement, contiguous tensors)
- Training loop with forward and backward passes

**Quick start:**
```bash
# Quick API test (random data, 20 steps)
python examples/minimal_decoder_transformer.py

# See actual learning (synthetic task with patterns)
python examples/minimal_decoder_transformer.py --use-synthetic-task --steps 200

# CPU testing with reference backend
python examples/minimal_decoder_transformer.py --backend reference --device cpu

# Verbose mode (shows shapes and gradient flow)
python examples/minimal_decoder_transformer.py --verbose --steps 5
```

**Random vs Learnable Task:**
```bash
# Random task (no patterns - loss fluctuates)
python examples/minimal_decoder_transformer.py --steps 200 --seed 42
# Expected: Loss ~1.6 → ~1.4 (small decrease, mostly noise)

# Synthetic task (learnable patterns - loss decreases)
python examples/minimal_decoder_transformer.py --use-synthetic-task --steps 200 --seed 42
# Expected: Loss ~1.5 → ~0.03 (98% reduction - actual learning!)
```

**Backend comparison:**
```bash
# Compare reference vs fused backend (should give similar losses)
python examples/minimal_decoder_transformer.py --backend reference --steps 10 --seed 42
python examples/minimal_decoder_transformer.py --backend triton_full_fused --steps 10 --seed 42
```

**Architecture:**
- 2-layer decoder-only transformer
- Multi-head self-attention (4 heads, D=16 per head)
- Pre-norm residual blocks
- Sequence classification (token 0 pooling)

**Task options:**
1. **Random (default)**: Completely random tokens → random labels
   - No learnable patterns
   - Loss fluctuates, doesn't decrease
   - Fast sanity check for API correctness

2. **Synthetic (`--use-synthetic-task`)**: Token patterns → class labels
   - Signal tokens 95-99 determine class (Token 95→Class 0, etc.)
   - Model must learn to find signal token and classify
   - Loss decreases dramatically (~1.5 → 0.03 in 200 steps)
   - Demonstrates actual learning

**Key implementation details:**
```python
# CRITICAL: Must call .contiguous() before fused attention
q = q.transpose(1, 2).contiguous()  # (B, T, H, D//H) → (B, H, T, D//H)
k = k.transpose(1, 2).contiguous()
v = v.transpose(1, 2).contiguous()

# Call fused attention
attn_out = sdpa_custom(q, k, v, backend="triton_full_fused")

# CRITICAL: For multi-head attention, ensure gradient contiguity
# Transpose backward creates non-contiguous gradients, but triton requires contiguous dout
# Use a custom autograd wrapper (see _ContiguousGrad in the example)
attn_out = _ensure_contiguous_grad(attn_out)
```

**Constraints for `triton_full_fused` backend:**
- D ≤ 64 per attention head
- CUDA tensors required
- Inputs must be contiguous
- **Gradients must be contiguous** (handled automatically by `_ensure_contiguous_grad`)

---

## Important: Gradient Contiguity for Multi-Head Attention

**Why this matters:**

Multi-head attention reshapes tensors using transpose operations:
```
Forward:  (B,T,D) → reshape → (B,T,H,D//H) → transpose → (B,H,T,D//H)
Backward: (B,H,T,D//H) ← transpose ← gradients become NON-CONTIGUOUS
```

**The problem:** Triton backward kernels require `dout` (gradient w.r.t. output) to be contiguous, but transpose backward creates non-contiguous gradients. This causes:
```
AssertionError: dout must be contiguous for Phase9 backward
```

**The solution:** Use a custom autograd wrapper that ensures gradient contiguity:
```python
class _ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x  # Identity (no overhead)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()  # Fix non-contiguous gradients

# Apply right after sdpa_custom()
attn_out = sdpa_custom(q, k, v, backend="triton_full_fused")
attn_out = _ensure_contiguous_grad(attn_out)  # ✅ Ensures contiguous gradients
```

**Alternative:** Use single-head attention with `unsqueeze` instead of transpose (avoids the issue entirely):
```python
# Single-head approach (like theria's TinyAttentionModel)
q = q.unsqueeze(1)  # (B, 1, T, D) - no transpose needed
k = k.unsqueeze(1)
v = v.unsqueeze(1)
attn_out = sdpa_custom(q, k, v, backend="triton_full_fused")  # (B, 1, T, D)
attn_out = attn_out.squeeze(1)  # (B, T, D) - no gradient contiguity issue
```

---

## Usage Tips

1. **Start with CPU + reference backend** to verify correctness:
   ```bash
   python examples/minimal_decoder_transformer.py --backend reference --device cpu --steps 5
   ```

2. **Then test GPU + fused backend**:
   ```bash
   python examples/minimal_decoder_transformer.py --backend triton_full_fused --steps 5
   ```

3. **Use verbose mode** to debug shape mismatches:
   ```bash
   python examples/minimal_decoder_transformer.py --verbose
   ```

4. **Test constraint violations** to understand error messages:
   ```bash
   # D > 64 per head (should fail with clear error)
   python examples/minimal_decoder_transformer.py --d_model 128 --n_heads 1

   # CUDA required for fused (should fail with clear error)
   python examples/minimal_decoder_transformer.py --backend triton_full_fused --device cpu
   ```

## Troubleshooting

### Error: `AssertionError: dout must be contiguous for Phase9 backward`

**Cause:** Gradient flowing back through transpose operations is non-contiguous.

**Fix:** Use `_ensure_contiguous_grad()` wrapper after `sdpa_custom()` (see example implementation).

### Error: `AssertionError: D must be ≤ 64 for Phase9 backward`

**Cause:** Per-head dimension exceeds 64 (current kernel limitation).

**Fix:** Either:
- Decrease `d_model`: `--d_model 64`
- Increase `n_heads`: `--n_heads 4` (so d_model/n_heads ≤ 64)

### Error: `Fused backend requires CUDA tensors`

**Cause:** Trying to use `triton_full_fused` backend on CPU.

**Fix:** Either:
- Use `--device cuda` (requires GPU)
- Use `--backend reference` for CPU testing

### Numerical Differences Between Backends

**Expected behavior:**
- Reference vs Fused with same seed: **bit-identical** for first few steps
- Small divergence (~1e-4) after many steps due to floating-point accumulation order
- Loss trajectories should be similar (not identical)

**Example:**
```bash
# Same seed, both backends
$ python examples/minimal_decoder_transformer.py --backend reference --seed 42 --steps 5
Step   0 | Loss: 1.6555
Step   4 | Loss: 1.9228

$ python examples/minimal_decoder_transformer.py --backend triton_full_fused --seed 42 --steps 5
Step   0 | Loss: 1.6555
Step   4 | Loss: 1.9228  # Identical!
```

### Loss Behavior: Random vs Synthetic Task

**Random task (default):**
```bash
$ python examples/minimal_decoder_transformer.py --steps 200
Step   0 | Loss: 1.6555
Step  50 | Loss: 1.8563  ← Fluctuating
Step 100 | Loss: 1.5751
Step 150 | Loss: 1.8969
Step 199 | Loss: 1.4044  ← Still high
Loss decrease: 0.2511 (15% - just noise, no real learning)
```

**Synthetic task (--use-synthetic-task):**
```bash
$ python examples/minimal_decoder_transformer.py --use-synthetic-task --steps 200
Step   0 | Loss: 1.5354
Step  50 | Loss: 1.7764  ← Starting to learn
Step 100 | Loss: 1.0847  ← Clear progress
Step 150 | Loss: 0.0575  ← Near-perfect!
Step 199 | Loss: 0.0289  ← Converged
Loss decrease: 1.5065 (98% - actual learning!)
```

**Why the difference?**
- **Random**: No patterns to learn, model overfits to noise
- **Synthetic**: Clear rule (signal token → class), model learns the pattern

### Performance Expectations

**CPU (reference backend):**
- Training step: ~50-100ms (depends on CPU)
- Use for correctness checking, not performance

**GPU (triton_full_fused backend):**
- Training step: ~10-20ms (depends on GPU)
- 2-5x faster than reference on typical GPUs
- Memory: O(N) instead of O(N²) for sequence length N

**Note:** These are toy model timings. Real performance gains appear at larger scales (T > 512, D = 64, multiple layers).

---

## Contributing

To add a new example:
1. Create a standalone `.py` file in this directory
2. Include comprehensive docstring at the top
3. Add CLI arguments for flexibility
4. Include validation and error handling
5. Test both CPU (reference) and GPU (fused) backends
6. Document any gradient contiguity requirements
7. Update this README

---

## Reference

For the full API and operator contracts, see:
- Main documentation: [`docs/STATUS.md`](../docs/STATUS.md)
- Attention API: [`theria/attention/custom.py`](../theria/attention/custom.py)
- Operator contracts: [`docs/design/`](../docs/design/)
- Phase 9 backward contracts: [`docs/phase9_backward.md`](../docs/phase9_backward.md)
