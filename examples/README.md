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
# Default: fused attention on CUDA (requires GPU)
python examples/minimal_decoder_transformer.py

# CPU testing with reference backend
python examples/minimal_decoder_transformer.py --backend reference --device cpu

# Verbose mode (shows shapes and gradient flow)
python examples/minimal_decoder_transformer.py --verbose --steps 5
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
- Simple sequence classification task

**Key implementation details:**
```python
# CRITICAL: Must call .contiguous() before fused attention
q = q.transpose(1, 2).contiguous()  # (B, T, H, D//H) → (B, H, T, D//H)
k = k.transpose(1, 2).contiguous()
v = v.transpose(1, 2).contiguous()

# Call fused attention
attn_out = sdpa_custom(q, k, v, backend="triton_full_fused")
```

**Constraints for `triton_full_fused` backend:**
- D ≤ 64 per attention head
- CUDA tensors required
- Inputs must be contiguous

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

## Contributing

To add a new example:
1. Create a standalone `.py` file in this directory
2. Include comprehensive docstring at the top
3. Add CLI arguments for flexibility
4. Include validation and error handling
5. Update this README

## Reference

For the full API and operator contracts, see:
- Main documentation: [`docs/STATUS.md`](../docs/STATUS.md)
- Attention API: [`theria/attention/custom.py`](../theria/attention/custom.py)
- Operator contracts: [`docs/design/`](../docs/design/)
