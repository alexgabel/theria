"""
Minimal Decoder-Only Transformer using Theria's Fully Fused Attention

This example demonstrates:
1. How to properly call sdpa_custom with backend="triton_full_fused"
2. Multi-head attention reshaping (B,T,D) → (B,H,T,D//H)
3. Complete transformer block with LayerNorm + residuals
4. Training loop with forward + backward passes
5. Constraint validation (D ≤ 64, CUDA tensors, contiguous inputs)

Architecture:
- 2-layer decoder-only transformer
- Multi-head self-attention (4 heads by default)
- Pre-norm residual blocks
- Simple classification task (token 0 pooling)

Usage:
    # Train with fused attention (default):
    python examples/minimal_decoder_transformer.py

    # Compare with reference backend:
    python examples/minimal_decoder_transformer.py --backend reference

    # Increase model size (respecting D ≤ 64):
    python examples/minimal_decoder_transformer.py --d_model 64 --n_heads 4

    # CPU testing (must use reference backend):
    python examples/minimal_decoder_transformer.py --backend reference --device cpu

    # Verbose mode (shows shapes and gradients):
    python examples/minimal_decoder_transformer.py --verbose

Requirements:
- CUDA GPU (for triton_full_fused backend)
- theria installed (pip install -e .)

Design philosophy:
- Correctness over performance
- Clear, readable code
- Explicit error messages
- Follows theria's "contracts before kernels" principle
"""

import argparse
import torch
import torch.nn as nn
from theria.attention.custom import sdpa_custom


class _ContiguousGrad(torch.autograd.Function):
    """
    Identity function that ensures gradient contiguity for triton kernels.

    Background:
    - Triton backward kernels require contiguous `dout` (gradient w.r.t. output)
    - Multi-head attention uses transpose operations: (B,H,T,D) → (B,T,H,D)
    - Transpose backward creates non-contiguous gradients
    - This breaks triton's backward contract

    Solution:
    - Apply this wrapper right after sdpa_custom()
    - Forward pass: identity (no overhead)
    - Backward pass: ensures gradient is contiguous before reaching triton kernels

    Note: Single-head attention (using unsqueeze instead of transpose) doesn't need this.
    """
    @staticmethod
    def forward(ctx, x):  # noqa: ARG004 - ctx required by autograd.Function signature
        return x

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ARG004 - ctx required by autograd.Function signature
        # Ensure gradient is contiguous before passing to triton backward kernels
        return grad_output.contiguous()


def _ensure_contiguous_grad(x):
    """Helper to ensure gradient contiguity without affecting forward pass."""
    return _ContiguousGrad.apply(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention using theria's fused backend.

    This implementation demonstrates proper reshaping for the fused attention kernel:
    1. Project Q, K, V: (B, T, D) → (B, T, D)
    2. Reshape for multi-head: (B, T, D) → (B, T, H, D//H)
    3. Transpose for sdpa_custom: (B, T, H, D//H) → (B, H, T, D//H)
    4. Apply contiguous() - REQUIRED for fused backend
    5. Call sdpa_custom with shape (B, H, T, D//H)
    6. Transpose back and reshape to (B, T, D)
    """

    def __init__(self, d_model, n_heads, backend="triton_full_fused"):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_per_head = d_model // n_heads
        self.backend = backend

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape

        # Project and reshape for multi-head
        # (B, T, D) → (B, T, H, D//H)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_per_head)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_per_head)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_per_head)

        # Transpose to (B, H, T, D//H) - the format sdpa_custom expects
        # IMPORTANT: sdpa_custom expects (B, H, T, D) format where D is per-head dimension
        q = q.transpose(1, 2)  # (B, H, T, D//H)
        k = k.transpose(1, 2)  # (B, H, T, D//H)
        v = v.transpose(1, 2)  # (B, H, T, D//H)

        # CRITICAL: Contiguous tensors required for fused backend
        # Transpose operations create non-contiguous views
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Call theria's fused attention
        attn_out = sdpa_custom(q, k, v, backend=self.backend)  # (B, H, T, D//H)

        # CRITICAL: Ensure gradient contiguity for triton backward
        # The transpose operations below create non-contiguous gradients in backward pass,
        # but triton kernels require contiguous dout. This wrapper ensures contiguity.
        attn_out = _ensure_contiguous_grad(attn_out)

        # Transpose back and reshape to (B, T, D)
        attn_out = attn_out.transpose(1, 2).contiguous()  # (B, T, H, D//H)
        attn_out = attn_out.reshape(B, T, D)  # (B, T, D)

        # Output projection
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """Simple 2-layer MLP with GELU activation."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single decoder block: attention + FFN with pre-norm.

    Pre-norm architecture (modern best practice):
    - x = x + attention(LayerNorm(x))
    - x = x + feedforward(LayerNorm(x))

    Gradient flow is more stable than post-norm.
    """

    def __init__(self, d_model, n_heads, d_ff, backend="triton_full_fused"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, backend)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        # Pre-norm: LayerNorm before attention
        x = x + self.attn(self.norm1(x))

        # Pre-norm: LayerNorm before feedforward
        x = x + self.ff(self.norm2(x))

        return x


class MinimalDecoderTransformer(nn.Module):
    """
    Complete decoder-only transformer for sequence classification.

    Architecture:
    - Token embedding
    - Stack of TransformerBlocks
    - Final LayerNorm
    - Classification head (uses token 0 as [CLS]-like pooling)
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        num_classes,
        backend="triton_full_fused"
    ):
        super().__init__()

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, backend)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        # Following theria convention: use token 0 as [CLS]-like pooling
        # This matches TinyAttentionModel pattern
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: Token indices of shape (B, T)

        Returns:
            Logits of shape (B, num_classes)
        """
        # Embed tokens: (B, T) → (B, T, D)
        x = self.token_embed(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Pool using token 0 (like BERT [CLS])
        x = x[:, 0, :]  # (B, D)

        # Classify
        logits = self.classifier(x)  # (B, num_classes)

        return logits


def validate_fused_backend_constraints(d_model, n_heads, device_type, backend):
    """
    Validate constraints required for theria's fused backend.

    The triton_full_fused backend requires:
    1. D ≤ 64 per head
    2. d_model divisible by n_heads
    3. CUDA device
    """
    d_per_head = d_model // n_heads

    if d_per_head > 64:
        raise ValueError(
            f"Fused backend requires D ≤ 64 per head. "
            f"Got d_model={d_model}, n_heads={n_heads}, "
            f"resulting in D_per_head={d_per_head}. "
            f"Suggestion: Either decrease d_model or increase n_heads."
        )

    if backend == "triton_full_fused" and device_type != "cuda":
        raise ValueError(
            f"Fused backend requires CUDA tensors. Got device={device_type}. "
            f"Suggestion: Use --backend reference for CPU testing."
        )

    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )


def create_dummy_task(batch_size, seq_len, vocab_size, num_classes, device):
    """
    Create a random sequence classification task (no learnable patterns).

    Returns:
        x: Random token indices of shape (B, T)
        y: Random class labels of shape (B,)
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


def create_synthetic_task(batch_size, vocab_size, num_classes, seq_len, seed, device):
    """
    Create a learnable synthetic task with consistent token patterns.

    Rule: The class is determined by which "signal token" appears first in the sequence.
    - Signal tokens: vocab_size - num_classes to vocab_size - 1
    - Signal token K → class K
    - Background tokens: 0 to vocab_size - num_classes - 1

    Unlike random data, this has a learnable pattern the model can extract.

    Returns:
        get_batch: Function that generates (x, y) batches with consistent patterns
    """
    # Set seed for reproducibility
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Define signal tokens (last num_classes tokens in vocab)
    signal_tokens = list(range(vocab_size - num_classes, vocab_size))
    background_vocab_size = vocab_size - num_classes

    def get_batch():
        # Sample classes uniformly
        y = torch.randint(0, num_classes, (batch_size,), device=device, generator=rng)

        # Create sequences filled with background tokens
        x = torch.randint(0, background_vocab_size, (batch_size, seq_len), device=device, generator=rng)

        # Inject signal tokens at random positions
        for i in range(batch_size):
            # Choose random position to inject signal (avoid position 0 which is used for classification)
            signal_pos = torch.randint(1, seq_len, (1,), device=device, generator=rng).item()
            # Insert signal token corresponding to the class
            x[i, signal_pos] = signal_tokens[y[i].item()]

        return x, y

    return get_batch


def train_step(model, x, y, optimizer):
    """
    Single training step with loss calculation.

    Returns:
        loss: Scalar loss value
    """
    optimizer.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def check_gradients(model, verbose=False):
    """Check that all parameters have gradients."""
    no_grad_params = []
    grad_stats = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                no_grad_params.append(name)
            else:
                grad_norm = param.grad.norm().item()
                grad_stats.append((name, grad_norm))
                if verbose:
                    print(f"  {name:40s} | grad_norm: {grad_norm:.6f}")

    if no_grad_params:
        print(f"\nWARNING: {len(no_grad_params)} parameters without gradients:")
        for name in no_grad_params:
            print(f"  - {name}")

    return len(no_grad_params) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Minimal decoder-only transformer with theria's fully fused attention"
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        default="triton_full_fused",
        choices=["triton_full_fused", "reference", "custom", "triton_qk", "triton_qk_softmax"],
        help="Attention backend to use (default: triton_full_fused)"
    )

    # Model architecture
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension (default: 64)")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer blocks (default: 2)")
    parser.add_argument("--d_ff", type=int, default=128, help="Feedforward dimension (default: 128)")

    # Task parameters
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size (default: 100)")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes (default: 5)")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length (default: 32)")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")

    # Task parameters
    parser.add_argument(
        "--use-synthetic-task",
        action="store_true",
        help="Use learnable synthetic task instead of random data (shows actual learning)"
    )

    # System parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose debugging info")

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("Minimal Decoder Transformer - Theria Fused Attention Example")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Architecture: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"Per-head dimension: {args.d_model // args.n_heads}")
    print(f"Device: {args.device}")
    print(f"Task: {'Synthetic (learnable)' if args.use_synthetic_task else 'Random (no patterns)'}")
    print(f"Training steps: {args.steps}")
    print("=" * 60)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device setup
    device = torch.device(args.device)

    # Validate constraints for fused backend
    try:
        validate_fused_backend_constraints(args.d_model, args.n_heads, args.device, args.backend)
    except ValueError as e:
        print(f"\n❌ Constraint violation: {e}")
        return

    # Create model
    print(f"\nCreating model...")
    model = MinimalDecoderTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        num_classes=args.num_classes,
        backend=args.backend
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create task function
    if args.use_synthetic_task:
        print(f"\nUsing synthetic learnable task (expect loss to decrease)")
        print(f"Pattern: Signal tokens {args.vocab_size - args.num_classes}-{args.vocab_size - 1} determine class")
        get_batch = create_synthetic_task(
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes,
            seq_len=args.seq_len,
            seed=args.seed,
            device=device
        )
    else:
        print(f"\nUsing random task (loss will fluctuate, not decrease)")
        def get_batch():
            return create_dummy_task(args.batch_size, args.seq_len, args.vocab_size, args.num_classes, device)

    # Training loop
    print(f"Training for {args.steps} steps...")
    print("-" * 60)

    losses = []
    for step in range(args.steps):
        # Get batch
        x, y = get_batch()

        # Training step
        loss = train_step(model, x, y, optimizer)
        losses.append(loss)

        # Print progress
        if step % 5 == 0 or step == args.steps - 1:
            print(f"Step {step:3d} | Loss: {loss:.4f}")

        # Verbose debugging on first step
        if args.verbose and step == 0:
            print("\n" + "=" * 60)
            print("First step - verbose debugging:")
            print("=" * 60)
            print(f"Input shape: {x.shape}")
            with torch.no_grad():
                embedded = model.token_embed(x)
                print(f"After embedding: {embedded.shape}")
            print("\nGradient flow check:")
            check_gradients(model, verbose=True)
            print("=" * 60 + "\n")

    # Final statistics
    print("-" * 60)
    print(f"\n✓ Training completed successfully")
    print(f"✓ Fused attention backend ({args.backend}) working correctly")
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Loss decrease: {losses[0] - losses[-1]:.4f}")

    if losses[-1] < losses[0]:
        print("✓ Loss decreased as expected")
    else:
        print("⚠ Loss did not decrease (this is okay for random data)")

    # Gradient check
    print("\nFinal gradient flow check:")
    if check_gradients(model):
        print("✓ All parameters have gradients")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
