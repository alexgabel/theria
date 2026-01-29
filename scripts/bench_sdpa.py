import argparse
import time

import torch

from theria.attention.custom import sdpa_custom

# Performance note:
# The Phase 6 Triton QK kernel is correctness-first and intentionally disables
# tensor cores, fusion, and TF32. Benchmarks will be much slower than cuBLAS.
# This is expected. Performance optimization is deferred to Phase 7+.


def bench(fn, warmup=10, iters=50):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters


def make_inputs(B=4, H=4, T=128, M=128, D=64, dtype=torch.float16, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, M, D, device=device, dtype=dtype)
    v = torch.randn(B, H, M, D, device=device, dtype=dtype)
    return q, k, v


def main():
    parser = argparse.ArgumentParser(description="Bench SDPA reference vs Triton QK")
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark")

    dtype = getattr(torch, args.dtype)
    q, k, v = make_inputs(args.B, args.H, args.T, args.M, args.D, dtype=dtype, device="cuda")

    ref = lambda: sdpa_custom(q, k, v, backend="reference")
    tri = lambda: sdpa_custom(q, k, v, backend="triton")

    t_ref = bench(ref, warmup=args.warmup, iters=args.iters)
    t_tri = bench(tri, warmup=args.warmup, iters=args.iters)

    print(f"B={args.B} H={args.H} T={args.T} M={args.M} D={args.D} dtype={args.dtype}")
    print(f"reference: {t_ref*1e3:.3f} ms/iter")
    print(f"triton   : {t_tri*1e3:.3f} ms/iter")


if __name__ == "__main__":
    main()
