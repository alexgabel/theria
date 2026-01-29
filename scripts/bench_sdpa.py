import argparse
import time

import torch

from theria.attention.custom import sdpa_custom

# Performance note:
# The correctness-first Triton path is slow; fast modes are opt-in and may drift.


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


PRESETS = {
    "small": dict(B=4, H=4, T=128, M=128, D=64),
    "medium": dict(B=8, H=8, T=256, M=256, D=64),
    "large": dict(B=2, H=16, T=1024, M=1024, D=64),
}


def main():
    parser = argparse.ArgumentParser(description="Bench SDPA reference vs Triton QK")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default=None)
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark")

    cfg = PRESETS.get(args.preset, {})
    B = args.B if args.B is not None else cfg.get("B", 4)
    H = args.H if args.H is not None else cfg.get("H", 4)
    T = args.T if args.T is not None else cfg.get("T", 128)
    M = args.M if args.M is not None else cfg.get("M", 128)
    D = args.D if args.D is not None else cfg.get("D", 64)

    dtype = getattr(torch, args.dtype)
    q, k, v = make_inputs(B, H, T, M, D, dtype=dtype, device="cuda")

    ref = lambda: sdpa_custom(q, k, v, backend="reference")
    tri_ref = lambda: sdpa_custom(q, k, v, backend="triton_ref")
    tri_fast = lambda: sdpa_custom(q, k, v, backend="triton_fast")

    t_ref = bench(ref, warmup=args.warmup, iters=args.iters)
    t_tri_ref = bench(tri_ref, warmup=args.warmup, iters=args.iters)
    t_tri_fast = bench(tri_fast, warmup=args.warmup, iters=args.iters)

    print(f"B={B} H={H} T={T} M={M} D={D} dtype={args.dtype}")
    print(f"reference   : {t_ref*1e3:.3f} ms/iter")
    print(f"triton_ref  : {t_tri_ref*1e3:.3f} ms/iter")
    print(f"triton_fast : {t_tri_fast*1e3:.3f} ms/iter")


if __name__ == "__main__":
    main()
