"""
Attention-only benchmark for sdpa_custom backends.

Purpose:
- isolate raw SDPA kernel/runtime behavior from full-model orchestration
- compare forward-only and forward+backward timing
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from theria.attention.custom import sdpa_custom


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_backend(name: str) -> str:
    if name == "triton_fused":
        return "triton_full_fused"
    return name


def _measure_cuda(fn, warmup: int, steps: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end))


def _measure_cpu(fn, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    return float((time.perf_counter() - t0) * 1000.0)


def run_one(
    *,
    backend: str,
    mode: str,
    b: int,
    h: int,
    t: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    steps: int,
    seed: int,
) -> dict[str, str | int | float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    q = torch.randn((b, h, t, d), device=device, dtype=dtype, requires_grad=(mode == "fwd_bwd")).contiguous()
    k = torch.randn((b, h, t, d), device=device, dtype=dtype, requires_grad=(mode == "fwd_bwd")).contiguous()
    v = torch.randn((b, h, t, d), device=device, dtype=dtype, requires_grad=(mode == "fwd_bwd")).contiguous()
    resolved_backend = _resolve_backend(backend)

    if mode == "forward":
        def _step():
            _ = sdpa_custom(q, k, v, backend=resolved_backend)
    else:
        # Keep gradient output fixed/contiguous so backward path cost is stable.
        dout = torch.randn((b, h, t, d), device=device, dtype=dtype).contiguous()

        def _step():
            q.grad = None
            k.grad = None
            v.grad = None
            out = sdpa_custom(q, k, v, backend=resolved_backend)
            torch.autograd.backward(out, dout)

    if device.type == "cuda":
        total_ms = _measure_cuda(_step, warmup, steps, device)
    else:
        total_ms = _measure_cpu(_step, warmup, steps)

    return {
        "backend": backend,
        "resolved_backend": resolved_backend,
        "mode": mode,
        "B": b,
        "H": h,
        "T": t,
        "D": d,
        "dtype": str(dtype).replace("torch.", ""),
        "warmup": warmup,
        "steps": steps,
        "status": "OK",
        "total_ms": total_ms,
        "ms_per_step": total_ms / max(steps, 1),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", type=str, default="reference,triton_full_fused")
    parser.add_argument("--modes", type=str, default="forward,fwd_bwd")
    parser.add_argument("--batch-sizes", type=str, default="8,16,32")
    parser.add_argument("--seq-lens", type=str, default="32,64,128")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--csv-out", type=str, default="experiments/phase12/runs/attention_only_bench.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    backends = _parse_str_list(args.backends)
    modes = _parse_str_list(args.modes)
    batch_sizes = _parse_int_list(args.batch_sizes)
    seq_lens = _parse_int_list(args.seq_lens)

    rows: list[dict[str, str | int | float]] = []
    total = len(backends) * len(modes) * len(batch_sizes) * len(seq_lens)
    idx = 0

    for backend in backends:
        for mode in modes:
            for b in batch_sizes:
                for t in seq_lens:
                    idx += 1
                    try:
                        row = run_one(
                            backend=backend,
                            mode=mode,
                            b=b,
                            h=args.heads,
                            t=t,
                            d=args.head_dim,
                            device=device,
                            dtype=dtype,
                            warmup=args.warmup,
                            steps=args.steps,
                            seed=args.seed,
                        )
                    except Exception as e:
                        row = {
                            "backend": backend,
                            "resolved_backend": _resolve_backend(backend),
                            "mode": mode,
                            "B": b,
                            "H": args.heads,
                            "T": t,
                            "D": args.head_dim,
                            "dtype": str(dtype).replace("torch.", ""),
                            "warmup": args.warmup,
                            "steps": args.steps,
                            "status": "HARD_FAIL",
                            "total_ms": float("nan"),
                            "ms_per_step": float("nan"),
                            "error": repr(e),
                        }
                    rows.append(row)
                    print(
                        f"[{idx}/{total}] backend={backend} mode={mode} "
                        f"B={b} T={t} D={args.head_dim} status={row['status']} "
                        f"ms/step={row['ms_per_step']}"
                    )

    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "resolved_backend",
                "mode",
                "B",
                "H",
                "T",
                "D",
                "dtype",
                "warmup",
                "steps",
                "status",
                "total_ms",
                "ms_per_step",
                "error",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

