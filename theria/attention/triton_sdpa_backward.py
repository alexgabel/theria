"""
Triton backward kernels for fused SDPA (Phase 9 scaffolding).

Phase 9 is frozen; Phase 10 may only apply bugfixes here.

This file currently implements sdpa_bwd_dv: dV = P^T @ dO
where P is reconstructed blockwise using saved row-wise m and l.

Constraints (v0):
- Dv == D
- D, Dv <= 64
- Contiguous inputs
"""

from __future__ import annotations

import os
import torch
import triton
import triton.language as tl


_ALLOWED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_JVP_HVP_ENV = "THERIA_TRITON_JVP_HVP"
_SDPA_DEBUG_ENV = "THERIA_TRITON_SDPA_DEBUG"
_SDPA_DEBUG_ASSERT_ENV = "THERIA_TRITON_SDPA_DEBUG_ASSERT"
_SDPA_DEBUG_TINY_L_ENV = "THERIA_TRITON_SDPA_DEBUG_TINY_L_THRESH"
_SDPA_DEBUG_P_ENV = "THERIA_TRITON_SDPA_DEBUG_EXTREME_P_THRESH"
_SDPA_DEBUG_DP_ENV = "THERIA_TRITON_SDPA_DEBUG_EXTREME_DP_THRESH"
_SDPA_DEBUG_DS_ENV = "THERIA_TRITON_SDPA_DEBUG_EXTREME_DS_THRESH"
_SDPA_DEBUG_ROW_SUM_ENV = "THERIA_TRITON_SDPA_DEBUG_ROW_SUM_ERR_THRESH"
_SDPA_DEBUG_DK_REF_ENV = "THERIA_TRITON_SDPA_DEBUG_DK_REF"

_TRITON_SDPA_DEBUG_COUNTERS = {
    "n_audit_calls": 0,
    "audit_call_index": 0,
    "n_nonfinite_m": 0,
    "n_nonfinite_l": 0,
    "n_tiny_l": 0,
    "n_nonfinite_p": 0,
    "n_negative_p": 0,
    "n_p_gt_1": 0,
    "n_row_sum_err": 0,
    "n_extreme_p": 0,
    "n_nonfinite_dp": 0,
    "n_extreme_dp": 0,
    "n_nonfinite_ds": 0,
    "n_extreme_ds": 0,
    "n_nonfinite_dq": 0,
    "n_nonfinite_dk": 0,
    "n_nonfinite_dv": 0,
    "max_abs_p": 0.0,
    "max_p_row_sum_err": 0.0,
    "max_abs_dp": 0.0,
    "max_abs_ds": 0.0,
    "n_dk_ref_before_accumulation": 0,
    "n_dk_ref_accumulation_only": 0,
    "dk_ref_max_abs_diff": 0.0,
}
_TRITON_SDPA_DEBUG_LAST_FAILURE: dict[str, object] = {}


def reset_triton_sdpa_debug_counters() -> None:
    for key, value in list(_TRITON_SDPA_DEBUG_COUNTERS.items()):
        _TRITON_SDPA_DEBUG_COUNTERS[key] = 0.0 if isinstance(value, float) else 0


def reset_triton_sdpa_debug_artifact() -> None:
    _TRITON_SDPA_DEBUG_LAST_FAILURE.clear()


def get_triton_sdpa_debug_counters(*, reset: bool = False) -> dict[str, float]:
    out = {
        key: (float(val) if isinstance(val, float) else int(val))
        for key, val in _TRITON_SDPA_DEBUG_COUNTERS.items()
    }
    if reset:
        reset_triton_sdpa_debug_counters()
    return out


def get_triton_sdpa_debug_artifact(*, reset: bool = False) -> dict[str, object]:
    out = dict(_TRITON_SDPA_DEBUG_LAST_FAILURE)
    if reset:
        reset_triton_sdpa_debug_artifact()
    return out


def _sdpa_debug_enabled() -> bool:
    return os.getenv(_SDPA_DEBUG_ENV, "0") == "1"


def _sdpa_debug_assert_enabled() -> bool:
    return os.getenv(_SDPA_DEBUG_ASSERT_ENV, "0") == "1"


def _sdpa_debug_threshold(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _sdpa_debug_dk_ref_enabled() -> bool:
    return os.getenv(_SDPA_DEBUG_DK_REF_ENV, "0") == "1"


def _sdpa_debug_inc(name: str, count: int) -> None:
    if count <= 0:
        return
    _TRITON_SDPA_DEBUG_COUNTERS[name] += int(count)


def _sdpa_debug_max(name: str, value: float) -> None:
    if not torch.isfinite(torch.tensor(value)):
        return
    _TRITON_SDPA_DEBUG_COUNTERS[name] = max(
        float(_TRITON_SDPA_DEBUG_COUNTERS[name]),
        float(value),
    )


def _sdpa_debug_raise(label: str, *, context: str, count: int | None = None, value: float | None = None) -> None:
    parts = [f"SDPA_DEBUG {label}", f"context={context}"]
    if count is not None:
        parts.append(f"count={count}")
    if value is not None:
        parts.append(f"value={value}")
    raise RuntimeError(" ".join(parts))


def debug_audit_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    scale: float,
    *,
    context: str,
    forward_debug_call_index: int | None = None,
    m_dtype_read: str | None = None,
    l_dtype_read: str | None = None,
) -> None:
    if not _sdpa_debug_enabled():
        return

    _TRITON_SDPA_DEBUG_COUNTERS["n_audit_calls"] += 1
    _TRITON_SDPA_DEBUG_COUNTERS["audit_call_index"] += 1
    audit_call_index = int(_TRITON_SDPA_DEBUG_COUNTERS["audit_call_index"])
    tiny_l_thresh = _sdpa_debug_threshold(_SDPA_DEBUG_TINY_L_ENV, 1e-8)
    extreme_p_thresh = _sdpa_debug_threshold(_SDPA_DEBUG_P_ENV, 1.0001)
    extreme_dp_thresh = _sdpa_debug_threshold(_SDPA_DEBUG_DP_ENV, 1e4)
    extreme_ds_thresh = _sdpa_debug_threshold(_SDPA_DEBUG_DS_ENV, 1e4)
    row_sum_err_thresh = _sdpa_debug_threshold(_SDPA_DEBUG_ROW_SUM_ENV, 1e-3)

    mf = m.detach().float()
    lf = l.detach().float()
    nonfinite_m = int((~torch.isfinite(mf)).sum().item())
    nonfinite_l = int((~torch.isfinite(lf)).sum().item())
    tiny_l = int((torch.isfinite(lf) & (lf <= tiny_l_thresh)).sum().item())
    _sdpa_debug_inc("n_nonfinite_m", nonfinite_m)
    _sdpa_debug_inc("n_nonfinite_l", nonfinite_l)
    _sdpa_debug_inc("n_tiny_l", tiny_l)
    if _sdpa_debug_assert_enabled():
        if nonfinite_m:
            _sdpa_debug_raise("nonfinite_m", context=context, count=nonfinite_m)
        if nonfinite_l:
            _sdpa_debug_raise("nonfinite_l", context=context, count=nonfinite_l)
        if tiny_l:
            _sdpa_debug_raise("tiny_l", context=context, count=tiny_l)

    safe_m = torch.where(torch.isfinite(mf), mf, torch.zeros_like(mf))
    safe_l = torch.where(torch.isfinite(lf), lf, torch.ones_like(lf)).clamp_min(tiny_l_thresh)

    qf = q.detach().float()
    kf = k.detach().float()
    vf = v.detach().float()
    dof = dout.detach().float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    probs = _python_reconstruct_p(scores, safe_m, safe_l)
    prob_nonfinite = int((~torch.isfinite(probs)).sum().item())
    row_sum_errors = (probs.sum(dim=-1) - 1.0).abs()
    row_sum_err = float(row_sum_errors.amax().item())
    abs_p = float(torch.nan_to_num(probs.abs(), nan=0.0, posinf=0.0, neginf=0.0).amax().item())
    negative_p = int((torch.isfinite(probs) & (probs < -1e-6)).sum().item())
    p_gt_1 = int((torch.isfinite(probs) & (probs > extreme_p_thresh)).sum().item())
    failing_rows = row_sum_errors > row_sum_err_thresh
    row_sum_bad = int(failing_rows.sum().item())
    extreme_p = negative_p + p_gt_1 + row_sum_bad
    _sdpa_debug_inc("n_nonfinite_p", prob_nonfinite)
    _sdpa_debug_inc("n_negative_p", negative_p)
    _sdpa_debug_inc("n_p_gt_1", p_gt_1)
    _sdpa_debug_inc("n_row_sum_err", row_sum_bad)
    _sdpa_debug_inc("n_extreme_p", extreme_p)
    _sdpa_debug_max("max_abs_p", abs_p)
    _sdpa_debug_max("max_p_row_sum_err", row_sum_err)
    if row_sum_bad:
        row_entries = []
        row_indices = failing_rows.nonzero(as_tuple=False)[:8]
        for rank, idx in enumerate(row_indices):
            b_idx = int(idx[0].item())
            h_idx = int(idx[1].item())
            t_idx = int(idx[2].item())
            row_scores = scores[b_idx, h_idx, t_idx]
            row_probs = probs[b_idx, h_idx, t_idx]
            row_entries.append(
                {
                    "failing_row_rank": rank,
                    "batch_idx": b_idx,
                    "head_idx": h_idx,
                    "query_row_idx": t_idx,
                    "row_sum_fused": float(row_probs.sum().item()),
                    "row_sum_err_fused": float(row_sum_errors[b_idx, h_idx, t_idx].item()),
                    "m_fused": float(mf[b_idx, h_idx, t_idx].item()),
                    "l_fused": float(lf[b_idx, h_idx, t_idx].item()),
                    "score_min": float(row_scores.min().item()),
                    "score_max": float(row_scores.max().item()),
                    "p_min_fused": float(row_probs.min().item()),
                    "p_max_fused": float(row_probs.max().item()),
                    "n_negative_p_fused": int(
                        (torch.isfinite(row_probs) & (row_probs < -1e-6)).sum().item()
                    ),
                    "n_p_gt_1_fused": int(
                        (torch.isfinite(row_probs) & (row_probs > extreme_p_thresh)).sum().item()
                    ),
                }
            )
        _TRITON_SDPA_DEBUG_LAST_FAILURE.clear()
        _TRITON_SDPA_DEBUG_LAST_FAILURE.update(
            {
                "context": context,
                "audit_call_index": audit_call_index,
                "scale": float(scale),
                "forward_debug_call_index": (
                    int(forward_debug_call_index)
                    if forward_debug_call_index is not None
                    else -1
                ),
                "m_dtype_read": m_dtype_read or "",
                "l_dtype_read": l_dtype_read or "",
                "q": q.detach().cpu(),
                "k": k.detach().cpu(),
                "m": m.detach().cpu(),
                "l": l.detach().cpu(),
                "rows": row_entries,
            }
        )
    if _sdpa_debug_assert_enabled():
        if prob_nonfinite:
            _sdpa_debug_raise("nonfinite_p", context=context, count=prob_nonfinite)
        if negative_p:
            _sdpa_debug_raise("negative_p", context=context, count=negative_p, value=abs_p)
        if p_gt_1:
            _sdpa_debug_raise("p_gt_1", context=context, count=p_gt_1, value=abs_p)
        if row_sum_bad:
            _sdpa_debug_raise("row_sum_err", context=context, count=row_sum_bad, value=row_sum_err)

    dp = torch.matmul(dof, vf.transpose(-2, -1))
    dp_nonfinite = int((~torch.isfinite(dp)).sum().item())
    abs_dp = float(torch.nan_to_num(dp.abs(), nan=0.0, posinf=0.0, neginf=0.0).amax().item())
    extreme_dp = int(
        (torch.isfinite(dp) & (dp.abs() > extreme_dp_thresh)).sum().item()
    )
    _sdpa_debug_inc("n_nonfinite_dp", dp_nonfinite)
    _sdpa_debug_inc("n_extreme_dp", extreme_dp)
    _sdpa_debug_max("max_abs_dp", abs_dp)
    if _sdpa_debug_assert_enabled():
        if dp_nonfinite:
            _sdpa_debug_raise("nonfinite_dp", context=context, count=dp_nonfinite)
        if extreme_dp:
            _sdpa_debug_raise("extreme_dp", context=context, count=extreme_dp, value=abs_dp)

    z = (dp * probs).sum(dim=-1, keepdim=True)
    ds = probs * (dp - z)
    ds_nonfinite = int((~torch.isfinite(ds)).sum().item())
    abs_ds = float(torch.nan_to_num(ds.abs(), nan=0.0, posinf=0.0, neginf=0.0).amax().item())
    extreme_ds = int(
        (torch.isfinite(ds) & (ds.abs() > extreme_ds_thresh)).sum().item()
    )
    _sdpa_debug_inc("n_nonfinite_ds", ds_nonfinite)
    _sdpa_debug_inc("n_extreme_ds", extreme_ds)
    _sdpa_debug_max("max_abs_ds", abs_ds)
    if _sdpa_debug_assert_enabled():
        if ds_nonfinite:
            _sdpa_debug_raise("nonfinite_ds", context=context, count=ds_nonfinite)
        if extreme_ds:
            _sdpa_debug_raise("extreme_ds", context=context, count=extreme_ds, value=abs_ds)


def _debug_check_output(name: str, tensor: torch.Tensor) -> None:
    if not _sdpa_debug_enabled():
        return
    nonfinite = int((~torch.isfinite(tensor)).sum().item())
    if nonfinite <= 0:
        return
    _sdpa_debug_inc(f"n_nonfinite_{name}", nonfinite)
    if _sdpa_debug_assert_enabled():
        _sdpa_debug_raise(f"nonfinite_{name}", context=f"output[{name}]", count=nonfinite)


def _python_reconstruct_p(scores: torch.Tensor, m: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    shifted = torch.minimum(scores - m.unsqueeze(-1), torch.zeros_like(scores))
    denom = l.unsqueeze(-1).clamp_min(1e-12)
    return torch.exp(shifted) / denom


def _debug_check_dk_output(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    scale: float,
    dk: torch.Tensor,
) -> None:
    if not _sdpa_debug_enabled():
        return
    nonfinite = int((~torch.isfinite(dk)).sum().item())
    if nonfinite <= 0:
        if _sdpa_debug_dk_ref_enabled():
            qf = q.detach().float()
            kf = k.detach().float()
            vf = v.detach().float()
            dof = dout.detach().float()
            mf = m.detach().float()
            lf = l.detach().float()
            probs = _python_reconstruct_p(torch.matmul(qf, kf.transpose(-2, -1)) * scale, mf, lf)
            dp = torch.matmul(dof, vf.transpose(-2, -1))
            z = (dp * probs).sum(dim=-1, keepdim=True)
            ds = probs * (dp - z)
            dk_ref = torch.matmul(ds.transpose(-2, -1), qf) * scale
            if torch.isfinite(dk_ref).all():
                diff = float((dk.float() - dk_ref).abs().amax().item())
                _sdpa_debug_max("dk_ref_max_abs_diff", diff)
        return

    _sdpa_debug_inc("n_nonfinite_dk", nonfinite)
    qf = q.detach().float()
    kf = k.detach().float()
    vf = v.detach().float()
    dof = dout.detach().float()
    mf = m.detach().float()
    lf = l.detach().float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    probs = _python_reconstruct_p(scores, mf, lf)
    dp = torch.matmul(dof, vf.transpose(-2, -1))
    z = (dp * probs).sum(dim=-1, keepdim=True)
    ds = probs * (dp - z)
    dk_ref = torch.matmul(ds.transpose(-2, -1), qf) * scale
    ref_probs_finite = bool(torch.isfinite(probs).all())
    ref_dp_finite = bool(torch.isfinite(dp).all())
    ref_ds_finite = bool(torch.isfinite(ds).all())
    ref_dk_finite = bool(torch.isfinite(dk_ref).all())
    probable_stage = "before_accumulation"
    if ref_probs_finite and ref_dp_finite and ref_ds_finite and ref_dk_finite:
        probable_stage = "during_accumulation"
        _sdpa_debug_inc("n_dk_ref_accumulation_only", 1)
        diff = float(
            torch.nan_to_num(dk.float() - dk_ref, nan=0.0, posinf=0.0, neginf=0.0)
            .abs()
            .amax()
            .item()
        )
        _sdpa_debug_max("dk_ref_max_abs_diff", diff)
    else:
        _sdpa_debug_inc("n_dk_ref_before_accumulation", 1)
    if _sdpa_debug_assert_enabled():
        raise RuntimeError(
            "SDPA_DEBUG nonfinite_dk "
            f"context=output[dk] count={nonfinite} probable_stage={probable_stage} "
            f"ref_probs_finite={int(ref_probs_finite)} ref_dp_finite={int(ref_dp_finite)} "
            f"ref_ds_finite={int(ref_ds_finite)} ref_dk_finite={int(ref_dk_finite)}"
        )


def _require_jvp_hvp_enabled() -> None:
    if os.getenv(_JVP_HVP_ENV, "0") != "1":
        raise RuntimeError(
            f"Triton JVP/HVP is opt-in. Set {_JVP_HVP_ENV}=1 to enable."
        )


def _assert_backward_contract(q, k, v, dout, m, l, require_v: bool = True):
    """Guardrails for Phase 9 backward/JVP.

    Raises loudly instead of silently falling back.
    """
    # Device / dtype
    if not (q.is_cuda and k.is_cuda and dout.is_cuda and (not require_v or (v is not None and v.is_cuda))):
        raise AssertionError("Phase9 backward requires CUDA tensors")
    dtype_check = (
        ("q", q),
        ("k", k),
        ("v", v) if require_v else (),
        ("dout", dout),
    )
    for item in dtype_check:
        if not item:
            continue
        name, t = item
        if t.dtype not in _ALLOWED_DTYPES:
            raise AssertionError(f"Unsupported dtype for {name}: {t.dtype}")
        if not t.is_contiguous():
            raise AssertionError(f"{name} must be contiguous for Phase9 backward")

    # Shapes
    B, H, T, D = q.shape
    if k.ndim != 4 or k.shape[0] != B or k.shape[1] != H or k.shape[3] != D:
        raise AssertionError("k must have shape (B,H,M,D) with D matching q")
    if dout.shape != (B, H, T, D):
        raise AssertionError("dout shape must be (B,H,T,D)")
    if require_v:
        if v is None or v.ndim != 4 or v.shape[0] != B or v.shape[1] != H or v.shape[3] != D:
            raise AssertionError("v must have shape (B,H,M,D) with D matching q")

    # Stats
    if m.shape != (B, H, T) or l.shape != (B, H, T):
        raise AssertionError("m,l must be saved forward stats with shape (B,H,T)")

    # Feature limits
    if D > 64:
        raise AssertionError("Phase9 v0 supports D<=64")

    # No mask/dropout/causal supported in v0; enforced by API (no args)


def _assert_jvp_contract(q, k, v, dq, dk, dv, m, l):
    """Guardrails for Phase 9 JVP (frozen-stats operator)."""
    # Device / dtype
    tensors = (
        ("q", q),
        ("k", k),
        ("v", v),
        ("dq", dq),
        ("dk", dk),
        ("dv", dv),
    )
    for name, t in tensors:
        if not t.is_cuda:
            raise AssertionError("Phase9 JVP requires CUDA tensors")
        if t.dtype not in _ALLOWED_DTYPES:
            raise AssertionError(f"Unsupported dtype for {name}: {t.dtype}")
        if not t.is_contiguous():
            raise AssertionError(f"{name} must be contiguous for Phase9 JVP")

    # Shapes
    B, H, T, D = q.shape
    if k.ndim != 4 or k.shape[0] != B or k.shape[1] != H or k.shape[3] != D:
        raise AssertionError("k must have shape (B,H,M,D) with D matching q")
    if v.ndim != 4 or v.shape[0] != B or v.shape[1] != H or v.shape[3] != D:
        raise AssertionError("v must have shape (B,H,M,D) with D matching q")
    if dq.shape != q.shape or dk.shape != k.shape or dv.shape != v.shape:
        raise AssertionError("dq, dk, dv must match q, k, v shapes respectively")
    if m.shape != (B, H, T) or l.shape != (B, H, T):
        raise AssertionError("m,l must be saved forward stats with shape (B,H,T)")

    # Feature limits
    if D > 64:
        raise AssertionError("Phase9 v0 supports D<=64")

    # Stats sanity
    if torch.any(l <= 0):
        raise AssertionError("Phase9 JVP requires l > 0 (forward sumexp stats)")


@triton.jit
def _load_safe_m_l(
    m_ptrs_base,
    l_ptrs_base,
    row_idx,
    stride_mt,
    stride_lt,
    row_valid_f,
    DENOM_EPS: tl.constexpr,
):
    m_block = tl.load(m_ptrs_base + row_idx * stride_mt).to(tl.float32)
    l_block = tl.load(l_ptrs_base + row_idx * stride_lt).to(tl.float32)
    m_block = m_block * row_valid_f
    l_block = tl.maximum(l_block * row_valid_f + (1.0 - row_valid_f), DENOM_EPS)
    return m_block, l_block


@triton.jit
def _reconstruct_p_from_stats(
    scores,
    m_block,
    l_block,
    mask_mn_f,
):
    shifted = tl.minimum(scores - m_block, 0.0)
    exp_scores = tl.exp(shifted) * mask_mn_f
    p = exp_scores / l_block
    return p * mask_mn_f


@triton.jit
def _sdpa_bwd_dv_kernel(
    Q, K, DO, DV, M, L,
    Tq, Tk, D,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_dobh, stride_dom, stride_dok,
    stride_dvbh, stride_dvn, stride_dvk,
    stride_mbh, stride_mt,
    stride_lbh, stride_lt,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    DENOM_EPS: tl.constexpr,
):
    """
    Each program computes DV for one (bh, key block).
    """
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    # Accumulator for dv block
    acc = tl.zeros((BLOCK_N, BLOCK_DV), dtype=tl.float32)

    # Iterate over query blocks
    q_ptrs_base = Q + pid_bh * stride_qbh
    do_ptrs_base = DO + pid_bh * stride_dobh
    m_ptrs_base = M + pid_bh * stride_mbh
    l_ptrs_base = L + pid_bh * stride_lbh

    for m0 in range(0, Tq, BLOCK_M):
        # Load q_block: (BLOCK_M, D)
        q_ptrs = q_ptrs_base + (m0 + offs_m) * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(
            q_ptrs,
            mask=((m0 + offs_m) < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # Load k_block for this key tile: (BLOCK_N, D)
        k_ptrs = K + pid_bh * stride_kbh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(
            k_ptrs,
            mask=(offs_n[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # scores and P
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale  # (BLOCK_M, BLOCK_N)
        m_idx = m0 + offs_m  # (BM,1)
        row_valid_f = (m_idx < Tq).to(tl.float32)
        m_block, l_block = _load_safe_m_l(
            m_ptrs_base,
            l_ptrs_base,
            m_idx,
            stride_mt,
            stride_lt,
            row_valid_f,
            DENOM_EPS,
        )
        mask_n = (offs_n < Tk).to(tl.float32)[None, :]
        mask_mn_f = row_valid_f * mask_n
        p = _reconstruct_p_from_stats(scores, m_block, l_block, mask_mn_f)

        # Load dO block: (BLOCK_M, Dv) with Dv==D here
        do_ptrs = do_ptrs_base + (m0 + offs_m) * stride_dom + offs_dv[None, :] * stride_dok
        do = tl.load(
            do_ptrs,
            mask=((m0 + offs_m) < Tq) & (offs_dv[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # accumulate: (BLOCK_N, BLOCK_DV) += p^T @ do
        acc += tl.dot(tl.trans(p), do, out_dtype=tl.float32, input_precision="ieee")

    # Store dv block
    dv_ptrs = DV + pid_bh * stride_dvbh + offs_n[:, None] * stride_dvn + offs_dv[None, :] * stride_dvk
    tl.store(dv_ptrs, acc, mask=(offs_n[:, None] < Tk) & (offs_dv[None, :] < D))


def sdpa_bwd_dv(q, k, dout, m, l, scale):
    """
    Compute dV = P^T @ dO using saved (m, l) stats from fused forward.

    Args:
        q, k: (B,H,T/D) and (B,H,M,D)
        dout: (B,H,T,Dv) with Dv == D (v0)
        m, l: (B,H,T) row-wise stats from forward
        scale: 1/sqrt(D)

    Returns:
        dv with shape (B,H,M,Dv), same dtype as inputs.
    """
    _assert_backward_contract(q, k, None, dout, m, l, require_v=False)
    B, H, T, D = q.shape
    _, _, M, Dk = k.shape
    _, _, Tdo, Dv = dout.shape
    assert D == Dk
    assert T == Tdo
    assert D == Dv, "v0 assumes Dv == D"

    dv = torch.empty((B, H, M, Dv), device=q.device, dtype=q.dtype)

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    do_ = dout.reshape(B * H, T, Dv)
    dv_ = dv.reshape(B * H, M, Dv)
    m_ = m.reshape(B * H, T)
    l_ = l.reshape(B * H, T)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_D = 64
    BLOCK_DV = 64
    DENOM_EPS = 1e-12
    grid = (
        B * H,
        triton.cdiv(M, BLOCK_N),
    )

    _sdpa_bwd_dv_kernel[grid](
        q_,
        k_,
        do_,
        dv_,
        m_,
        l_,
        T,
        M,
        D,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        do_.stride(0),
        do_.stride(1),
        do_.stride(2),
        dv_.stride(0),
        dv_.stride(1),
        dv_.stride(2),
        m_.stride(0),
        m_.stride(1),
        l_.stride(0),
        l_.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
        DENOM_EPS=DENOM_EPS,
    )

    _debug_check_output("dv", dv)
    return dv


@triton.jit
def _sdpa_bwd_dk_kernel(
    Q, K, V, DO, DK, M, L,
    Tq, Tk, D,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_vbh, stride_vn, stride_vk,
    stride_dobh, stride_dom, stride_dok,
    stride_dkbh, stride_dkn, stride_dkk,
    stride_mbh, stride_mt,
    stride_lbh, stride_lt,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DENOM_EPS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)  # key block index

    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs_base = Q + pid_bh * stride_qbh
    do_ptrs_base = DO + pid_bh * stride_dobh
    k_ptrs_base = K + pid_bh * stride_kbh
    v_ptrs_base = V + pid_bh * stride_vbh
    m_ptrs_base = M + pid_bh * stride_mbh
    l_ptrs_base = L + pid_bh * stride_lbh

    # We'll accumulate dk for this key block
    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # Loop over query blocks
    for m0 in range(0, Tq, BLOCK_M):
        # load q
        q_ptrs = q_ptrs_base + (m0 + offs_m) * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(
            q_ptrs,
            mask=((m0 + offs_m) < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # pass 1: compute z for these query rows (over all key blocks)
        z_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n0 in range(0, Tk, BLOCK_N):
            k_ptrs = k_ptrs_base + (n0 + tl.arange(0, BLOCK_N))[:, None] * stride_kn + offs_d[None, :] * stride_kk
            v_ptrs = v_ptrs_base + (n0 + tl.arange(0, BLOCK_N))[:, None] * stride_vn + offs_d[None, :] * stride_vk
            k = tl.load(
                k_ptrs,
                mask=((n0 + tl.arange(0, BLOCK_N))[:, None] < Tk) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptrs,
                mask=((n0 + tl.arange(0, BLOCK_N))[:, None] < Tk) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)

            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
            m_idx = m0 + offs_m  # (BM,1)
            row_valid_f = (m_idx < Tq).to(tl.float32)
            m_block, l_block = _load_safe_m_l(
                m_ptrs_base,
                l_ptrs_base,
                m_idx,
                stride_mt,
                stride_lt,
                row_valid_f,
                DENOM_EPS,
            )
            n_mask = ((n0 + tl.arange(0, BLOCK_N)) < Tk).to(tl.float32)[None, :]
            mask_mn = row_valid_f * n_mask
            p = _reconstruct_p_from_stats(scores, m_block, l_block, mask_mn)

            do_ptrs = do_ptrs_base + (m0 + offs_m) * stride_dom + offs_d[None, :] * stride_dok
            do = tl.load(
                do_ptrs,
                mask=((m0 + offs_m) < Tq) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)

            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")
            z_acc += tl.sum(dp * p, axis=1)

        # pass 2: only the target key block pid_n
        k_ptrs = k_ptrs_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = v_ptrs_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(
            k_ptrs,
            mask=(offs_n[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptrs,
            mask=(offs_n[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
        m_idx = m0 + offs_m
        row_valid_f = (m_idx < Tq).to(tl.float32)
        m_block, l_block = _load_safe_m_l(
            m_ptrs_base,
            l_ptrs_base,
            m_idx,
            stride_mt,
            stride_lt,
            row_valid_f,
            DENOM_EPS,
        )
        n_mask = (offs_n < Tk).to(tl.float32)[None, :]
        mask_mn = row_valid_f * n_mask
        p = _reconstruct_p_from_stats(scores, m_block, l_block, mask_mn)

        do_ptrs = do_ptrs_base + (m0 + offs_m) * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(
            do_ptrs,
            mask=((m0 + offs_m) < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")
        ds = p * (dp - z_acc[:, None])
        dk_acc += tl.dot(tl.trans(ds), q, out_dtype=tl.float32, input_precision="ieee") * scale

    dk_ptrs = DK + pid_bh * stride_dkbh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    tl.store(dk_ptrs, dk_acc, mask=(offs_n[:, None] < Tk) & (offs_d[None, :] < D))


def sdpa_bwd_dk(q, k, v, dout, m, l, scale):
    """
    Compute dK for SDPA using saved (m, l). Dv == D assumed.
    """
    _assert_backward_contract(q, k, v, dout, m, l, require_v=True)
    B, H, T, D = q.shape
    M = k.shape[2]
    dk = torch.empty_like(k)

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    v_ = v.reshape(B * H, M, D)
    do_ = dout.reshape(B * H, T, D)
    dk_ = dk.reshape(B * H, M, D)
    m_ = m.reshape(B * H, T)
    l_ = l.reshape(B * H, T)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = 64
    DENOM_EPS = 1e-12
    grid = (
        B * H,
        triton.cdiv(M, BLOCK_N),
    )

    _sdpa_bwd_dk_kernel[grid](
        q_,
        k_,
        v_,
        do_,
        dk_,
        m_,
        l_,
        T,
        M,
        D,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        do_.stride(0),
        do_.stride(1),
        do_.stride(2),
        dk_.stride(0),
        dk_.stride(1),
        dk_.stride(2),
        m_.stride(0),
        m_.stride(1),
        l_.stride(0),
        l_.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        DENOM_EPS=DENOM_EPS,
    )

    _debug_check_dk_output(
        q=q,
        k=k,
        v=v,
        dout=dout,
        m=m,
        l=l,
        scale=scale,
        dk=dk,
    )
    return dk


def sdpa_jvp(q, k, v, dq, dk, dv, m, l, scale):
    """
    Explicit JVP for the **frozen-stats** SDPA operator using saved (m, l).

    Important: this is *not* the JVP of the full recomputed softmax; it
    linearizes the map (q,k,v) -> P(q,k; m,l) @ v with P rebuilt from saved
    stats. Do not compare directly to autograd.functional.jvp(triton_sdpa_fused).

    No autograd inside; CUDA-only in Phase 9. Accumulates in fp32.

    Args:
        q,k,v: (B,H,T,D) / (B,H,M,D)
        dq,dk,dv: same shapes as q,k,v (tangents)
        m,l: (B,H,T) forward row-wise max and sumexp
        scale: 1/sqrt(D)
    Returns:
        dO with shape (B,H,T,D) in q.dtype
    """
    _require_jvp_hvp_enabled()
    _assert_jvp_contract(q, k, v, dq, dk, dv, m, l)
    B, H, T, D = q.shape

    # fp32 compute
    qf = q.float()
    kf = k.float()
    vf = v.float()
    dqf = dq.float()
    dkf = dk.float()
    dvf = dv.float()

    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale  # (B,H,T,M)
    # reconstruct P from saved stats
    P = torch.exp(scores - m.unsqueeze(-1)) / l.unsqueeze(-1)

    dS = (torch.matmul(dqf, kf.transpose(-2, -1)) + torch.matmul(qf, dkf.transpose(-2, -1))) * scale
    dS_centered = dS - (dS * P).sum(dim=-1, keepdim=True)
    dP = P * dS_centered

    dO = torch.matmul(dP, vf) + torch.matmul(P, dvf)
    return dO.to(q.dtype)


def sdpa_hvp(q, k, v, vq, vk, vv, m, l, scale):
    """
    Explicit HVP for the frozen-stats SDPA operator using saved (m, l).

    This matches the same contract as sdpa_jvp: P is reconstructed from saved
    stats and the softmax Jacobian/Hessian are evaluated at that base point.
    It is intended for small-shape correctness checks (CPU oracle), not speed.

    Args:
        q,k,v: (B,H,T,D) / (B,H,M,D)
        vq,vk,vv: direction vectors (same shapes as q,k,v)
        m,l: (B,H,T) forward row-wise max and sumexp
        scale: 1/sqrt(D)
    Returns:
        (hvp_q, hvp_k, hvp_v) with the same shapes as q/k/v.
    """
    _require_jvp_hvp_enabled()
    # Allow CPU for oracle tests; keep compute in fp32.
    qf = q.float()
    kf = k.float()
    vf = v.float()
    vqf = vq.float()
    vkf = vk.float()
    vvf = vv.float()

    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    P = torch.exp(scores - m.unsqueeze(-1)) / l.unsqueeze(-1)

    def softmax_jvp(delta_scores: torch.Tensor) -> torch.Tensor:
        inner = (delta_scores * P).sum(dim=-1, keepdim=True)
        return P * (delta_scores - inner)

    scores_dir = (
        torch.matmul(vqf, kf.transpose(-2, -1))
        + torch.matmul(qf, vkf.transpose(-2, -1))
    ) * scale
    probs_dir = softmax_jvp(scores_dir)

    grad_probs = vf.sum(dim=-1).unsqueeze(-2).expand_as(P)
    grad_probs_dir = vvf.sum(dim=-1).unsqueeze(-2).expand_as(P)

    inner = (grad_probs * P).sum(dim=-1, keepdim=True)
    inner_dir = (grad_probs_dir * P + grad_probs * probs_dir).sum(dim=-1, keepdim=True)

    grad_scores = P * (grad_probs - inner)
    grad_scores_dir = probs_dir * (grad_probs - inner) + P * (grad_probs_dir - inner_dir)

    hvp_q = scale * (
        torch.matmul(grad_scores_dir, kf) + torch.matmul(grad_scores, vkf)
    )
    hvp_k = scale * (
        torch.matmul(grad_scores_dir.transpose(-2, -1), qf)
        + torch.matmul(grad_scores.transpose(-2, -1), vqf)
    )
    hvp_v = probs_dir.sum(dim=-2).unsqueeze(-1).expand_as(vf)

    return hvp_q.to(q.dtype), hvp_k.to(k.dtype), hvp_v.to(v.dtype)

@triton.jit
def _sdpa_bwd_dq_kernel(
    Q, K, V, DO, DQ, M, L,
    Tq, Tk, D,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_vbh, stride_vn, stride_vk,
    stride_dobh, stride_dom, stride_dok,
    stride_dqbh, stride_dqm, stride_dqk,
    stride_mbh, stride_mt,
    stride_lbh, stride_lt,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DENOM_EPS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs_base = Q + pid_bh * stride_qbh
    do_ptrs_base = DO + pid_bh * stride_dobh
    k_ptrs_base = K + pid_bh * stride_kbh
    v_ptrs_base = V + pid_bh * stride_vbh
    m_ptrs_base = M + pid_bh * stride_mbh
    l_ptrs_base = L + pid_bh * stride_lbh

    # Load q_block once (fp32)
    q_ptrs = q_ptrs_base + offs_m * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(
        q_ptrs,
        mask=(offs_m < Tq) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    # Pass 1: accumulate z
    z_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n0 in range(0, Tk, BLOCK_N):
        k_ptrs = k_ptrs_base + (n0 + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = v_ptrs_base + (n0 + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk

        k = tl.load(
            k_ptrs,
            mask=((n0 + offs_n)[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptrs,
            mask=((n0 + offs_n)[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale  # (Mblock, Nblock)
        m_idx = offs_m
        row_valid_f = (m_idx < Tq).to(tl.float32)
        m_block, l_block = _load_safe_m_l(
            m_ptrs_base,
            l_ptrs_base,
            m_idx,
            stride_mt,
            stride_lt,
            row_valid_f,
            DENOM_EPS,
        )
        mask_n = ((n0 + offs_n) < Tk).to(tl.float32)[None, :]
        mask_mn_f = row_valid_f * mask_n
        p = _reconstruct_p_from_stats(scores, m_block, l_block, mask_mn_f)

        do_ptrs = do_ptrs_base + offs_m * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(
            do_ptrs,
            mask=(offs_m < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # dp = dO @ V^T for this key block (reuse v)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")  # (Mblock, Nblock)
        z_acc += tl.sum(dp * p, axis=1)

    # Pass 2: accumulate dQ
    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for n0 in range(0, Tk, BLOCK_N):
        k_ptrs = k_ptrs_base + (n0 + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = v_ptrs_base + (n0 + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk

        k = tl.load(
            k_ptrs,
            mask=((n0 + offs_n)[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptrs,
            mask=((n0 + offs_n)[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
        m_idx = offs_m
        row_valid_f = (m_idx < Tq).to(tl.float32)
        m_block, l_block = _load_safe_m_l(
            m_ptrs_base,
            l_ptrs_base,
            m_idx,
            stride_mt,
            stride_lt,
            row_valid_f,
            DENOM_EPS,
        )
        mask_n = ((n0 + offs_n) < Tk).to(tl.float32)[None, :]
        mask_mn_f = row_valid_f * mask_n
        p = _reconstruct_p_from_stats(scores, m_block, l_block, mask_mn_f)

        do_ptrs = do_ptrs_base + offs_m * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(
            do_ptrs,
            mask=(offs_m < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")  # (Mblock, Nblock)
        ds = p * (dp - z_acc[:, None])
        dq_acc += tl.dot(ds, k, out_dtype=tl.float32, input_precision="ieee") * scale

    dq_ptrs = DQ + pid_bh * stride_dqbh + offs_m * stride_dqm + offs_d[None, :] * stride_dqk
    tl.store(dq_ptrs, dq_acc, mask=(offs_m < Tq) & (offs_d[None, :] < D))


def sdpa_bwd_dq(q, k, v, dout, m, l, scale):
    """
    Compute dQ for SDPA using saved (m, l). Dv == D assumed.
    """
    _assert_backward_contract(q, k, v, dout, m, l, require_v=True)
    B, H, T, D = q.shape
    M = k.shape[2]
    dq = torch.empty_like(q)

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    v_ = v.reshape(B * H, M, D)
    do_ = dout.reshape(B * H, T, D)
    dq_ = dq.reshape(B * H, T, D)
    m_ = m.reshape(B * H, T)
    l_ = l.reshape(B * H, T)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = 64
    DENOM_EPS = 1e-12
    grid = (
        B * H,
        triton.cdiv(T, BLOCK_M),
    )

    _sdpa_bwd_dq_kernel[grid](
        q_,
        k_,
        v_,
        do_,
        dq_,
        m_,
        l_,
        T,
        M,
        D,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        do_.stride(0),
        do_.stride(1),
        do_.stride(2),
        dq_.stride(0),
        dq_.stride(1),
        dq_.stride(2),
        m_.stride(0),
        m_.stride(1),
        l_.stride(0),
        l_.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        DENOM_EPS=DENOM_EPS,
    )

    _debug_check_output("dq", dq)
    return dq


__all__ = [
    "sdpa_bwd_dq",
    "sdpa_bwd_dk",
    "sdpa_bwd_dv",
    "sdpa_jvp",
    "sdpa_hvp",
    "debug_audit_fast_path",
    "get_triton_sdpa_debug_counters",
    "get_triton_sdpa_debug_artifact",
    "reset_triton_sdpa_debug_counters",
    "reset_triton_sdpa_debug_artifact",
]
