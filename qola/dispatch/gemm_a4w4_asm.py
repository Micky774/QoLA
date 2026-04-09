# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Python dispatch for the QoLA cpp_itfs gemm_a4w4_asm kernel.

Loads the torch-free .so via ctypes.CDLL (RTLD_GLOBAL) and resolves the
C++ symbol by its Itanium ABI mangled name — same pattern as jax-aiter's
ffi/registry.py.
"""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path
from typing import Optional

import torch

# ---- C struct mirroring qola::gemm_a4w4_asm_args ----


class _GemmA4W4AsmArgs(ctypes.Structure):
    _fields_ = [
        ("a_ptr", ctypes.c_void_p),
        ("b_ptr", ctypes.c_void_p),
        ("a_scale_ptr", ctypes.c_void_p),
        ("b_scale_ptr", ctypes.c_void_p),
        ("out_ptr", ctypes.c_void_p),
        ("bias_ptr", ctypes.c_void_p),
        ("M", ctypes.c_int),
        ("N", ctypes.c_int),
        ("K", ctypes.c_int),
        ("stride_a", ctypes.c_int),
        ("stride_b", ctypes.c_int),
        ("stride_out", ctypes.c_int),
        ("stride_a_scale", ctypes.c_int),
        ("stride_b_scale", ctypes.c_int),
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
    ]


# Itanium ABI mangled name for:
#   int qola::gemm_a4w4_asm(const qola::gemm_a4w4_asm_args&, ihipStream_t*)
_MANGLED_SYM = "_ZN4qola13gemm_a4w4_asmERKNS_18gemm_a4w4_asm_argsEP12ihipStream_t"


def _find_lib(name: str) -> str:
    """Locate a QoLA-built .so by searching known paths."""
    candidates = [
        # QOLA_LIB_DIR env override (highest priority)
        Path(os.environ.get("QOLA_LIB_DIR", "")) / name,
        # Relative to QoLA package: artifacts/lib/
        Path(__file__).resolve().parent.parent.parent / "artifacts" / "lib" / name,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        f"Could not find {name}. Set QOLA_LIB_DIR to the directory "
        f"containing the QoLA-built .so files."
    )


@functools.lru_cache(maxsize=1)
def _load_lib():
    """Load the cpp_itfs .so and resolve the C++ symbol by mangled name."""
    so_path = _find_lib("module_gemm_a4w4_asm.so")
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

    fn = getattr(lib, _MANGLED_SYM)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(_GemmA4W4AsmArgs), ctypes.c_void_p]

    return fn


def gemm_a4w4_asm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """Dispatch A4W4 ASM GEMM via QoLA's torch-free cpp_itfs library.

    Args:
        A: [M, K/2] packed FP4 pairs (uint8 or float4_e2m1fn_x2)
        B: [N, K/2] packed FP4 pairs
        A_scale: [M, K/32] E8M0 block scales
        B_scale: [N, K/32] E8M0 block scales
        out: [M_padded, N] bf16 output buffer (M_padded must be multiple of 32)
        bias: optional bias tensor, or None
        alpha, beta: scaling factors

    Returns:
        out tensor.
    """
    fn = _load_lib()

    m = A.shape[0]
    k = A.shape[-1] * 2  # packed fp4

    args = _GemmA4W4AsmArgs(
        a_ptr=A.data_ptr(),
        b_ptr=B.data_ptr(),
        a_scale_ptr=A_scale.data_ptr(),
        b_scale_ptr=B_scale.data_ptr(),
        out_ptr=out.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else 0,
        M=m,
        N=B.shape[0],
        K=k,
        stride_a=A.stride(0),
        stride_b=B.stride(0),
        stride_out=out.stride(0) * 2,  # bf16 = 2 bytes
        stride_a_scale=A_scale.stride(0),
        stride_b_scale=B_scale.stride(0),
        alpha=alpha,
        beta=beta,
    )

    stream = torch.cuda.current_stream().cuda_stream

    ret = fn(ctypes.byref(args), ctypes.c_void_p(stream))
    if ret != 0:
        raise RuntimeError(
            f"qola::gemm_a4w4_asm failed (ret={ret}). "
            "No suitable kernel found for the given GPU arch / problem shape."
        )

    return out
