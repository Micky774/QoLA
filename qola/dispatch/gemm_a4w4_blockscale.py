# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Python dispatch for the QoLA cpp_itfs gemm_a4w4_blockscale kernel.

Loads the torch-free .so via ctypes.CDLL (RTLD_GLOBAL) and resolves the
C++ symbol by its Itanium ABI mangled name — same pattern as jax-aiter's
ffi/registry.py.
"""

from __future__ import annotations

import ctypes
import functools

import torch

from . import find_lib, mangled_sym

# ---- C struct mirroring qola::gemm_a4w4_blockscale_args ----


class _GemmA4W4BlockscaleArgs(ctypes.Structure):
    _fields_ = [
        ("a_ptr", ctypes.c_void_p),
        ("b_ptr", ctypes.c_void_p),
        ("a_scale_ptr", ctypes.c_void_p),
        ("b_scale_ptr", ctypes.c_void_p),
        ("out_ptr", ctypes.c_void_p),
        ("M", ctypes.c_int),
        ("N", ctypes.c_int),
        ("K", ctypes.c_int),
        ("stride_a", ctypes.c_int),
        ("stride_b", ctypes.c_int),
        ("stride_out", ctypes.c_int),
        ("stride_a_scale", ctypes.c_int),
        ("stride_b_scale", ctypes.c_int),
        ("splitK", ctypes.c_int),
        ("output_dtype", ctypes.c_int),
    ]


@functools.lru_cache(maxsize=1)
def _load_lib():
    """Load the cpp_itfs .so and resolve the C++ symbol by mangled name."""
    so_path = find_lib("module_gemm_a4w4_blockscale.so")
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

    sym = mangled_sym("gemm_a4w4_blockscale", "gemm_a4w4_blockscale_args")
    fn = getattr(lib, sym)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(_GemmA4W4BlockscaleArgs), ctypes.c_void_p]

    return fn


_DTYPE_MAP = {
    torch.float16: 0,
    torch.half: 0,
    torch.bfloat16: 1,
}


def gemm_a4w4_blockscale(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out: torch.Tensor,
    splitK: int = 0,
) -> torch.Tensor:
    """Dispatch A4W4 blockscale CK GEMM via QoLA's torch-free cpp_itfs library.

    Args:
        A: [M, K/2] packed FP4 pairs (uint8 or float4_e2m1fn_x2)
        B: [N, K/2] packed FP4 pairs
        A_scale: [M, K/32] E8M0 block scales (int32)
        B_scale: [N, K/32] E8M0 block scales (int32)
        out: [M, N] fp16 or bf16 output buffer (pre-allocated)
        splitK: log2 of split factor (0 = no split)

    Returns:
        out tensor.
    """
    fn = _load_lib()

    output_dtype = _DTYPE_MAP.get(out.dtype)
    if output_dtype is None:
        raise ValueError(f"Unsupported output dtype: {out.dtype}. Use fp16 or bf16.")

    m = A.shape[0]
    k = A.shape[-1] * 2  # packed fp4

    args = _GemmA4W4BlockscaleArgs(
        a_ptr=A.data_ptr(),
        b_ptr=B.data_ptr(),
        a_scale_ptr=A_scale.data_ptr(),
        b_scale_ptr=B_scale.data_ptr(),
        out_ptr=out.data_ptr(),
        M=m,
        N=B.shape[0],
        K=k,
        stride_a=A.stride(0),
        stride_b=B.stride(0),
        stride_out=out.stride(0),
        stride_a_scale=A_scale.stride(0),
        stride_b_scale=B_scale.stride(0),
        splitK=splitK,
        output_dtype=output_dtype,
    )

    stream = torch.cuda.current_stream().cuda_stream

    ret = fn(ctypes.byref(args), ctypes.c_void_p(stream))
    if ret != 0:
        raise RuntimeError(
            f"qola::gemm_a4w4_blockscale failed (ret={ret}). "
            "No suitable kernel found for the given GPU arch / problem shape."
        )

    return out
