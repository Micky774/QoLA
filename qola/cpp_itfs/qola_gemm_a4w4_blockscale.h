// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// QoLA cpp_itfs wrapper for AITER's gemm_a4w4_blockscale kernel.
// Torch-free: takes raw device pointers and a HIP stream.
#pragma once

#include "qola_common.h"

QOLA_NS_BEGIN

struct gemm_a4w4_blockscale_args {
    const void* a_ptr;       // [M, K/2] fp4x2
    const void* b_ptr;       // [N, K/2] fp4x2
    const void* a_scale_ptr; // [M, K/32] e8m0 (int32)
    const void* b_scale_ptr; // [N, K/32] e8m0 (int32)
    void* out_ptr;           // [M, N] fp16 or bf16
    int M;
    int N;
    int K;                   // full K dimension (not K/2)
    int stride_a;            // row stride of A in fp4x2 elements (= K/2 for contiguous)
    int stride_b;            // row stride of B in fp4x2 elements
    int stride_out;          // row stride of output in elements (not bytes)
    int stride_a_scale;      // row stride of A_scale
    int stride_b_scale;      // row stride of B_scale
    int splitK;              // log2 of split factor (0 = no split)
    int output_dtype;        // 0 = fp16, 1 = bf16
};

// Launch the A4W4 blockscale GEMM kernel via CK templates.
//
// Returns 0 on success, -1 if no suitable kernel is found or
// the problem configuration is not supported.
//
// The caller owns all device memory and the HIP stream.
__attribute__((visibility("default")))
int gemm_a4w4_blockscale(const gemm_a4w4_blockscale_args& args, hipStream_t stream);

QOLA_NS_END
