// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// QoLA cpp_itfs wrapper for AITER's gemm_a4w4_asm kernel.
// Torch-free: takes raw device pointers and a HIP stream.
#pragma once

#include "qola_common.h"

QOLA_NS_BEGIN

struct gemm_a4w4_asm_args {
    const void* a_ptr;       // [M, K/2] fp4x2
    const void* b_ptr;       // [N, K/2] fp4x2
    const void* a_scale_ptr; // [M, K/32] e8m0
    const void* b_scale_ptr; // [N, K/32] e8m0
    void* out_ptr;           // [M, N] bf16
    const void* bias_ptr;    // [M, N] f32, or nullptr
    int M;
    int N;
    int K;                   // full K dimension (not K/2)
    int stride_a;            // row stride of A in fp4 elements (= K/2 for contiguous)
    int stride_b;            // row stride of B in fp4 elements
    int stride_out;          // row stride of out in bytes (includes sizeof(bf16))
    int stride_a_scale;      // row stride of A_scale
    int stride_b_scale;      // row stride of B_scale
    float alpha;             // default 1.0
    float beta;              // default 0.0
};

// Launch the A4W4 ASM GEMM kernel: D = alpha * A * B + beta * C.
//
// Returns 0 on success, -1 if no suitable kernel is found for the current
// GPU architecture or problem dimensions.
//
// The caller owns all device memory and the HIP stream.  For split-K
// configurations the output buffer is zeroed internally via hipMemsetAsync
// on the provided stream before accumulation.
__attribute__((visibility("default")))
int gemm_a4w4_asm(const gemm_a4w4_asm_args& args, hipStream_t stream);

QOLA_NS_END
