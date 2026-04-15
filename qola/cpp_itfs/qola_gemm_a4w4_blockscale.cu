// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin cpp_itfs entry point for AITER's A4W4 blockscale CK GEMM.
// Constructs stub Tensors from the raw-pointer args struct and calls
// AITER's gemm_a4w4_blockscale() directly — no dispatch duplication.

#include "qola_gemm_a4w4_blockscale.h"
#include <torch/all.h>            // stub — provides torch::Tensor, ScalarType, etc.
#include <ATen/hip/HIPContext.h>  // stub — provides qola_set_stream()

// Forward-declare AITER's entry point (defined in AITER's gemm_a4w4_blockscale.cu,
// compiled as part of the same module via the source list).
torch::Tensor gemm_a4w4_blockscale(
    torch::Tensor& XQ, torch::Tensor& WQ,
    torch::Tensor& x_scale, torch::Tensor& w_scale,
    torch::Tensor& Y, int splitK);

QOLA_NS_BEGIN

int gemm_a4w4_blockscale(const gemm_a4w4_blockscale_args& a, hipStream_t stream)
{
    at::hip::qola_set_stream(stream);

    const int K_packed = a.K / 2;
    at::ScalarType out_dt = (a.output_dtype == 0)
        ? at::ScalarType::Half : at::ScalarType::BFloat16;

    torch::Tensor XQ(const_cast<void*>(a.a_ptr),
                     {a.M, K_packed}, {a.stride_a, 1}, at::ScalarType::Byte);
    torch::Tensor WQ(const_cast<void*>(a.b_ptr),
                     {a.N, K_packed}, {a.stride_b, 1}, at::ScalarType::Byte);
    torch::Tensor x_scale(const_cast<void*>(a.a_scale_ptr),
                          {a.M, a.K / 32}, {a.stride_a_scale, 1}, at::ScalarType::Int);
    torch::Tensor w_scale(const_cast<void*>(a.b_scale_ptr),
                          {a.N, a.K / 32}, {a.stride_b_scale, 1}, at::ScalarType::Int);
    torch::Tensor Y(a.out_ptr,
                    {a.M, a.N}, {a.stride_out, 1}, out_dt);

    try {
        ::gemm_a4w4_blockscale(XQ, WQ, x_scale, w_scale, Y, a.splitK);
    } catch (...) {
        return -1;
    }
    return 0;
}

QOLA_NS_END
