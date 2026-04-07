// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin cpp_itfs entry point for AITER's A4W4 ASM GEMM.
// Constructs stub Tensors from the raw-pointer args struct and calls
// AITER's gemm_a4w4_asm() directly — no dispatch duplication.

#include "qola_gemm_a4w4_asm.h"
#include <torch/all.h>            // stub — provides torch::Tensor, ScalarType, etc.
#include <ATen/hip/HIPContext.h>  // stub — provides qola_set_stream()

// Forward-declare AITER's entry point (defined in AITER's asm_gemm_a4w4.cu,
// compiled as part of the same module via the source list).
torch::Tensor gemm_a4w4_asm(
    torch::Tensor& A, torch::Tensor& B,
    torch::Tensor& A_scale, torch::Tensor& B_scale,
    torch::Tensor& out, std::string& kernelName,
    std::optional<torch::Tensor>& bias,
    std::optional<float> alpha, std::optional<float> beta,
    std::optional<bool> bpreshuffle,
    std::optional<int> log2_k_split);

QOLA_NS_BEGIN

int gemm_a4w4_asm(const gemm_a4w4_asm_args& a, hipStream_t stream)
{
    at::hip::qola_set_stream(stream);

    const int K_packed = a.K / 2;

    torch::Tensor A(const_cast<void*>(a.a_ptr),
                    {a.M, K_packed}, {a.stride_a, 1}, at::ScalarType::Byte);
    torch::Tensor B(const_cast<void*>(a.b_ptr),
                    {a.N, K_packed}, {a.stride_b, 1}, at::ScalarType::Byte);
    torch::Tensor A_scale(const_cast<void*>(a.a_scale_ptr),
                          {a.M, a.K / 32}, {a.stride_a_scale, 1}, at::ScalarType::Byte);
    torch::Tensor B_scale(const_cast<void*>(a.b_scale_ptr),
                          {a.N, a.K / 32}, {a.stride_b_scale, 1}, at::ScalarType::Byte);
    torch::Tensor out(a.out_ptr,
                      {a.M, a.N}, {a.stride_out, 1}, at::ScalarType::BFloat16);

    std::string kernelName;  // empty = auto-select via heuristic
    std::optional<torch::Tensor> bias;
    if (a.bias_ptr) {
        bias.emplace(const_cast<void*>(a.bias_ptr),
                     std::initializer_list<int64_t>{a.M, a.N},
                     std::initializer_list<int64_t>{a.N, 1},
                     at::ScalarType::Float);
    }

    try {
        ::gemm_a4w4_asm(A, B, A_scale, B_scale, out, kernelName,
                        bias, a.alpha, a.beta,
                        /*bpreshuffle=*/true, /*log2_k_split=*/std::nullopt);
    } catch (...) {
        return -1;
    }
    return 0;
}

QOLA_NS_END
