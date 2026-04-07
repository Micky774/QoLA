// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free cpp_itfs wrapper for AITER's A4W4 blockscale CK GEMM kernel.
// Mirrors the dispatch logic of csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu
// but operates on raw device pointers instead of torch::Tensor.

#include "gemm_a4w4_blockscale.h"

#include "qola_gemm_a4w4_blockscale_common.cuh"
#include "gemm_a4w4_blockscale_manifest.h"
#include "gemm_a4w4_blockscale_lookup.h"

#include <cmath>
#include <climits>
#include <functional>
#include <tuple>
#include <unordered_map>

// ---- getPaddedM (inlined from gemm_common.cu to avoid torch dep) ----

static constexpr int nextPow2(unsigned int num)
{
    if (num <= 1)
        return 1;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

static int getPaddedM(int M, int N, int K, int gl)
{
    int padded_m = M;
    if (gl == 0) {
        if (M <= 256)
            padded_m = (M + 15) / 16 * 16;
        else if (M <= 1024)
            padded_m = (M + 31) / 32 * 32;
        else if (M <= 4096)
            padded_m = (M + 63) / 64 * 64;
        else
            padded_m = (M + 127) / 128 * 128;
    } else if (gl == 1) {
        if (M > 8192 && N > 4096)
            padded_m = 8192;
        else
            padded_m = nextPow2(M);
    }
    return padded_m;
}

// ---- Dispatch types ----

// Raw-pointer kernel function type (matches generated instance signatures).
using BlockwiseKernel = std::function<
    int(const void*, const void*,
        const void*, const void*,
        void*,
        int, int, int,
        int, int, int,
        int, int,
        int, hipStream_t)>;

struct IntTupleHash
{
    size_t operator()(const std::tuple<int, int, int>& t) const
    {
        auto hash1 = std::hash<int>{}(std::get<0>(t));
        auto hash2 = std::hash<int>{}(std::get<1>(t));
        auto hash3 = std::hash<int>{}(std::get<2>(t));
        return hash1 ^ hash2 ^ hash3;
    }
};

using BlockwiseKernelMap = std::unordered_map<
    std::tuple<int, int, int>,
    BlockwiseKernel,
    IntTupleHash>;

// ---- Kernel dispatch ----

template <typename CDataType>
BlockwiseKernel blockscale_dispatch(int M, int N, int K)
{
    static const auto lookup = []
    {
        if constexpr (std::is_same_v<CDataType, F16>) {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(F16)};
        } else if constexpr (std::is_same_v<CDataType, B16>) {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(B16)};
        } else {
            static_assert(false, "blockscale_dispatch used with unsupported dtype!");
        }
    }();

    // Exact match.
    auto it = lookup.find({M, N, K});
    if (it != lookup.end())
        return it->second;

    // Fine-grained padded M.
    int padded_m = getPaddedM(M, N, K, 0);
    it = lookup.find({padded_m, N, K});
    if (it != lookup.end())
        return it->second;

    // Coarse-grained padded M.
    padded_m = getPaddedM(M, N, K, 1);
    it = lookup.find({padded_m, N, K});
    if (it != lookup.end())
        return it->second;

    // Fallback to default kernel.
    return a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CDataType>;
}

// ---- Entry point ----

QOLA_NS_BEGIN

int gemm_a4w4_blockscale(const gemm_a4w4_blockscale_args& a, hipStream_t stream)
{
    const int M = a.M;
    const int N = a.N;
    const int K = a.K;

    BlockwiseKernel kernel;
    if (a.output_dtype == 0) {
        kernel = blockscale_dispatch<F16>(M, N, K);
    } else if (a.output_dtype == 1) {
        kernel = blockscale_dispatch<B16>(M, N, K);
    } else {
        return -1;
    }

    return kernel(
        a.a_ptr, a.a_scale_ptr,
        a.b_ptr, a.b_scale_ptr,
        a.out_ptr,
        M, N, K,
        a.stride_a, a.stride_b, a.stride_out,
        a.stride_a_scale, a.stride_b_scale,
        a.splitK, stream);
}

QOLA_NS_END
