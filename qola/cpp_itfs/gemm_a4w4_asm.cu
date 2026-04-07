// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free cpp_itfs wrapper for AITER's A4W4 ASM GEMM kernel.
// Mirrors the logic of csrc/py_itfs_cu/asm_gemm_a4w4.cu but operates on
// raw device pointers instead of torch::Tensor.

#include "gemm_a4w4_asm.h"

#include "aiter_hip_common.h"
#include "asm_f4gemm_configs.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// ---- KernelArgs layout (must match AITER's asm_gemm_a4w4.cu) ----
struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;
    p2 _p0;
    void* ptr_C;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    void* ptr_B;
    p2 _p3;
    float alpha;
    p3 _p4;
    float beta;
    p3 _p5;
    unsigned int stride_D0;
    p3 _p6;
    unsigned int stride_D1;
    p3 _p7;
    unsigned int stride_C0;
    p3 _p8;
    unsigned int stride_C1;
    p3 _p9;
    unsigned int stride_A0;
    p3 _p10;
    unsigned int stride_A1;
    p3 _p11;
    unsigned int stride_B0;
    p3 _p12;
    unsigned int stride_B1;
    p3 _p13;
    unsigned int M;
    p3 _p14;
    unsigned int N;
    p3 _p15;
    unsigned int K;
    p3 _p16;
    void* ptr_ScaleA;
    p2 _p17;
    void* ptr_ScaleB;
    p2 _p18;
    unsigned int stride_ScaleA0;
    p3 _p19;
    unsigned int stride_ScaleA1;
    p3 _p20;
    unsigned int stride_ScaleB0;
    p3 _p21;
    unsigned int stride_ScaleB1;
    p3 _p22;
    int log2_k_split;
};

// ---- Heuristic kernel selection (adapted from asm_gemm_a4w4.cu) ----

static std::tuple<std::string, int> get_heuristic_kernel(
    int M, int N, int K,
    const std::string& arch_id,
    int log2_k_split_hint,
    bool bpreshuffle,
    CFG* cfgs)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu        = dev_prop.multiProcessorCount;
    uint32_t empty_cu      = num_cu;
    uint32_t tg_num        = 0;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0;
    int log2_k_split_en    = (log2_k_split_hint > 0) ? 1 : 0;
    int bpreshuffle_en     = bpreshuffle ? 1 : 0;
    std::string selectedKernelName;
    int selectedsplitK = 1;

    for (const auto& el : *cfgs) {
        if (el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if (cfg.bpreshuffle == bpreshuffle_en && (cfg.splitK >= log2_k_split_en)) {
            if (cfg.tile_M != 128 || cfg.tile_N != 512 || (N % cfg.tile_N) == 0) {
                std::vector<int> splitK_list =
                    (log2_k_split_hint > 0 && cfg.splitK)
                        ? std::vector<int>{1 << log2_k_split_hint}
                        : (cfg.splitK ? std::vector<int>{2, 4, 8, 16} : std::vector<int>{1});

                for (auto& splitK : splitK_list) {
                    int tg_num_M         = (M + cfg.tile_M - 1) / cfg.tile_M;
                    int tg_num_N         = (N + cfg.tile_N - 1) / cfg.tile_N;
                    tg_num               = tg_num_M * tg_num_N * splitK;
                    uint32_t local_round = (tg_num + num_cu - 1) / num_cu;

                    float local_compute2mem_effi =
                        cfg.tile_M * cfg.tile_N / (cfg.tile_M + cfg.tile_N);

                    bool is_earlier_round        = (local_round < round);
                    bool is_same_round           = (local_round == round);
                    bool has_sufficient_empty_cu = (empty_cu > (local_round * num_cu - tg_num));
                    bool has_better_efficiency   = (local_compute2mem_effi > compute2mem_effi);
                    if (is_earlier_round ||
                        (is_same_round && (has_sufficient_empty_cu || has_better_efficiency))) {
                        round              = local_round;
                        empty_cu           = local_round * num_cu - tg_num;
                        compute2mem_effi   = local_compute2mem_effi;
                        selectedKernelName = el.first;
                        selectedsplitK     = splitK;
                    }
                }
            }
        }
    }

    if (selectedKernelName.empty())
        return {"", -1};

    int log2_result = 0;
    int tmp = selectedsplitK;
    while (tmp >>= 1)
        ++log2_result;
    return {selectedKernelName, log2_result};
}

// ---- Entry point ----

QOLA_NS_BEGIN

int gemm_a4w4_asm(const gemm_a4w4_asm_args& a, hipStream_t stream)
{
    CFG* config_map = &cfg_f4gemm_bf16_per1x32Fp4;
    if (config_map->empty())
        return -1;

    const int Mdim = a.M;
    const int Ndim = a.N;
    const int Kdim = a.K;

    // Populate kernel args from the raw-pointer args struct.
    KernelArgs kargs{};
    kargs.ptr_D      = a.out_ptr;
    kargs.ptr_C      = const_cast<void*>(a.bias_ptr); // nullptr when no bias
    kargs.ptr_A      = const_cast<void*>(a.a_ptr);
    kargs.ptr_B      = const_cast<void*>(a.b_ptr);
    kargs.alpha       = a.alpha;
    kargs.beta        = a.beta;
    kargs.stride_C0   = a.stride_out;
    kargs.stride_A0   = a.stride_a * 2;  // fp4_x2 → element stride
    kargs.stride_B0   = a.stride_b * 2;
    kargs.M           = Mdim;
    kargs.N           = Ndim;
    kargs.K           = Kdim;
    kargs.ptr_ScaleA  = const_cast<void*>(a.a_scale_ptr);
    kargs.ptr_ScaleB  = const_cast<void*>(a.b_scale_ptr);
    kargs.stride_ScaleA0 = a.stride_a_scale;
    kargs.stride_ScaleB0 = a.stride_b_scale;
    kargs.log2_k_split   = 0;

    // Heuristic kernel selection (cached per M,N,K).
    using DictKey = std::tuple<int, int, int>;
    struct SimpleHash {
        size_t operator()(const DictKey& key) const {
            return std::hash<int>()(std::get<0>(key)) ^
                   std::hash<int>()(std::get<1>(key)) ^
                   std::hash<int>()(std::get<2>(key));
        }
    };
    static std::unordered_map<DictKey, std::tuple<std::string, int>, SimpleHash>
        heuristic_cache;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_cache;

    std::string arch_id = get_gpu_arch();
    std::string kernelName;
    int selectedksplit = 0;

    auto cache_it = heuristic_cache.find({Mdim, Ndim, Kdim});
    if (cache_it != heuristic_cache.end()) {
        kernelName     = std::get<0>(cache_it->second);
        selectedksplit = std::get<1>(cache_it->second);
    } else {
        auto [name, split] = get_heuristic_kernel(
            Mdim, Ndim, Kdim, arch_id,
            /*log2_k_split_hint=*/0, /*bpreshuffle=*/true, config_map);
        if (name.empty())
            return -1;
        kernelName     = name;
        selectedksplit = split;
        heuristic_cache[{Mdim, Ndim, Kdim}] = {kernelName, selectedksplit};
    }

    // Find the config and prepare the ASM kernel.
    auto it = config_map->find(kernelName);
    if (it == config_map->end())
        return -1;

    const auto& cfg = it->second;
    int SUBM = cfg.tile_M;
    int SUBN = cfg.tile_N;
    int gdz  = 1;

    if (cfg.splitK == 1) {
        kargs.log2_k_split = selectedksplit;
        int k_num = 1 << kargs.log2_k_split;
        if (Kdim % k_num != 0)
            return -1;
        if (k_num > 1) {
            HIP_CALL(hipMemsetAsync(a.out_ptr, 0,
                                    static_cast<size_t>(Mdim) * Ndim * sizeof(uint16_t),
                                    stream));
        }
        int k_per_tg = Kdim / k_num;
        k_per_tg     = ((k_per_tg + 256 - 1) / 256) * 256;
        gdz          = (Kdim + k_per_tg - 1) / k_per_tg;
    }

    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();

    auto result = impl_cache.emplace(name, nullptr);
    if (result.second)
        result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
    AiterAsmKernel* impl_ptr = result.first->second.get();

    int gdx = (Ndim + SUBN - 1) / SUBN;
    int gdy = (Mdim + SUBM - 1) / SUBM;
    size_t arg_size = sizeof(kargs);

    impl_ptr->launch_kernel({&kargs, &arg_size,
                             gdx, gdy, gdz,
                             256, 1, 1,
                             stream});
    return 0;
}

QOLA_NS_END

