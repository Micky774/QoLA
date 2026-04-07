#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free version of AITER's gemm_a4w4_blockscale_common.cuh.
// Replaces torch::Tensor parameters with raw device pointers + dimensions.
#ifdef USE_ROCM

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <cmath>

#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using MFMA = ck::tensor_layout::gemm::MFMA;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using F4PK = ck::f4x2_pk_t;
using F16 = ck::half_t;
using B16 = ck::bhalf_t;
using F32 = float;
using E8M0PK = int32_t;

using ADataType = F4PK;
using BDataType = F4PK;
using XPackedDataType = E8M0PK;

using AccDataType = float;

using ALayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

constexpr ck::index_t DataPackedSize = 2;
constexpr ck::index_t ScaleBlockSize = 32;
constexpr ck::index_t KPerBlock = 256 / DataPackedSize; // 256 f4 = 128 fp4x2

static constexpr auto Intrawave = ck::BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = ck::BlockGemmPipelineScheduler::Interwave;

template <typename BLayout,
          typename CDataType,
          ck::index_t BlockSize,
          ck::index_t MPerBlock, ck::index_t NPerBlock, ck::index_t KPerBlock_,
          ck::index_t AK1, ck::index_t BK1,
          ck::index_t MPerXDL, ck::index_t NPerXDL,
          ck::index_t MXdlPerWave, ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched = Intrawave,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer = ck::BlockGemmPipelineVersion::v3,
          auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default>
using DeviceGemmHelperF4BlockScale = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3
    // clang-format off
         <ALayout, BLayout, CLayout,
          ADataType, XPackedDataType, BDataType, XPackedDataType, CDataType, AccDataType, CDataType,
          AElementOp, BElementOp, CElementOp, GemmSpec,
          ScaleBlockSize, BlockSize,
          MPerBlock, NPerBlock, KPerBlock_,
          AK1, BK1,
          MPerXDL, NPerXDL,
          MXdlPerWave, NXdlPerWave,
          ABlockTransferThreadClusterLengths_AK0_M_AK1,
          S<1, 0, 2>, S<1, 0, 2>,
          2, AK1, AK1,
          true,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
          S<1, 0, 2>, S<1, 0, 2>,
          2, BK1, BK1,
          true,
          CShuffleMXdlPerWavePerShuffle,
          CShuffleNXdlPerWavePerShuffle,
          CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlkGemmPipeSched,
          BlkGemmPipelineVer,
          ADataType, BDataType>;
// clang-format on

// Torch-free implementation: accepts raw device pointers and dimensions.
// Returns 0 on success, -1 on error.
template <typename CDataType, typename DeviceGemmInstance>
__forceinline__ int gemm_a4w4_blockscale_impl(
    const void* a_ptr,
    const void* a_scale_ptr,
    const void* b_ptr,
    const void* b_scale_ptr,
    void* out_ptr,
    int M, int N, int K,
    int StrideA,
    int StrideB,
    int StrideC,
    int Scale_Stride_A,
    int Scale_Stride_B,
    int splitK,
    hipStream_t stream)
{
    int KBatch = static_cast<int>(std::pow(2, splitK));

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    auto device_gemm = DeviceGemmInstance{};
    auto invoker = device_gemm.MakeInvoker();
    auto argument = device_gemm.MakeArgument(
        static_cast<const ADataType*>(a_ptr),
        static_cast<const XPackedDataType*>(a_scale_ptr),
        static_cast<const BDataType*>(b_ptr),
        static_cast<const XPackedDataType*>(b_scale_ptr),
        static_cast<CDataType*>(out_ptr),
        M, N, K,
        StrideA, Scale_Stride_A,
        StrideB, Scale_Stride_B,
        StrideC, KBatch,
        a_element_op, b_element_op, c_element_op);

    if (!device_gemm.IsSupportedArgument(argument))
        return -1;

    invoker.Run(argument, StreamConfig{stream});
    return 0;
}

#endif // USE_ROCM
