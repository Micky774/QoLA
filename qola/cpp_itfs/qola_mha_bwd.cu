// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin cpp_itfs entry point for AITER's mha_bwd.

#include "qola_mha_bwd.h"

QOLA_NS_BEGIN

float mha_bwd(const aiter::mha_bwd_args& args, hipStream_t stream)
{
    const ck_tile::stream_config s{stream};
    return ::aiter::mha_bwd(args, s);
}

QOLA_NS_END
