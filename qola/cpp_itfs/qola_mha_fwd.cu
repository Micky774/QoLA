// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin cpp_itfs entry point for AITER's mha_fwd.

#include "qola_mha_fwd.h"

QOLA_NS_BEGIN

float mha_fwd(const aiter::mha_fwd_args& args, hipStream_t stream)
{
    const ck_tile::stream_config s{stream};
    return ::aiter::mha_fwd(args, s);
}

QOLA_NS_END
