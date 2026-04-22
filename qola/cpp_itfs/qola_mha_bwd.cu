// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin cpp_itfs entry point for AITER's mha_bwd.

#include "qola_mha_bwd.h"

QOLA_NS_BEGIN

float mha_bwd(const aiter::mha_bwd_args& args, const ck_tile::stream_config& stream_config)
{
    return ::aiter::mha_bwd(args, stream_config);
}

QOLA_NS_END
