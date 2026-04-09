// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Thin cpp_itfs entry point for AITER's mha_fwd.

#include "qola_mha_fwd.h"

QOLA_NS_BEGIN

float mha_fwd(const aiter::mha_fwd_args& args, const ck_tile::stream_config& stream_config)
{
    return ::aiter::mha_fwd(args, stream_config);
}

QOLA_NS_END
