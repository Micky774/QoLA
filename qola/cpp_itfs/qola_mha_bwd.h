// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// QoLA cpp_itfs wrapper for AITER's mha_bwd kernel.
#pragma once

#include "qola_common.h"
#include "mha_bwd.h"  // aiter::mha_bwd_args, aiter::mha_bwd()
#include "ck_tile/host/stream_config.hpp"

QOLA_NS_BEGIN

__attribute__((visibility("default")))
float mha_bwd(const aiter::mha_bwd_args& args, const ck_tile::stream_config& stream_config);

QOLA_NS_END
