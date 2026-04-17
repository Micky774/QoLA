// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// QoLA cpp_itfs wrapper for AITER's mha_fwd kernel.
#pragma once

#include "qola_common.h"
#include "mha_fwd.h"  // aiter::mha_fwd_args, aiter::mha_fwd()
#include "ck_tile/host/stream_config.hpp"

QOLA_NS_BEGIN

__attribute__((visibility("default")))
float mha_fwd(const aiter::mha_fwd_args& args, const ck_tile::stream_config& stream_config);

QOLA_NS_END
