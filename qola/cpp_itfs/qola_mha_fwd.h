// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// QoLA cpp_itfs wrapper for AITER's mha_fwd kernel.
#pragma once

#include "qola_common.h"
#include "mha_fwd.h"  // aiter::mha_fwd_args, aiter::mha_fwd()

QOLA_NS_BEGIN

__attribute__((visibility("default")))
float mha_fwd(const aiter::mha_fwd_args& args, hipStream_t stream);

QOLA_NS_END
