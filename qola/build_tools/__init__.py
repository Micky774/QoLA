# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AOTA build layer — manifest-driven ahead-of-time AITER kernel compilation."""

from .builder import build_kernels
from .submodule import checkout_aiter

__all__ = ["build_kernels", "checkout_aiter"]
