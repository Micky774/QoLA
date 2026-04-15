# SPDX-License-Identifier: MIT
"""AOTA build layer — manifest-driven ahead-of-time AITER kernel compilation."""

from .builder import build_kernels

__all__ = ["build_kernels"]
