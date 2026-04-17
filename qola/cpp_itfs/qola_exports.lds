/* SPDX-License-Identifier: MIT */
/* Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved. */

/* Linker version script for QoLA cpp_itfs shared libraries.
 *
 * Only symbols in the qola:: namespace are exported.  Everything else —
 * including aiter:: symbols that carry explicit visibility("default") in
 * their headers — is forced local.  This prevents symbol collisions when
 * multiple QoLA-built (or AITER-built) .so files coexist in one process.
 *
 * Usage:  hipcc -shared ... -Wl,--version-script,qola_exports.lds
 */
{
  global:
    _ZN4qola*;          /* qola::*                                    */
    _ZTIN4qola*;        /* qola:: typeinfo names  (for exceptions)    */
    _ZTSN4qola*;        /* qola:: typeinfo strings                    */
  local:
    *;
};
