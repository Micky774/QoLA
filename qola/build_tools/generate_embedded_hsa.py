#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
# This file is copied from
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/hip/flash_attn/ck/fav_v3/generate_aiter_embedded_hsa.py

"""Generate a C++ header embedding AITER HSA binary .co files.

The generated header provides an ``std::unordered_map`` that maps
``"hsa/{arch}/{subdir}/{kernel}.co"`` keys to ``std::string_view``
values pointing at compile-time-embedded byte arrays.  When compiled
with ``-DAITER_EMBEDDED_HSA_HEADER='"header.h"'``, AITER's
``load_asm_kernel`` uses this map instead of reading ``.co`` files
from disk via ``AITER_ASM_DIR``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def sanitize_identifier(name: str) -> str:
    """Convert a file path to a valid C++ identifier."""
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def bytes_to_hex_array(data: bytes, bytes_per_line: int = 16) -> str:
    """Convert bytes to a formatted C hex array string."""
    hex_bytes = []
    for i, byte in enumerate(data):
        if i > 0 and i % bytes_per_line == 0:
            hex_bytes.append("\n    ")
        hex_bytes.append(f"0x{byte:02x}")
        if i < len(data) - 1:
            hex_bytes.append(",")
    return "".join(hex_bytes)


def generate_embedded_hsa_header(
    hsa_dir: Path, output_file: Path, subdirs: list[str]
) -> int:
    """Generate a C++ header embedding all .co files from *subdirs*.

    Parameters
    ----------
    hsa_dir
        Base directory containing HSA files (e.g. ``3rdparty/aiter/hsa``).
    output_file
        Path to the output header file.
    subdirs
        Subdirectories to scan for ``.co`` files
        (e.g. ``["gfx950/fmha_v3_fwd", "gfx950/fmha_v3_bwd"]``).

    Returns
    -------
    int
        Number of ``.co`` files embedded.
    """
    co_files: list[tuple[str, Path]] = []
    for subdir in subdirs:
        pattern_dir = hsa_dir / subdir
        if pattern_dir.exists():
            for co_file in sorted(pattern_dir.glob("**/*.co")):
                rel_path = co_file.relative_to(hsa_dir).as_posix()
                map_key = f"hsa/{rel_path}"
                co_files.append((map_key, co_file))

    if not co_files:
        print(f"Warning: No .co files found in {hsa_dir} under {subdirs}")
        return 0

    lines = [
        "// Auto-generated file. Do not edit.",
        "// Embedded AITER HSA binary files",
        "#pragma once",
        "",
        "#include <cstdint>",
        "#include <string>",
        "#include <string_view>",
        "#include <unordered_map>",
        "",
        "// Define AITER_EMBEDDED_HSA_MAP macro so that aiter_hip_common.h",
        "// can detect the embedded map is available via #if defined(AITER_EMBEDDED_HSA_MAP)",
        "#define AITER_EMBEDDED_HSA_MAP ::aiter_hsa::embedded_hsa_map",
        "",
        "namespace aiter_hsa {",
        "",
    ]

    array_entries = []
    for map_key, co_file in co_files:
        with open(co_file, "rb") as f:
            data = f.read()

        if len(data) > 0:
            safe_name = sanitize_identifier(co_file.relative_to(hsa_dir).as_posix())
            array_name = f"data_{safe_name}"
            file_size = len(data)
            array_entries.append((map_key, array_name, file_size))

            hex_array = bytes_to_hex_array(data)
            lines.append(
                f"alignas(4096) inline const unsigned char {array_name}[] = {{\n    {hex_array}\n}};"
            )
            lines.append("")

    lines.append(
        "inline const std::unordered_map<std::string, std::string_view> embedded_hsa_map = {"
    )
    for map_key, array_name, file_size in array_entries:
        lines.append(
            f'    {{"{map_key}", std::string_view(reinterpret_cast<const char*>({array_name}), {file_size})}},'
        )
    lines.append("};")
    lines.append("")
    lines.append("} // namespace aiter_hsa")
    lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    return len(array_entries)


def main():
    parser = argparse.ArgumentParser(
        description="Generate embedded HSA header from AITER .co files"
    )
    parser.add_argument(
        "--hsa-dir", required=True, type=Path, help="Path to the aiter hsa directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Path to the output header file"
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        required=True,
        help="Subdirectories to scan for .co files (e.g. gfx950/fmha_v3_fwd)",
    )

    args = parser.parse_args()

    if not args.hsa_dir.exists():
        print(f"Error: HSA directory does not exist: {args.hsa_dir}", file=sys.stderr)
        return 1

    count = generate_embedded_hsa_header(args.hsa_dir, args.output, args.subdirs)
    print(f"Generated {args.output} with {count} embedded .co files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
