# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QoLA command-line interface."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qola",
        description="QoLA's ahead-of-time AITER kernel builder",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build kernels from a manifest")
    build_p.add_argument(
        "--manifest",
        "-m",
        required=True,
        help="Path to the TOML consumer manifest",
    )
    build_p.add_argument(
        "--aiter-root",
        "-a",
        default=None,
        help="Path to the AITER source tree root. Defaults to the bundled "
        "submodule at <QoLA repo>/3rdparty/aiter.",
    )
    build_p.add_argument(
        "--aiter-commit",
        default=None,
        help="AITER git commit (full SHA, short SHA, tag, or branch) to "
        "fetch and checkout in --aiter-root before building. Overrides the "
        "manifest's [qola] aiter_commit. When unset everywhere, builds "
        "against whatever is currently checked out.",
    )
    build_p.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Structured output directory",
    )
    build_p.add_argument(
        "--arch",
        action="append",
        dest="archs",
        help="GPU arch target (e.g. gfx942). Repeatable.",
    )
    build_p.add_argument(
        "--mode",
        choices=["pybind", "cpp_itfs"],
        default=None,
        help="Build mode: 'pybind' (torch-enabled Python modules) or "
        "'cpp_itfs' (torch-free C-linkable shared libraries). When set, "
        "overrides the manifest's [build].mode but per-module 'mode' entries "
        "in [[modules]] still win. Defaults to 'pybind' if unset everywhere.",
    )
    build_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        from .build_tools import build_kernels

        archs: list[str] | None = None
        if args.archs:
            archs = [a for entry in args.archs for a in entry.split(";") if a]

        result = build_kernels(
            manifest_path=args.manifest,
            aiter_root=args.aiter_root,
            output_dir=args.output_dir,
            archs=archs,
            verbose=args.verbose,
            build_mode=args.mode,
            aiter_commit=args.aiter_commit,
        )
        s = result["summary"]
        print(
            f"Build complete: {s['success']}/{s['total']} succeeded"
            f", {s['failed']} failed."
        )
        if s["failed"]:
            for r in result["modules"]:
                if not r["success"]:
                    print(f"  FAILED: {r['md_name']}: {r['error']}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
