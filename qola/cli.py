# SPDX-License-Identifier: MIT
"""AOTA command-line interface."""

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
        "--manifest", "-m", required=True,
        help="Path to the TOML consumer manifest",
    )
    build_p.add_argument(
        "--aiter-root", "-a", required=True,
        help="Path to the AITER source tree root",
    )
    build_p.add_argument(
        "--output-dir", "-o", required=True,
        help="Structured output directory",
    )
    build_p.add_argument(
        "--arch", action="append", dest="archs",
        help="GPU arch target (e.g. gfx942). Repeatable.",
    )
    build_p.add_argument(
        "--verbose", "-v", action="store_true",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        from .build import build_kernels

        result = build_kernels(
            manifest_path=args.manifest,
            aiter_root=args.aiter_root,
            output_dir=args.output_dir,
            archs=args.archs,
            verbose=args.verbose,
        )
        s = result["summary"]
        print(f"Build complete: {s['success']}/{s['total']} succeeded"
              f", {s['failed']} failed.")
        if s["failed"]:
            for r in result["modules"]:
                if not r["success"]:
                    print(f"  FAILED: {r['md_name']}: {r['error']}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
