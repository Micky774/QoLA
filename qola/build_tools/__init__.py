# SPDX-License-Identifier: MIT
"""AOTA build layer — manifest-driven ahead-of-time AITER kernel compilation."""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from .config import BuildSpec, load_manifest
from .resolver import AiterNamespace, build_namespace, load_build_module_fn


def build_kernels(
    manifest_path: str,
    aiter_root: str,
    output_dir: str,
    archs: Optional[List[str]] = None,
    verbose: bool = False,
    build_mode: str = "pybind",
) -> dict[str, Any]:
    """Build AITER kernel modules from a consumer manifest.

    Parameters
    ----------
    manifest_path
        Path to the TOML consumer manifest.
    aiter_root
        Path to the AITER source tree root (``3rdparty/aiter/``).
    output_dir
        Root of the structured output directory.
    archs
        GPU arch targets (e.g. ``["gfx942"]``).  Falls back to
        ``$GPU_ARCHS`` or ``"native"``.
    verbose
        Forward verbose flag to ``build_module()`` calls.
    build_mode
        ``"pybind"`` (default) for torch-enabled Python modules, or
        ``"cpp_itfs"`` for torch-free C-linkable shared libraries.

    Returns
    -------
    dict
        Contents of the written ``manifest.json``.
    """
    aiter_root = str(Path(aiter_root).resolve())
    output_dir = str(Path(output_dir).resolve())
    manifest_path = str(Path(manifest_path).resolve())

    # Fall back to manifest's [build] architectures when not specified via CLI.
    if not archs:
        with open(manifest_path, "rb") as f:
            archs = tomllib.load(f).get("build", {}).get("architectures")

    if archs:
        os.environ["GPU_ARCHS"] = ";".join(archs)

    # Redirect AITER's JIT build dir so we can harvest the .so files.
    jit_build_dir = os.path.join(output_dir, "_jit_build")
    os.environ["AITER_JIT_DIR"] = jit_build_dir
    os.makedirs(jit_build_dir, exist_ok=True)

    # 1. Resolve namespace
    ns = build_namespace(aiter_root)

    # 2. Parse manifest
    specs = load_manifest(manifest_path, ns, build_mode=build_mode)

    # 3. Load build_module from AITER
    build_module = load_build_module_fn(aiter_root)

    # 4. Prepare output directories
    lib_dir = os.path.join(output_dir, "lib")
    configs_dir = os.path.join(output_dir, "configs")
    asm_dir = os.path.join(output_dir, "asm")
    for d in (lib_dir, configs_dir, asm_dir):
        os.makedirs(d, exist_ok=True)

    _copy_tuning_csvs(ns, configs_dir)
    _copy_asm_blobs(ns, asm_dir)

    # 4b. Generate embedded HSA header.
    _generate_embedded_hsa(ns, output_dir, archs or [], specs)

    # 5. Build each module
    results: list[dict[str, Any]] = []
    for spec in specs:
        t0 = time.perf_counter()
        success = True
        error_msg = ""
        try:
            _invoke_build(build_module, spec, verbose)
        except Exception as exc:
            success = False
            error_msg = str(exc)

        # Harvest the .so — AITER places it in different locations depending
        # on the build mode (torch_exclude, is_python_module, etc.).
        so_name = f"{spec.md_name}.so"
        so_candidates = [
            os.path.join(jit_build_dir, so_name),
            os.path.join(jit_build_dir, "build", spec.md_name, "build", so_name),
        ]
        so_dst: Optional[str] = None
        for so_src in so_candidates:
            if os.path.isfile(so_src):
                so_dst = os.path.join(lib_dir, so_name)
                shutil.copy2(so_src, so_dst)
                break

        results.append(
            {
                "md_name": spec.md_name,
                "success": success,
                "error": error_msg,
                "so_path": so_dst,
                "duration_s": round(time.perf_counter() - t0, 2),
            }
        )

    # 6. Write manifest.json
    record = _write_manifest(output_dir, manifest_path, aiter_root, results)
    return record


# ------------------------------------------------------------------
# internal helpers
# ------------------------------------------------------------------


def _invoke_build(build_module_fn, spec: BuildSpec, verbose: bool) -> None:
    prev_clang = os.environ.get("HIP_CLANG_PATH")
    if spec.hip_clang_path:
        os.environ["HIP_CLANG_PATH"] = spec.hip_clang_path
    try:
        build_module_fn(
            md_name=spec.md_name,
            srcs=spec.srcs,
            flags_extra_cc=spec.flags_extra_cc,
            flags_extra_hip=spec.flags_extra_hip,
            blob_gen_cmd=spec.blob_gen_cmd,
            extra_include=spec.extra_include,
            extra_ldflags=spec.extra_ldflags,
            verbose=verbose or spec.verbose,
            is_python_module=spec.is_python_module,
            is_standalone=spec.is_standalone,
            torch_exclude=spec.torch_exclude,
            hipify=spec.hipify,
        )
    finally:
        if spec.hip_clang_path:
            if prev_clang is not None:
                os.environ["HIP_CLANG_PATH"] = prev_clang
            else:
                os.environ.pop("HIP_CLANG_PATH", None)


def _generate_embedded_hsa(
    ns: AiterNamespace,
    output_dir: str,
    archs: List[str],
    specs: List[BuildSpec],
) -> None:
    """Generate per-module embedded HSA headers and inject compile flags.

    Each module's ``hsa_subdirs`` field (set via the registry or manifest)
    declares which kernel subdirectories it needs.  A separate header is
    generated for each module that has matching ``.co`` blobs, so non-MHA
    modules never carry MHA binary data and vice-versa.  Modules with no
    ``hsa_subdirs`` are left untouched.
    """
    from .generate_embedded_hsa import generate_embedded_hsa_header

    hsa_dir = Path(os.path.join(ns.AITER_META_DIR, "hsa"))

    for spec in specs:
        if not spec.hsa_subdirs:
            continue

        # Resolve arch × kernel_type subdirs for this module.
        subdirs: List[str] = []
        for arch in archs:
            for kernel_type in spec.hsa_subdirs:
                subdir = f"{arch}/{kernel_type}"
                if (hsa_dir / subdir).is_dir():
                    subdirs.append(subdir)

        if not subdirs:
            continue

        header_dir = os.path.join(output_dir, "_embedded_hsa", spec.md_name)
        header_name = f"aiter_embedded_hsa_{spec.md_name}.h"
        header_path = os.path.join(header_dir, header_name)

        count = generate_embedded_hsa_header(hsa_dir, Path(header_path), subdirs)
        print(f"[QoLA] Embedded {count} HSA .co files into {header_path}")

        spec.flags_extra_cc.append(f'-DAITER_EMBEDDED_HSA_HEADER=\'"{header_name}"\'')
        spec.extra_include.insert(0, header_dir)


def _copy_tuning_csvs(ns: AiterNamespace, dst: str) -> None:
    src_dir = os.path.join(ns.AITER_ROOT_DIR, "aiter", "configs")
    if not os.path.isdir(src_dir):
        return
    for csv in glob.glob(os.path.join(src_dir, "*.csv")):
        shutil.copy2(csv, os.path.join(dst, os.path.basename(csv)))
    model_src = os.path.join(src_dir, "model_configs")
    if os.path.isdir(model_src):
        model_dst = os.path.join(dst, "model_configs")
        if os.path.exists(model_dst):
            shutil.rmtree(model_dst)
        shutil.copytree(model_src, model_dst)


def _copy_asm_blobs(ns: AiterNamespace, dst: str) -> None:
    hsa_root = os.path.join(ns.AITER_META_DIR, "hsa")
    if not os.path.isdir(hsa_root):
        return
    for child in Path(hsa_root).iterdir():
        if child.is_dir() and child.name.startswith("gfx"):
            arch_dst = os.path.join(dst, child.name)
            if os.path.exists(arch_dst):
                shutil.rmtree(arch_dst)
            shutil.copytree(str(child), arch_dst)


def _write_manifest(
    output_dir: str,
    manifest_path: str,
    aiter_root: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aiter_root": aiter_root,
        "manifest_src": manifest_path,
        "gpu_archs": os.environ.get("GPU_ARCHS", "native"),
        "modules": results,
        "summary": {
            "total": len(results),
            "success": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
        },
    }
    out = os.path.join(output_dir, "manifest.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    return record
