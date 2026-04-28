# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
from .submodule import default_aiter_root, ensure_aiter_commit


def build_kernels(
    manifest_path: str,
    *,
    output_dir: str,
    aiter_root: Optional[str] = None,
    archs: Optional[List[str]] = None,
    verbose: bool = False,
    build_mode: Optional[str] = None,
    aiter_commit: Optional[str] = None,
) -> dict[str, Any]:
    """Build AITER kernel modules from a consumer manifest.

    Parameters
    ----------
    manifest_path
        Path to the TOML consumer manifest.
    aiter_root
        Path to the AITER source tree root.  When ``None``, defaults to the
        bundled submodule at ``<QoLA repo>/3rdparty/aiter``.
    output_dir
        Root of the structured output directory.
    archs
        GPU arch targets (e.g. ``["gfx942"]``).  When provided, wins over
        the manifest's ``[build].architectures``.  When ``None``, falls back
        to manifest, then ``$GPU_ARCHS``, then ``"native"``.
    verbose
        Forward verbose flag to ``build_module()`` calls.
    build_mode
        ``"pybind"`` for torch-enabled Python modules, or ``"cpp_itfs"`` for
        torch-free C-linkable shared libraries.  When provided, wins over
        the manifest's ``[build].mode``.  Per-module ``mode`` entries in
        ``[[modules]]`` still take final precedence (most specific scope).
        When ``None`` everywhere, defaults to ``"pybind"``.
    aiter_commit
        AITER commit to checkout in *aiter_root* before building.  When
        provided, overrides the manifest's ``[qola] aiter_commit``.  When
        unset everywhere, builds against whatever is currently checked out.

    Returns
    -------
    dict
        Contents of the written ``manifest.json``.
    """
    if aiter_root is None:
        aiter_root = default_aiter_root()
    aiter_root = str(Path(aiter_root).resolve())
    output_dir = str(Path(output_dir).resolve())
    manifest_path = str(Path(manifest_path).resolve())

    # Save env vars we'll override so we can restore them on exit.
    prev_gpu_archs = os.environ.get("GPU_ARCHS")
    prev_jit_dir = os.environ.get("AITER_JIT_DIR")

    # Read manifest once for archs and aiter_commit fallbacks (load_manifest
    # re-parses it later for the full module schema).
    with open(manifest_path, "rb") as f:
        manifest = tomllib.load(f)

    # Resolve and apply AITER commit before any path resolution that depends
    # on the AITER tree contents.
    effective_commit = aiter_commit or manifest.get("qola", {}).get("aiter_commit")
    ensure_aiter_commit(aiter_root, effective_commit)

    # Fall back to manifest's [build] architectures when not specified via CLI.
    if not archs:
        archs = manifest.get("build", {}).get("architectures")

    if archs:
        os.environ["GPU_ARCHS"] = ";".join(archs)

    # Redirect AITER's JIT build dir so we can harvest the .so files.
    jit_build_dir = os.path.join(output_dir, "_jit_build")
    os.environ["AITER_JIT_DIR"] = jit_build_dir
    os.makedirs(jit_build_dir, exist_ok=True)

    try:
        return _build_kernels_inner(
            aiter_root, output_dir, manifest_path, archs,
            jit_build_dir, verbose, build_mode,
        )
    finally:
        _restore_env("GPU_ARCHS", prev_gpu_archs)
        _restore_env("AITER_JIT_DIR", prev_jit_dir)


def _restore_env(key: str, prev: Optional[str]) -> None:
    """Restore an environment variable to its previous value, or remove it."""
    if prev is not None:
        os.environ[key] = prev
    else:
        os.environ.pop(key, None)


def _build_kernels_inner(
    aiter_root: str,
    output_dir: str,
    manifest_path: str,
    archs: Optional[List[str]],
    jit_build_dir: str,
    verbose: bool,
    build_mode: str,
) -> dict[str, Any]:
    # 1. Resolve namespace
    ns = build_namespace(aiter_root)

    # 2. Parse manifest
    specs = load_manifest(manifest_path, ns, build_mode=build_mode)

    # 3. Load build_module from AITER
    build_module = load_build_module_fn(aiter_root)

    # 4. Prepare output directories
    lib_dir = os.path.join(output_dir, "lib")
    configs_dir = os.path.join(output_dir, "configs")
    for d in (lib_dir, configs_dir):
        os.makedirs(d, exist_ok=True)

    _copy_tuning_csvs(ns, configs_dir)

    # 4b. Export public headers (with namespace baked into qola_config.h)
    # so prebuilt caches are self-contained for downstream C++ consumers.
    with open(manifest_path, "rb") as f:
        namespace = tomllib.load(f).get("qola", {}).get("namespace", "")
    _export_public_headers(output_dir, namespace)

    # 4c. Generate embedded HSA header.
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


_PUBLIC_HEADERS = ("qola_common.h", "qola_mha_fwd.h", "qola_mha_bwd.h")


def _export_public_headers(output_dir: str, namespace: str) -> None:
    """Stage public headers under ``${output_dir}/include/``.

    Copies the stable cpp_itfs API headers verbatim and emits a fresh
    ``qola_config.h`` with the resolved namespace baked in.  Downstream
    consumers only need to add this directory to their include path —
    no ``-DQOLA_NAMESPACE`` flag is required at consumer compile time.
    """
    src_dir = Path(__file__).resolve().parent.parent / "cpp_itfs"
    dst_dir = Path(output_dir) / "include"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for header in _PUBLIC_HEADERS:
        shutil.copy2(src_dir / header, dst_dir / header)

    if namespace:
        content = (
            "// Generated by QoLA. Do not edit.\n"
            f"#define QOLA_NAMESPACE {namespace}\n"
        )
    else:
        content = (
            "// Generated by QoLA. Do not edit.\n"
            "/* No namespace configured. */\n"
        )
    (dst_dir / "qola_config.h").write_text(content)


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
