# SPDX-License-Identifier: MIT
"""TOML manifest parsing and optCompilerConfig.json eval."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from .resolver import AiterNamespace, make_eval_globals


@dataclass
class BuildSpec:
    """Fully resolved arguments for one ``build_module()`` call."""

    md_name: str
    srcs: List[str]
    flags_extra_cc: List[str] = field(default_factory=list)
    flags_extra_hip: List[str] = field(default_factory=list)
    blob_gen_cmd: Union[str, List[str]] = ""
    extra_include: List[str] = field(default_factory=list)
    extra_ldflags: Optional[List[str]] = None
    verbose: bool = False
    is_python_module: bool = True
    is_standalone: bool = False
    torch_exclude: bool = False
    hipify: bool = False
    hip_clang_path: Optional[str] = None


# Defaults matching core.py's d_opt_build_args (line 712)
_DEFAULTS: dict[str, Any] = {
    "srcs": [],
    "flags_extra_cc": [],
    "flags_extra_hip": [],
    "extra_ldflags": None,
    "extra_include": [],
    "verbose": False,
    "is_python_module": True,
    "is_standalone": False,
    "torch_exclude": False,
    "blob_gen_cmd": "",
    "hipify": False,
    "hip_clang_path": None,
}

# QoLA root directory (one level above this file's package).
_QOLA_ROOT = str(Path(__file__).resolve().parent.parent.parent)

# Registry of cpp_itfs source replacements, loaded from the external TOML file
# so that new modules can be added without touching Python code.
_CPP_ITFS_REGISTRY = os.path.join(_QOLA_ROOT, "qola", "cpp_itfs", "registry.toml")


def _load_cpp_itfs_src_map() -> Dict[str, Dict[str, List[str]]]:
    """Load the cpp_itfs source registry from ``registry.toml``."""
    with open(_CPP_ITFS_REGISTRY, "rb") as f:
        return tomllib.load(f)


def load_manifest(
    manifest_path: str,
    ns: AiterNamespace,
    build_mode: str = "pybind",
) -> List[BuildSpec]:
    """Parse a TOML manifest and return resolved :class:`BuildSpec` instances.

    Parameters
    ----------
    manifest_path
        Path to the TOML consumer manifest.
    ns
        Resolved AITER namespace.
    build_mode
        ``"pybind"`` (default) for torch-enabled Python modules, or
        ``"cpp_itfs"`` for torch-free C-linkable shared libraries.
        Can be overridden per-module via the ``mode`` key in the manifest.
        The CLI ``--mode`` flag sets this, but the manifest's ``[build] mode``
        takes precedence, and per-module ``mode`` takes highest precedence.

    Manifest schema::

        [qola]
        aiter_commit = "d32b0cb6..."
        namespace = "te"                  # optional, prevents symbol collisions

        [build]
        architectures = ["gfx942", "gfx950"]
        mode = "cpp_itfs"                 # optional global default

        [[modules]]
        name = "module_gemm_a4w4_asm"
        mode = "pybind"                   # optional per-module override

        [[mha_variants]]
        dtype = "bf16"
        use_mask = true
        ...
    """
    with open(manifest_path, "rb") as f:
        manifest = tomllib.load(f)

    config_json_path = os.path.join(
        ns.AITER_ROOT_DIR, "aiter", "jit", "optCompilerConfig.json"
    )
    eval_globals = make_eval_globals(ns)

    # Resolve effective build mode: CLI < [build].mode < per-module mode.
    global_mode = manifest.get("build", {}).get("mode", build_mode)
    namespace = manifest.get("qola", {}).get("namespace", "")

    specs: List[BuildSpec] = []
    mha_section = manifest.get("mha_variants", [])
    module_names = {m["name"] for m in manifest.get("modules", [])}

    # Determine whether [[mha_variants]] filters are consumed by static
    # libmha_fwd/libmha_bwd modules (combined filter into one .so) or
    # expanded as separate pybind modules.
    _MHA_LIB_MODULES = {"libmha_fwd", "libmha_bwd"}
    mha_consumed_by_static = bool(mha_section and (module_names & _MHA_LIB_MODULES))

    # Pre-compute variant filters once if needed.
    mha_filters: List[Dict[str, Any]] = []
    if mha_consumed_by_static:
        from .variant_matrix import compute_mha_variant_filters

        mha_filters = compute_mha_variant_filters(mha_section, ns)

    # --- static modules ---
    for mod_entry in manifest.get("modules", []):
        name = mod_entry["name"]
        mod_mode = mod_entry.get("mode", global_mode)
        overrides = {k: v for k, v in mod_entry.items() if k not in ("name", "mode")}
        spec = _resolve_static_module(name, config_json_path, eval_globals, overrides)

        if mha_filters and name in _MHA_LIB_MODULES:
            _apply_mha_variant_filter(spec, name, mha_filters, ns)

        if mod_mode == "cpp_itfs":
            _apply_cpp_itfs(spec, name)

        if namespace:
            spec.md_name = f"{namespace}_{spec.md_name}"
            spec.flags_extra_cc.append(f"-DQOLA_NAMESPACE={namespace}")

        specs.append(spec)

    # --- MHA variants (pybind per-variant expansion) ---
    # Only expand as separate pybind modules when libmha_fwd/bwd are NOT
    # declared as static modules — otherwise the variants are already
    # folded into the static entries above.
    if mha_section and not mha_consumed_by_static:
        from .variant_matrix import expand_mha_variants

        mha_specs = expand_mha_variants(mha_section, ns)
        if namespace:
            for spec in mha_specs:
                spec.md_name = f"{namespace}_{spec.md_name}"
                spec.flags_extra_cc.append(f"-DQOLA_NAMESPACE={namespace}")
        specs.extend(mha_specs)

    return specs


def _apply_cpp_itfs(spec: BuildSpec, module_name: str) -> None:
    """Rewrite *spec* for torch-free cpp_itfs mode.

    Drops pybind source files and replaces them with QoLA's cpp_itfs
    wrappers.  Forces ``torch_exclude=True`` and ``is_python_module=False``.

    For CK modules, AITER's codegen and common headers are used
    unmodified — QoLA's stub torch headers (``torch_stubs/``) shadow
    real torch so the generated code compiles without libtorch.
    The ``add_includes`` order in the registry matters: stub paths
    must come before AITER include paths.
    """
    src_map = _load_cpp_itfs_src_map()
    mapping = src_map.get(module_name)
    if mapping is None:
        raise ValueError(
            f"No cpp_itfs variant available for module '{module_name}'. "
            f"Supported modules: {', '.join(sorted(src_map))}"
        )

    # Drop pybind sources (match by basename).
    drop_basenames = {os.path.basename(s) for s in mapping["drop_srcs"]}
    spec.srcs = [s for s in spec.srcs if os.path.basename(s) not in drop_basenames]

    # Add cpp_itfs sources (resolved relative to QoLA root).
    for src in mapping["add_srcs"]:
        spec.srcs.append(os.path.join(_QOLA_ROOT, src))

    # Prepend cpp_itfs include directories (order matters — stubs first).
    new_includes = [os.path.join(_QOLA_ROOT, inc) for inc in mapping.get("add_includes", [])]
    spec.extra_include = new_includes + spec.extra_include

    spec.torch_exclude = True
    spec.is_python_module = False


# Directions in blob_gen_cmd that should be filtered per-variant.
_MHA_FWD_FILTERED_DIRS = {"fwd"}
_MHA_BWD_FILTERED_DIRS = {"bwd"}

# Regex to extract the ``-d <direction>`` from a generate.py command.
_DIR_RE = re.compile(r"-d\s+(\S+)")


def _apply_mha_variant_filter(
    spec: BuildSpec,
    module_name: str,
    mha_filters: List[Dict[str, Any]],
    ns: "AiterNamespace",
) -> None:
    """Replace unfiltered ``blob_gen_cmd`` with per-variant filtered commands.

    For each CK codegen direction that supports variant filtering (``fwd``
    for libmha_fwd, ``bwd`` for libmha_bwd), the single unfiltered
    ``generate.py`` invocation is replaced with N invocations — one per
    ``[[mha_variants]]`` entry — each carrying the variant's ``--filter``
    and ``--receipt``.

    Directions that don't support variant filtering (``fwd_splitkv``,
    ``batch_prefill``) are kept unchanged.  All invocations write to the
    same ``--output_dir``, so CK template instances from different
    variants are compiled together into a single ``.so``.
    """
    filtered_dirs = (
        _MHA_FWD_FILTERED_DIRS if module_name == "libmha_fwd"
        else _MHA_BWD_FILTERED_DIRS
    )

    old_cmds: List[str] = (
        spec.blob_gen_cmd if isinstance(spec.blob_gen_cmd, list)
        else [spec.blob_gen_cmd] if spec.blob_gen_cmd else []
    )

    new_cmds: List[str] = []
    for cmd in old_cmds:
        m = _DIR_RE.search(cmd)
        direction = m.group(1) if m else None

        if direction not in filtered_dirs:
            # Keep unfiltered (splitkv, batch_prefill, etc.)
            new_cmds.append(cmd)
            continue

        # Replace with one filtered invocation per variant.
        for vf in mha_filters:
            # Strip any existing --receipt from the original command and
            # inject the variant's receipt + filter.
            base = re.sub(r"--receipt\s+\S+", "", cmd).rstrip()
            new_cmds.append(
                f"{base} --receipt {vf['receipt']} --filter {vf['filter']}"
            )

    spec.blob_gen_cmd = new_cmds


def _resolve_static_module(
    op_name: str,
    config_json_path: str,
    eval_globals: dict[str, Any],
    overrides: dict[str, Any],
) -> BuildSpec:
    """Look up *op_name* in ``optCompilerConfig.json`` and eval its fields."""
    with open(config_json_path, "r") as f:
        all_entries: dict = json.load(f)

    if op_name not in all_entries:
        raise ValueError(
            f"Module '{op_name}' not found in optCompilerConfig.json"
        )
    raw = dict(all_entries[op_name])
    raw.update(overrides)
    return _eval_entry(op_name, raw, eval_globals)


def _eval_entry(
    op_name: str, raw: dict[str, Any], eval_globals: dict[str, Any]
) -> BuildSpec:
    """Eval all string fields in a JSON entry.  Mirrors ``convert()``."""
    resolved: dict[str, Any] = dict(_DEFAULTS)
    resolved["md_name"] = op_name

    for key, val in raw.items():
        if isinstance(val, list):
            evaled = []
            for el in val:
                if isinstance(el, str):
                    _ensure_torch(el, eval_globals)
                    evaled.append(eval(el, eval_globals))  # noqa: S307
                else:
                    evaled.append(el)
            resolved[key] = evaled
        elif isinstance(val, str):
            _ensure_torch(val, eval_globals)
            resolved[key] = eval(val, eval_globals)  # noqa: S307
        else:
            resolved[key] = val

    md_name = resolved.get("md_name", op_name)

    return BuildSpec(
        md_name=str(md_name),
        srcs=resolved.get("srcs", []),
        flags_extra_cc=resolved.get("flags_extra_cc", []),
        flags_extra_hip=resolved.get("flags_extra_hip", []),
        blob_gen_cmd=resolved.get("blob_gen_cmd", ""),
        extra_include=resolved.get("extra_include", []),
        extra_ldflags=resolved.get("extra_ldflags"),
        verbose=bool(resolved.get("verbose", False)),
        is_python_module=bool(resolved.get("is_python_module", True)),
        is_standalone=bool(resolved.get("is_standalone", False)),
        torch_exclude=bool(resolved.get("torch_exclude", False)),
        hipify=bool(resolved.get("hipify", False)),
        hip_clang_path=resolved.get("hip_clang_path"),
    )


def _ensure_torch(expr: str, eval_globals: dict[str, Any]) -> None:
    """Lazily import torch into *eval_globals* if *expr* references it."""
    if "torch" in expr and "torch" not in eval_globals:
        try:
            import torch

            eval_globals["torch"] = torch
        except ImportError:
            pass
