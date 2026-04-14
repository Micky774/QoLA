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
    hsa_subdirs: List[str] = field(default_factory=list)


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
_CPP_ITFS_DIR = os.path.join(_QOLA_ROOT, "qola", "cpp_itfs")

# Registry of cpp_itfs source replacements, loaded from the external TOML file
# so that new modules can be added without touching Python code.
_CPP_ITFS_REGISTRY = os.path.join(_CPP_ITFS_DIR, "registry.toml")
_QOLA_CONFIG_HEADER = os.path.join(_CPP_ITFS_DIR, "qola_config.h")


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

        # Separate fwd/bwd variant sections (different option spaces):
        [[mha_fwd_variants]]
        dtype = ["bf16", "fp16"]
        return_lse = true
        ...

        [[mha_bwd_variants]]
        dtype = ["bf16", "fp16"]
        has_dropout = false
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
    fwd_section = manifest.get("mha_fwd_variants", [])
    module_names = {m["name"] for m in manifest.get("modules", [])}

    has_fwd_variants = bool(fwd_section)
    has_static_fwd = "libmha_fwd" in module_names

    # Keys consumed by load_manifest before passing to _resolve_static_module.
    _MANIFEST_KEYS = {"name", "mode", "drop_srcs", "drop_directions", "hsa_subdirs"}

    # --- static modules ---
    # NOTE: Variant filtering is NOT applied to static libmha_fwd /
    # libmha_bwd modules.  CK's generate.py produces both instance .cpp
    # files and the dispatch API file (fmha_*_api.cpp) on every call.
    # Running it N times with different --filter patterns overwrites the
    # API dispatch, leaving only the last filter's branches.
    for mod_entry in manifest.get("modules", []):
        name = mod_entry["name"]
        mod_mode = mod_entry.get("mode", global_mode)
        drop_srcs = set(mod_entry.get("drop_srcs", []))
        drop_directions = set(mod_entry.get("drop_directions", []))
        overrides = {k: v for k, v in mod_entry.items() if k not in _MANIFEST_KEYS}
        spec = _resolve_static_module(name, config_json_path, eval_globals, overrides)

        if drop_srcs:
            spec.srcs = [s for s in spec.srcs if os.path.basename(s) not in drop_srcs]
        if drop_directions:
            _drop_blob_directions(spec, drop_directions)

        if mod_mode == "cpp_itfs":
            _apply_cpp_itfs(spec, name, namespace)
        else:
            # For pybind mode, hsa_subdirs comes from the manifest entry,
            # falling back to the registry (which is authoritative for the
            # module→kernel-type mapping regardless of build mode).
            manifest_hsa = mod_entry.get("hsa_subdirs")
            if manifest_hsa is not None:
                spec.hsa_subdirs = manifest_hsa
            else:
                src_map = _load_cpp_itfs_src_map()
                mapping = src_map.get(name, {})
                spec.hsa_subdirs = mapping.get("hsa_subdirs", [])

        if namespace:
            spec.md_name = f"{namespace}_{spec.md_name}"

        specs.append(spec)

    # --- MHA variants (pybind per-variant expansion, fwd only) ---
    # Only expand as separate pybind modules when libmha_fwd is NOT
    # declared as a static module — otherwise the variants are already
    # folded into the static entry above.
    if has_fwd_variants and not has_static_fwd:
        from .variant_matrix import expand_mha_variants

        mha_specs = expand_mha_variants(fwd_section, ns)
        if namespace:
            for spec in mha_specs:
                spec.md_name = f"{namespace}_{spec.md_name}"
        specs.extend(mha_specs)

    return specs


def _write_qola_config(namespace: str) -> None:
    """Write ``qola_config.h`` into the cpp_itfs directory.

    This injects ``QOLA_NAMESPACE`` via a header that only the wrapper
    files include (through ``qola_common.h``), avoiding a global ``-D``
    flag that would invalidate ninja's cache for every CK variant file.
    """
    if namespace:
        content = f"#define QOLA_NAMESPACE {namespace}\n"
    else:
        content = "/* No namespace configured. */\n"

    # Only rewrite if the content changed, to avoid needless rebuilds.
    if os.path.isfile(_QOLA_CONFIG_HEADER):
        with open(_QOLA_CONFIG_HEADER) as f:
            if f.read() == content:
                return
    with open(_QOLA_CONFIG_HEADER, "w") as f:
        f.write(content)


def _apply_cpp_itfs(spec: BuildSpec, module_name: str, namespace: str = "") -> None:
    """Rewrite *spec* for torch-free cpp_itfs mode.

    Drops pybind source files and replaces them with QoLA's cpp_itfs
    wrappers.  Forces ``torch_exclude=True`` and ``is_python_module=False``.

    For CK modules, AITER's codegen and common headers are used
    unmodified — QoLA's stub torch headers (``torch_stubs/``) shadow
    real torch so the generated code compiles without libtorch.
    The ``add_includes`` order in the registry matters: stub paths
    must come before AITER include paths.

    The *namespace* is injected via a generated ``qola_config.h`` header
    placed in the cpp_itfs include directory rather than via a global
    ``-D`` flag.  This avoids changing compile commands for the thousands
    of CK variant source files (which never reference the macro), letting
    ninja skip their recompilation when only the namespace changes.
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
    new_includes = [
        os.path.join(_QOLA_ROOT, inc) for inc in mapping.get("add_includes", [])
    ]
    spec.extra_include = new_includes + spec.extra_include

    # Write namespace config into cpp_itfs/ so only the wrapper files see it.
    _write_qola_config(namespace)

    spec.torch_exclude = True
    spec.is_python_module = False
    spec.hsa_subdirs = mapping.get("hsa_subdirs", [])

    # Apply default linker flags (e.g. version script to hide aiter:: symbols).
    defaults = src_map.get("defaults", {})
    vs = defaults.get("version_script")
    if vs:
        vs_path = os.path.join(_QOLA_ROOT, vs)
        if spec.extra_ldflags is None:
            spec.extra_ldflags = []
        spec.extra_ldflags.append(f"-Wl,--version-script,{vs_path}")


# Regex to extract the ``-d <direction>`` from a generate.py command.
_DIR_RE = re.compile(r"-d\s+(\S+)")


def _drop_blob_directions(spec: BuildSpec, drop_directions: set) -> None:
    """Remove ``blob_gen_cmd`` entries whose ``-d <direction>`` is in *drop_directions*."""
    old_cmds: List[str] = (
        spec.blob_gen_cmd
        if isinstance(spec.blob_gen_cmd, list)
        else [spec.blob_gen_cmd] if spec.blob_gen_cmd else []
    )
    spec.blob_gen_cmd = [
        cmd for cmd in old_cmds
        if not ((m := _DIR_RE.search(cmd)) and m.group(1) in drop_directions)
    ]


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
        raise ValueError(f"Module '{op_name}' not found in optCompilerConfig.json")
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
