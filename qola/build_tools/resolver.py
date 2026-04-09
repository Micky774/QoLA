# SPDX-License-Identifier: MIT
"""Resolve AITER source tree paths and reconstruct the eval() namespace
that optCompilerConfig.json string fields expect."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


class AiterNamespace:
    """Reconstructed eval() namespace mirroring core.py module-level globals."""

    AITER_ROOT_DIR: str
    AITER_META_DIR: str
    AITER_CSRC_DIR: str
    AITER_GRADLIB_DIR: str
    AITER_ASM_DIR: str
    CK_DIR: str
    CK_3RDPARTY_DIR: str
    CK_HELPER_DIR: str
    AITER_CONFIGS: Any
    get_gfx: Any


def build_namespace(aiter_root: str) -> AiterNamespace:
    """Build the eval namespace from an AITER source tree path.

    Does NOT ``import aiter``.  Uses ``sys.path`` injection to reach
    ``jit/utils/`` helpers and ``exec()`` of a marked source block in
    ``core.py`` to obtain the ``AITER_CONFIGS`` singleton.
    """
    aiter_root = str(Path(aiter_root).resolve())
    jit_dir = os.path.join(aiter_root, "aiter", "jit")
    utils_dir = os.path.join(jit_dir, "utils")

    for d in (utils_dir, jit_dir):
        if d not in sys.path:
            sys.path.insert(0, d)

    # --- path constants (mirrors core.py lines 63-324) ---
    env_meta = os.environ.get("AITER_META_DIR")
    if env_meta and os.path.isdir(os.path.join(env_meta, "csrc")):
        meta = str(Path(env_meta).resolve())
    else:
        meta = aiter_root

    ns = AiterNamespace()
    ns.AITER_ROOT_DIR = aiter_root
    ns.AITER_META_DIR = meta
    ns.AITER_CSRC_DIR = os.path.join(meta, "csrc")
    ns.AITER_GRADLIB_DIR = os.path.join(meta, "gradlib")
    ns.AITER_ASM_DIR = os.path.join(meta, "hsa", "")  # trailing sep
    ns.CK_3RDPARTY_DIR = os.environ.get(
        "CK_DIR", os.path.join(meta, "3rdparty", "composable_kernel")
    )
    ns.CK_DIR = ns.CK_3RDPARTY_DIR
    ns.CK_HELPER_DIR = os.path.join(meta, "3rdparty", "ck_helper")

    # --- AITER_CONFIGS singleton ---
    ns.AITER_CONFIGS = _build_aiter_configs(
        os.path.join(jit_dir, "core.py"), aiter_root
    )

    # --- get_gfx callable ---
    try:
        import chip_info as _chip_info  # type: ignore[import-not-found]

        ns.get_gfx = _chip_info.get_gfx
    except Exception:
        ns.get_gfx = lambda: os.getenv("GPU_ARCHS", "gfx942").split(";")[-1]

    return ns


# ------------------------------------------------------------------
# AITER_CONFIGS extraction
# ------------------------------------------------------------------

_CONFIG_START = "# config_env start here"
_CONFIG_END = "# config_env end here"


def _build_aiter_configs(core_path: str, aiter_root_dir: str) -> Any:
    """Extract the AITER_CONFIG class from core.py and instantiate it.

    Executes only the marked ``# config_env start/end`` block (lines 68-279)
    in a namespace where ``AITER_ROOT_DIR`` is bound.  All properties of
    ``AITER_CONFIG`` reference ``AITER_ROOT_DIR`` through the module-level
    ``AITER_CONFIG_*`` string constants defined in the same block.
    """
    with open(core_path, "r") as f:
        source = f.read()

    start = source.find(_CONFIG_START)
    end = source.find(_CONFIG_END)
    if start == -1 or end == -1:
        raise RuntimeError(
            f"Could not find config_env markers in {core_path}. "
            "AITER core.py structure may have changed."
        )
    block = source[start : end + len(_CONFIG_END)]

    exec_ns: dict[str, Any] = {
        "os": os,
        "functools": __import__("functools"),
        "AITER_ROOT_DIR": aiter_root_dir,
    }
    exec(compile(block, core_path, "exec"), exec_ns)  # noqa: S102
    return exec_ns["AITER_CONFIGS"]


# ------------------------------------------------------------------
# eval globals
# ------------------------------------------------------------------


def make_eval_globals(ns: AiterNamespace) -> dict[str, Any]:
    """Return a globals dict for ``eval()``-ing optCompilerConfig.json fields.

    Mirrors the module-level namespace of ``core.py`` that ``convert()``
    relies on when it calls ``eval(el)``.
    """
    g: dict[str, Any] = {
        "AITER_ROOT_DIR": ns.AITER_ROOT_DIR,
        "AITER_META_DIR": ns.AITER_META_DIR,
        "AITER_CSRC_DIR": ns.AITER_CSRC_DIR,
        "AITER_GRADLIB_DIR": ns.AITER_GRADLIB_DIR,
        "AITER_ASM_DIR": ns.AITER_ASM_DIR,
        "CK_DIR": ns.CK_DIR,
        "CK_3RDPARTY_DIR": ns.CK_3RDPARTY_DIR,
        "CK_HELPER_DIR": ns.CK_HELPER_DIR,
        "AITER_CONFIGS": ns.AITER_CONFIGS,
        "get_gfx": ns.get_gfx,
        "os": os,
        # builtins needed by eval expressions
        "True": True,
        "False": False,
        "None": None,
        "int": int,
        "str": str,
        "hasattr": hasattr,
    }
    # torch is imported lazily inside convert() when the eval string
    # contains "torch".  Pre-populate if available.
    try:
        import torch

        g["torch"] = torch
    except ImportError:
        pass
    return g


# ------------------------------------------------------------------
# build_module loader
# ------------------------------------------------------------------


def load_build_module_fn(aiter_root: str):
    """Return the ``build_module`` callable from ``aiter/jit/core.py``.

    Loads ``core.py`` as an isolated module via importlib so that
    ``aiter/__init__.py`` is never executed.  Module-level side effects
    in ``core.py`` (directory creation via ``get_user_jit_dir()``) are
    acceptable — the caller controls ``AITER_JIT_DIR`` to redirect output.
    """
    jit_dir = os.path.join(aiter_root, "aiter", "jit")
    utils_dir = os.path.join(jit_dir, "utils")
    for d in (utils_dir, jit_dir):
        if d not in sys.path:
            sys.path.insert(0, d)

    spec = importlib.util.spec_from_file_location(
        "_qola_jit_core", os.path.join(jit_dir, "core.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load core.py from {jit_dir}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_module  # type: ignore[attr-defined]
