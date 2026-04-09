# SPDX-License-Identifier: MIT
"""Expand ``[[mha_variants]]`` manifest declarations into BuildSpec instances."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from .config import BuildSpec, _eval_entry
from .resolver import AiterNamespace, make_eval_globals


def expand_mha_variants(
    mha_section: List[Dict[str, Any]],
    ns: AiterNamespace,
) -> List[BuildSpec]:
    """Produce one :class:`BuildSpec` per declared MHA variant.

    Each entry in *mha_section* maps to a call to
    ``compose_mha_fwd_variant_suffix_and_filter()`` from AITER's
    ``mha_recipes.py``.  The srcs and compiler flags are inherited from
    the base ``module_mha_varlen_fwd`` entry in ``optCompilerConfig.json``
    (same pattern as AITER's ``setup.py``).

    Parameters
    ----------
    mha_section
        List of dicts from the TOML ``[[mha_variants]]`` array.
    ns
        Resolved AITER namespace.
    """
    utils_dir = os.path.join(ns.AITER_ROOT_DIR, "aiter", "jit", "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)

    import mha_recipes as _mha_recipes  # type: ignore[import-not-found]

    # Load base build args from module_mha_varlen_fwd
    config_json = os.path.join(
        ns.AITER_ROOT_DIR, "aiter", "jit", "optCompilerConfig.json"
    )
    with open(config_json, "r") as f:
        all_entries: dict = json.load(f)

    base_raw = dict(all_entries.get("module_mha_varlen_fwd", {}))
    eval_globals = make_eval_globals(ns)
    base = _eval_entry("module_mha_varlen_fwd", base_raw, eval_globals)

    results: List[BuildSpec] = []
    for decl in mha_section:
        receipt = int(decl.get("receipt", 200))

        suffix, filter_pattern = _mha_recipes.compose_mha_fwd_variant_suffix_and_filter(
            dtype=str(decl["dtype"]),
            logits_positive=bool(decl.get("logits_positive", False)),
            has_bias=bool(decl.get("has_bias", False)),
            has_alibi=bool(decl.get("has_alibi", False)),
            use_mask=bool(decl.get("use_mask", False)),
            return_lse=bool(decl.get("return_lse", False)),
            dropout_zero=bool(decl.get("dropout_zero", True)),
            skip_zero=bool(decl.get("skip_zero", True)),
            has_qscale=bool(decl.get("has_qscale", True)),
        )
        md_name = f"mha_varlen_fwd{suffix}"

        gen_py = os.path.join(
            ns.CK_DIR, "example", "ck_tile", "01_fmha", "generate.py"
        )
        blob_gen_cmd = [
            f"{gen_py} -d fwd --receipt {receipt}"
            f" --filter {filter_pattern} --output_dir {{}}",
            f"{gen_py} -d fwd_splitkv --receipt {receipt}"
            f' --filter " @ " --output_dir {{}}',
        ]

        results.append(
            BuildSpec(
                md_name=md_name,
                srcs=list(base.srcs),
                flags_extra_cc=list(base.flags_extra_cc),
                flags_extra_hip=list(base.flags_extra_hip),
                blob_gen_cmd=blob_gen_cmd,
                extra_include=list(base.extra_include),
                extra_ldflags=base.extra_ldflags,
                verbose=base.verbose,
                is_python_module=True,
                is_standalone=False,
                torch_exclude=False,
                hipify=base.hipify,
                hip_clang_path=base.hip_clang_path,
            )
        )

    return results
