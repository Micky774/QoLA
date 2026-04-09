# SPDX-License-Identifier: MIT
"""Expand ``[[mha_variants]]`` manifest declarations into BuildSpec instances.

Each ``[[mha_variants]]`` entry supports cartesian-product expansion:
options may be scalars, lists, or omitted entirely.  Omitted options
expand to **all** available values.  Duplicates (e.g. from the
``has_bias``/``has_alibi`` interaction) are deduplicated by filter
pattern.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Any, Dict, List

from .config import BuildSpec, _eval_entry
from .resolver import AiterNamespace, make_eval_globals

# Full value space for each MHA variant option.  When an option is
# omitted from a ``[[mha_variants]]`` entry, all values listed here
# are included in the cartesian product.
_MHA_OPTION_SPACE: Dict[str, list] = {
    k: [False, True]
    for k in (
        "logits_positive",
        "has_bias",
        "has_alibi",
        "use_mask",
        "return_lse",
        "dropout_zero",
        "skip_zero",
        "has_qscale",
    )
}
_MHA_OPTION_SPACE |= {"dtype": ["bf16", "fp16"]}

_MHA_OPTION_KEYS = list(_MHA_OPTION_SPACE.keys())


def _ensure_mha_recipes(ns: AiterNamespace):
    """Import AITER's ``mha_recipes`` module (idempotent)."""
    utils_dir = os.path.join(ns.AITER_ROOT_DIR, "aiter", "jit", "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)

    import mha_recipes as _mha_recipes  # type: ignore[import-not-found]

    return _mha_recipes


def _expand_variants_cartesian(
    mha_section: List[Dict[str, Any]],
    ns: AiterNamespace,
) -> List[Dict[str, Any]]:
    """Expand ``[[mha_variants]]`` entries into deduplicated variant dicts.

    Each entry may specify scalars, lists, or omit options entirely.
    Omitted options expand to all values in :data:`_MHA_OPTION_SPACE`.
    The cartesian product is computed per entry, then all entries are
    merged and deduplicated by filter pattern.

    Returns a list of ``{"suffix": ..., "filter": ..., "receipt": ...}``
    dicts.
    """
    mha_recipes = _ensure_mha_recipes(ns)

    seen_filters: set = set()
    results: List[Dict[str, Any]] = []

    for decl in mha_section:
        receipt = int(decl.get("receipt", 200))

        # Normalize each option to a list; missing → full value space.
        option_lists: List[list] = []
        for key in _MHA_OPTION_KEYS:
            val = decl.get(key)
            if val is None:
                option_lists.append(_MHA_OPTION_SPACE[key])
            elif isinstance(val, list):
                option_lists.append(val)
            else:
                option_lists.append([val])

        for combo in itertools.product(*option_lists):
            kwargs = dict(zip(_MHA_OPTION_KEYS, combo))
            kwargs["dtype"] = str(kwargs["dtype"])
            for k in _MHA_OPTION_KEYS:
                if k != "dtype":
                    kwargs[k] = bool(kwargs[k])

            suffix, filter_pattern = (
                mha_recipes.compose_mha_fwd_variant_suffix_and_filter(**kwargs)
            )

            if filter_pattern in seen_filters:
                continue
            seen_filters.add(filter_pattern)
            results.append(
                {
                    "suffix": suffix,
                    "filter": filter_pattern,
                    "receipt": receipt,
                }
            )
    return results


def compute_mha_variant_filters(
    mha_section: List[Dict[str, Any]],
    ns: AiterNamespace,
) -> List[Dict[str, Any]]:
    """Compute CK codegen filter patterns from ``[[mha_variants]]`` declarations.

    Returns a list of dicts, each with ``suffix``, ``filter``, and ``receipt``
    keys.  Used by :func:`_apply_mha_variant_filter` in ``config.py`` to
    replace the unfiltered ``blob_gen_cmd`` in ``libmha_fwd``/``libmha_bwd``
    with filtered versions.
    """
    return _expand_variants_cartesian(mha_section, ns)


def expand_mha_variants(
    mha_section: List[Dict[str, Any]],
    ns: AiterNamespace,
) -> List[BuildSpec]:
    """Produce one :class:`BuildSpec` per declared MHA variant.

    Each entry in *mha_section* is expanded via cartesian product and
    deduplicated.  The srcs and compiler flags are inherited from the
    base ``module_mha_varlen_fwd`` entry in ``optCompilerConfig.json``
    (same pattern as AITER's ``setup.py``).

    Parameters
    ----------
    mha_section
        List of dicts from the TOML ``[[mha_variants]]`` array.
    ns
        Resolved AITER namespace.
    """
    variants = _expand_variants_cartesian(mha_section, ns)

    # Load base build args from module_mha_varlen_fwd
    config_json = os.path.join(
        ns.AITER_ROOT_DIR, "aiter", "jit", "optCompilerConfig.json"
    )
    with open(config_json, "r") as f:
        all_entries: dict = json.load(f)

    base_raw = dict(all_entries.get("module_mha_varlen_fwd", {}))
    eval_globals = make_eval_globals(ns)
    base = _eval_entry("module_mha_varlen_fwd", base_raw, eval_globals)

    gen_py = os.path.join(ns.CK_DIR, "example", "ck_tile", "01_fmha", "generate.py")

    results: List[BuildSpec] = []
    for vf in variants:
        md_name = f"mha_varlen_fwd{vf['suffix']}"
        receipt = vf["receipt"]
        filter_pattern = vf["filter"]

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
