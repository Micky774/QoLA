# SPDX-License-Identifier: MIT
"""Expand MHA variant manifest declarations into BuildSpec instances.

Supports ``[[mha_fwd_variants]]``, ``[[mha_bwd_variants]]``, and the
legacy ``[[mha_variants]]`` shorthand (fwd-only).

Each entry supports cartesian-product expansion: options may be scalars,
lists, or omitted entirely.  Omitted options expand to **all** available
values.  Duplicates (e.g. from the ``has_bias``/``has_alibi`` interaction
in fwd) are deduplicated by filter pattern.

Forward and backward directions have **different** variant dimensions
because the CK codegen instance naming differs between the two.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Any, Dict, List, Tuple

from .config import BuildSpec, _eval_entry
from .resolver import AiterNamespace, make_eval_globals


# ---------------------------------------------------------------------------
# Forward variant option space (libmha_fwd / receipt 600)
# ---------------------------------------------------------------------------
# CK fwd instance names at receipt 600 use this variant suffix:
#   _{logits}_{bias}_{mask}_{lse}_{dropout}_{skip}_{qscale}_{trload}_{sink}
#
# Notable differences from AITER's compose_mha_fwd_variant_suffix_and_filter
# (which targets receipt 200 pybind modules):
#   - _logits is always present (never _nlogits)
#   - qscale uses _nqscale / _pertensor / _blockscale (not inverted)
#   - _sink / _nsink is a new dimension
#   - _trload is always _ntrload
_FWD_OPTION_SPACE: Dict[str, list] = {
    "dtype": ["bf16", "fp16"],
    "has_bias": [False, True],
    "has_alibi": [False, True],
    "has_mask": [False, True],
    "has_lse": [False, True],
    "has_dropout": [False, True],
    "has_skip": [False, True],
    "has_sink": [False, True],
}

_FWD_OPTION_KEYS = list(_FWD_OPTION_SPACE.keys())

# ---------------------------------------------------------------------------
# Backward variant option space
# ---------------------------------------------------------------------------
# CK bwd instance names use a different scheme:
#   fmha_bwd_d{hdim}_{dtype}_{mode}_{tile}_{pad}_{bias}_{dbias}
#   _{mask}_{dropout}_{deterministic}_{trload}
#
# The fnmatch filter matches against instance names, so the filter tokens
# must align with the bwd naming convention (see fmha_bwd.py in CK).
_BWD_OPTION_SPACE: Dict[str, list] = {
    "dtype": ["bf16", "fp16"],
    "has_bias": [False, True],
    "has_alibi": [False, True],
    "has_dbias": [False, True],
    "has_mask": [False, True],
    "has_dropout": [False, True],
    "deterministic": [False, True],
}

_BWD_OPTION_KEYS = list(_BWD_OPTION_SPACE.keys())


# ---------------------------------------------------------------------------
# Forward expansion (QoLA-native, matches receipt 600 instance naming)
# ---------------------------------------------------------------------------

def _compose_fwd_filter(
    dtype: str,
    has_bias: bool,
    has_alibi: bool,
    has_mask: bool,
    has_lse: bool,
    has_dropout: bool,
    has_skip: bool,
    has_sink: bool,
) -> Tuple[str, str]:
    """Build an fnmatch filter for CK fwd instance names (receipt 600).

    CK fwd instance names at receipt 600::

        fmha_fwd_d{hdim}_{dtype}_{mode}_{tile}_{pipeline}_{pad}
        _logits_{bias}_{mask}_{lse}_{dropout}_{skip}_{qscale}
        _{trload}_{sink}_{arch}.cpp

    ``_logits`` is always present.  We wildcard hdim/mode/tile/pipeline/
    pad/qscale/trload/arch and filter on variant-specific tokens.
    """
    if has_bias:
        bias_token = "_bias"
    elif has_alibi:
        bias_token = "_alibi"
    else:
        bias_token = "_nbias"

    mask_token = "_mask" if has_mask else "_nmask"
    lse_token = "_lse" if has_lse else "_nlse"
    dropout_token = "_dropout" if has_dropout else "_ndropout"
    skip_token = "_skip" if has_skip else "_nskip"
    sink_token = "_sink" if has_sink else "_nsink"

    # Wildcard everything before the variant tokens and after them
    # (qscale, trload, arch are not variant-filtered).
    filt = (
        f"*{dtype}*"
        f"{bias_token}"
        f"{mask_token}"
        f"{lse_token}"
        f"{dropout_token}"
        f"{skip_token}*"
        f"{sink_token}*"
    )

    suffix = (
        f"_{dtype}"
        + bias_token
        + mask_token
        + lse_token
        + dropout_token
        + skip_token
        + sink_token
    )

    return suffix, filt


def _expand_fwd_variants(
    mha_section: List[Dict[str, Any]],
    ns: AiterNamespace,
) -> List[Dict[str, Any]]:
    """Expand fwd variant declarations into deduplicated variant dicts.

    Returns a list of ``{"suffix": ..., "filter": ..., "receipt": ...}``
    dicts.
    """
    seen_filters: set = set()
    results: List[Dict[str, Any]] = []

    for decl in mha_section:
        receipt = int(decl.get("receipt", 600))

        option_lists: List[list] = []
        for key in _FWD_OPTION_KEYS:
            val = decl.get(key)
            if val is None:
                option_lists.append(_FWD_OPTION_SPACE[key])
            elif isinstance(val, list):
                option_lists.append(val)
            else:
                option_lists.append([val])

        for combo in itertools.product(*option_lists):
            kwargs = dict(zip(_FWD_OPTION_KEYS, combo))
            kwargs["dtype"] = str(kwargs["dtype"])
            for k in _FWD_OPTION_KEYS:
                if k != "dtype":
                    kwargs[k] = bool(kwargs[k])

            suffix, filter_pattern = _compose_fwd_filter(**kwargs)

            if filter_pattern in seen_filters:
                continue
            seen_filters.add(filter_pattern)
            results.append({
                "suffix": suffix,
                "filter": filter_pattern,
                "receipt": receipt,
            })
    return results


# ---------------------------------------------------------------------------
# Backward expansion (QoLA-native, no AITER helper exists for bwd)
# ---------------------------------------------------------------------------

def _compose_bwd_filter(
    dtype: str,
    has_bias: bool,
    has_alibi: bool,
    has_dbias: bool,
    has_mask: bool,
    has_dropout: bool,
    deterministic: bool,
) -> Tuple[str, str]:
    """Build a 3-part fnmatch filter for CK bwd codegen.

    CK's ``generate.py -d bwd --filter`` expects a 3-part filter
    separated by ``@``::

        dot_do_o_filter @ convert_dq_filter @ dq_dk_dv_filter

    Each part is matched (via ``fnmatch``) against the ``.name`` of
    the corresponding sub-kernel class:

    - **dot_do_o**: ``fmha_bwd_dot_do_o_d{hdim}_{dtype}_b{bm0}_{mode}``
      — only dtype varies across our options.
    - **convert_dq**: ``..._d{hdim}_{dtype}_b{bm0}x{bn0}_{mode}_..._{det}``
      — dtype and deterministic vary.
    - **dq_dk_dv**: ``fmha_bwd_d{hdim}_{dtype}_{mode}_{tile}_{pad}``
      ``_{bias}_{dbias}_{mask}_{dropout}_{det}_{trload}``
      — all variant options appear here.

    We use ``*`` wildcards to skip hdim/mode/tile/pad/trload (which are
    tile-size parameters, not variant parameters).
    """
    if has_bias:
        bias_token = "_bias"
    elif has_alibi:
        bias_token = "_alibi"
    else:
        bias_token = "_nbias"

    dbias_token = "_dbias" if has_dbias else "_ndbias"
    mask_token = "_mask" if has_mask else "_nmask"
    # CK bwd dropout: "no" → "_ndropout", others → "_dropout_wg16" etc.
    dropout_token = "_dropout*" if has_dropout else "_ndropout"
    det_token = "_deterministic" if deterministic else "_ndeterministic"

    # dq_dk_dv filter — variant tokens with wildcards for tile params.
    dq_filter = (
        f"*{dtype}*"
        f"{bias_token}"
        f"{dbias_token}"
        f"{mask_token}"
        f"*{dropout_token}"
        f"{det_token}*"
    )

    # dot_do_o filter — only dtype matters.
    dot_filter = f"*{dtype}*"

    # convert_dq filter — dtype and deterministic.
    convert_filter = f"*{dtype}*{det_token}"

    filter_pattern = f"{dot_filter}@{convert_filter}@{dq_filter}"

    suffix = (
        f"_{dtype}"
        + bias_token
        + dbias_token
        + ("_mask" if has_mask else "_nmask")
        + ("_dropout" if has_dropout else "_ndropout")
        + ("_deterministic" if deterministic else "_ndeterministic")
    )

    return suffix, filter_pattern


def _expand_bwd_variants(
    mha_section: List[Dict[str, Any]],
    ns: AiterNamespace,
) -> List[Dict[str, Any]]:
    """Expand bwd variant declarations into deduplicated variant dicts.

    Returns a list of ``{"suffix": ..., "filter": ..., "receipt": ...}``
    dicts.
    """
    seen_filters: set = set()
    results: List[Dict[str, Any]] = []

    for decl in mha_section:
        receipt = int(decl.get("receipt", 600))

        option_lists: List[list] = []
        for key in _BWD_OPTION_KEYS:
            val = decl.get(key)
            if val is None:
                option_lists.append(_BWD_OPTION_SPACE[key])
            elif isinstance(val, list):
                option_lists.append(val)
            else:
                option_lists.append([val])

        for combo in itertools.product(*option_lists):
            kwargs = dict(zip(_BWD_OPTION_KEYS, combo))
            kwargs["dtype"] = str(kwargs["dtype"])
            for k in _BWD_OPTION_KEYS:
                if k != "dtype":
                    kwargs[k] = bool(kwargs[k])

            suffix, filter_pattern = _compose_bwd_filter(**kwargs)

            if filter_pattern in seen_filters:
                continue
            seen_filters.add(filter_pattern)
            results.append({
                "suffix": suffix,
                "filter": filter_pattern,
                "receipt": receipt,
            })
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_mha_fwd_filters(
    mha_section: List[Dict[str, Any]],
    ns: AiterNamespace,
) -> List[Dict[str, Any]]:
    """Compute CK codegen filter patterns for fwd from variant declarations.

    Returns a list of dicts with ``suffix``, ``filter``, and ``receipt``
    keys.
    """
    return _expand_fwd_variants(mha_section, ns)


def compute_mha_bwd_filters(
    mha_section: List[Dict[str, Any]],
    ns: AiterNamespace,
) -> List[Dict[str, Any]]:
    """Compute CK codegen filter patterns for bwd from variant declarations.

    Returns a list of dicts with ``suffix``, ``filter``, and ``receipt``
    keys.
    """
    return _expand_bwd_variants(mha_section, ns)


def expand_mha_variants(
    mha_section: List[Dict[str, Any]],
    ns: AiterNamespace,
) -> List[BuildSpec]:
    """Produce one :class:`BuildSpec` per declared fwd MHA variant.

    Each entry in *mha_section* is expanded via cartesian product and
    deduplicated.  The srcs and compiler flags are inherited from the
    base ``module_mha_varlen_fwd`` entry in ``optCompilerConfig.json``
    (same pattern as AITER's ``setup.py``).

    Parameters
    ----------
    mha_section
        List of dicts from the TOML ``[[mha_fwd_variants]]`` array.
    ns
        Resolved AITER namespace.
    """
    variants = _expand_fwd_variants(mha_section, ns)

    # Load base build args from module_mha_varlen_fwd
    config_json = os.path.join(
        ns.AITER_ROOT_DIR, "aiter", "jit", "optCompilerConfig.json"
    )
    with open(config_json, "r") as f:
        all_entries: dict = json.load(f)

    base_raw = dict(all_entries.get("module_mha_varlen_fwd", {}))
    eval_globals = make_eval_globals(ns)
    base = _eval_entry("module_mha_varlen_fwd", base_raw, eval_globals)

    gen_py = os.path.join(
        ns.CK_DIR, "example", "ck_tile", "01_fmha", "generate.py"
    )

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
