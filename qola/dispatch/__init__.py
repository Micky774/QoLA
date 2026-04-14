# SPDX-License-Identifier: MIT
"""Shared utilities for QoLA Python dispatch modules."""

from __future__ import annotations

import os
from pathlib import Path


def find_lib(name: str) -> str:
    """Locate a QoLA-built .so by searching known paths.

    Search order:
    1. ``QOLA_LIB_DIR`` environment variable (highest priority)
    2. ``<qola-package-root>/artifacts/lib/``
    """
    candidates = [
        Path(os.environ.get("QOLA_LIB_DIR", "")) / name,
        Path(__file__).resolve().parent.parent.parent / "artifacts" / "lib" / name,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        f"Could not find {name}. Set QOLA_LIB_DIR to the directory "
        f"containing the QoLA-built .so files."
    )


# Namespace used when building the .so (matches manifest's [qola] namespace).
# Set via QOLA_NAMESPACE env var; empty string means no namespace.
NAMESPACE: str = os.environ.get("QOLA_NAMESPACE", "")


def mangled_sym(func_name: str, args_struct: str, namespace: str = NAMESPACE) -> str:
    """Build the Itanium ABI mangled name for a QoLA cpp_itfs function.

    Handles the optional consumer namespace:
    - No namespace:  ``qola::<func>``  -> ``_ZN4qola...``
    - With namespace: ``qola::<ns>::<func>`` -> ``_ZN4qola<len(ns)><ns>...``

    The function signature is always::

        int <func>(const <qola-ns>::<args_struct>&, ihipStream_t*)

    where ``<qola-ns>`` is ``qola`` or ``qola::<namespace>``.
    """
    # Itanium mangling: _ZN <segments...> E <param-types>
    # Each segment is <length><name>.
    # Return type (int) is not encoded for non-template functions.
    #
    # Parameters:
    #   const <args_struct>& -> RKN<segments>E  (R=ref, K=const, N...E=nested name)
    #   ihipStream_t*        -> P12ihipStream_t
    func_seg = f"{len(func_name)}{func_name}"
    args_seg = f"{len(args_struct)}{args_struct}"

    if namespace:
        ns_seg = f"{len(namespace)}{namespace}"
        # qola::<ns>::<func>(const qola::<ns>::<args>&, ihipStream_t*)
        return f"_ZN4qola{ns_seg}{func_seg}ERKNS0_{args_seg}EP12ihipStream_t"
    else:
        # qola::<func>(const qola::<args>&, ihipStream_t*)
        return f"_ZN4qola{func_seg}ERKNS_{args_seg}EP12ihipStream_t"
