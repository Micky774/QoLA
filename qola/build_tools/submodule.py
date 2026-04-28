# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER submodule checkout management."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

# QoLA repo root: <repo>/qola/build_tools/submodule.py -> <repo>
_QOLA_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_AITER_ROOT = _QOLA_ROOT / "3rdparty" / "aiter"


def default_aiter_root() -> str:
    """Path to the bundled AITER submodule (``<QoLA repo>/3rdparty/aiter``)."""
    return str(_DEFAULT_AITER_ROOT)


def ensure_aiter_commit(aiter_root: str, commit: Optional[str]) -> None:
    """Fetch and checkout *commit* in the AITER tree at *aiter_root*.

    No-op when *commit* is ``None`` — preserves the legacy behavior of
    building against whatever the caller has already checked out.

    When *commit* is given, the tree's HEAD is forced to that SHA.  If the
    tree's HEAD already matches, the working copy is left alone (a dirty
    tree is permitted in that case).  A dirty tree combined with a
    different requested commit is an error — we won't silently discard
    work.
    """
    if commit is None:
        return

    root = Path(aiter_root)
    if not (root / ".git").exists():
        raise RuntimeError(
            f"--aiter-root {aiter_root!r} is not a git checkout. "
            f"Run `git submodule update --init 3rdparty/aiter` from the "
            f"QoLA repo root, or pass --aiter-root pointing at a real "
            f"AITER git tree."
        )

    target = _resolve_commit(aiter_root, commit)
    head = _git(aiter_root, "rev-parse", "HEAD").strip()

    if head == target:
        return

    porcelain = _git(aiter_root, "status", "--porcelain").strip()
    if porcelain:
        raise RuntimeError(
            f"AITER tree at {aiter_root!r} is dirty; refusing to checkout "
            f"{target} over local changes:\n{porcelain}\n"
            f"Either commit/stash/discard the changes, or pin the manifest "
            f"to the currently-checked-out commit ({head})."
        )

    print(f"[QoLA] Checking out AITER {target} (was {head})")
    _git(aiter_root, "checkout", "--detach", target)
    _git(aiter_root, "submodule", "update", "--init", "--recursive")


def _resolve_commit(aiter_root: str, commit: str) -> str:
    """Return the full SHA for *commit*, fetching from origin if necessary."""
    try:
        return _git(aiter_root, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    except subprocess.CalledProcessError:
        pass

    # Commit not present locally — try a targeted fetch first, then a full
    # fetch as a fallback (some servers reject arbitrary-SHA fetches without
    # uploadpack.allowAnySHA1InWant).
    try:
        _git(aiter_root, "fetch", "origin", commit)
    except subprocess.CalledProcessError:
        _git(aiter_root, "fetch", "--tags", "origin")

    try:
        return _git(aiter_root, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"AITER commit {commit!r} not found in {aiter_root!r} even after "
            f"fetching from origin. Check the manifest's [qola] aiter_commit "
            f"or --aiter-commit value."
        ) from exc


def _git(cwd: str, *args: str) -> str:
    """Run ``git <args>`` inside *cwd*, returning stdout. Raises on non-zero exit."""
    result = subprocess.run(
        ["git", "-C", cwd, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout
