# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""On-the-fly AITER checkout management.

AITER is *not* a submodule of QoLA — it is cloned on demand into a
git-ignored directory (``<QoLA repo>/3rdparty/aiter`` by default).  The
manifest's ``[qola] aiter_commit`` (or ``--aiter-commit``) pins which
commit the build runs against.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

# QoLA repo root: <repo>/qola/build_tools/submodule.py -> <repo>
_QOLA_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_AITER_ROOT = _QOLA_ROOT / "3rdparty" / "aiter"
_AITER_REPO_URL = "https://github.com/ROCm/aiter.git"


def default_aiter_root() -> str:
    """Default path for the AITER checkout (``<QoLA repo>/3rdparty/aiter``).

    Git-ignored at the QoLA level. The build system clones into it on first
    use; subsequent builds fetch and check out the requested commit.
    """
    return str(_DEFAULT_AITER_ROOT)


def ensure_aiter_commit(aiter_root: str, commit: Optional[str]) -> None:
    """Ensure *aiter_root* is a git checkout at *commit*.

    Clones ``ROCm/aiter`` into *aiter_root* if no git tree exists there.
    Once a checkout is present, fetches and checks out *commit* using the
    same dirty-tree policy as before: if the tree's HEAD already matches
    *commit*, the working copy is left alone (a dirty tree is permitted in
    that case); a dirty tree combined with a different requested commit is
    an error.

    No-op when *commit* is ``None`` and the checkout already exists —
    builds against whatever is currently checked out.  When *commit* is
    ``None`` and the checkout is missing, raises (we don't know what to
    clone to).
    """
    root = Path(aiter_root)
    is_checkout = (root / ".git").exists()

    if not is_checkout:
        if commit is None:
            raise RuntimeError(
                f"AITER checkout at {aiter_root!r} does not exist and no "
                f"commit was specified. Set [qola] aiter_commit in the "
                f"manifest or pass --aiter-commit so QoLA can clone."
            )
        _clone(aiter_root)

    if commit is None:
        return

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


def _clone(aiter_root: str) -> None:
    """Partial-clone ``ROCm/aiter`` into *aiter_root*."""
    root = Path(aiter_root)
    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"[QoLA] Cloning {_AITER_REPO_URL} -> {aiter_root}")
    subprocess.run(
        ["git", "clone", "--filter=blob:none", _AITER_REPO_URL, str(root)],
        check=True,
    )


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
