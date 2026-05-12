# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""On-the-fly AITER checkout management.

AITER is *not* a submodule of QoLA — it is cloned on demand into a
git-ignored directory (``<QoLA repo>/build/third_party/aiter`` by
default).  The manifest's ``[qola] aiter_commit`` (or ``--aiter-commit``)
pins which commit the build runs against, and ``[qola] patches_dir``
(or ``--patches-dir``) carries QoLA-owned patches that are reapplied on
top of that commit on every build.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

# QoLA repo root: <repo>/qola/build_tools/submodule.py -> <repo>
_QOLA_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_AITER_ROOT = _QOLA_ROOT / "build" / "third_party" / "aiter"
_DEFAULT_PATCHES_DIR = _QOLA_ROOT / "patches" / "aiter"
_AITER_REPO_URL = "https://github.com/ROCm/aiter.git"


def default_aiter_root() -> str:
    """Default path for the AITER checkout (``<QoLA repo>/build/third_party/aiter``).

    Git-ignored at the QoLA level. The build system clones into it on first
    use; subsequent builds fetch and check out the requested commit.
    """
    return str(_DEFAULT_AITER_ROOT)


def default_patches_dir() -> str:
    """Default path for QoLA-owned AITER patches (``<QoLA repo>/patches/aiter``)."""
    return str(_DEFAULT_PATCHES_DIR)


def checkout_aiter(
    manifest_path: Optional[str] = None,
    aiter_root: Optional[str] = None,
    aiter_commit: Optional[str] = None,
    patches_dir: Optional[str] = None,
) -> str:
    """Resolve the AITER checkout + patch step from manifest/CLI inputs.

    High-level entry point shared by ``qola checkout`` and downstream
    Python consumers that want a patched AITER tree without running a
    full kernel build.  Applies the same precedence as ``build_kernels``:

    - ``aiter_root``: argument > ``default_aiter_root()``
    - ``aiter_commit``: argument > manifest's ``[qola] aiter_commit`` >
      None (reset to current HEAD; only valid if the checkout exists)
    - ``patches_dir``: argument > manifest's ``[qola] patches_dir`` >
      ``default_patches_dir()``

    Returns the absolute path to the prepared AITER root.

    Raises
    ------
    RuntimeError
        If the checkout does not exist and no commit was specified
        anywhere, or if a patch fails to apply.
    """
    if aiter_root is None:
        aiter_root = default_aiter_root()
    aiter_root = str(Path(aiter_root).resolve())

    qola_section: dict = {}
    if manifest_path is not None:
        with open(manifest_path, "rb") as f:
            qola_section = tomllib.load(f).get("qola", {})

    effective_commit = aiter_commit or qola_section.get("aiter_commit")
    effective_patches_dir = (
        patches_dir
        or qola_section.get("patches_dir")
        or default_patches_dir()
    )

    ensure_aiter_commit(aiter_root, effective_commit, effective_patches_dir)
    return aiter_root


def ensure_aiter_commit(
    aiter_root: str,
    commit: Optional[str],
    patches_dir: Optional[str] = None,
) -> None:
    """Ensure *aiter_root* is a clean checkout of *commit* with patches applied.

    Clones ``ROCm/aiter`` into *aiter_root* if no git tree exists there.
    Then on every call (regardless of current HEAD or working-tree state):

    1. Fetches *commit* from origin if it isn't already present locally.
    2. ``git reset --hard`` to *commit* — wipes any local edits.
    3. ``git submodule update --init --recursive --force`` — resyncs CK
       and any other AITER submodules to the recorded commits.
    4. Applies every ``*.patch`` file in *patches_dir* in lexicographic
       order via ``git apply --3way``. The first failing patch aborts the
       build with a pointer at the offending file.

    Local edits to the AITER checkout never survive a build — patches are
    the only sanctioned way to carry deltas. To skip the patch step,
    point *patches_dir* at an empty directory or ``/dev/null``.

    *commit* may be ``None`` only when the checkout already exists; in
    that case the function still resets to the current HEAD (clearing
    any dirty state) and reapplies patches.
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
        target = _git(aiter_root, "rev-parse", "HEAD").strip()
    else:
        target = _resolve_commit(aiter_root, commit)

    head = _git(aiter_root, "rev-parse", "HEAD").strip()
    if head != target:
        print(f"[QoLA] Checking out AITER {target} (was {head})")
    _git(aiter_root, "reset", "--hard", target)
    _git(aiter_root, "submodule", "update", "--init", "--recursive", "--force")

    _apply_patches(aiter_root, patches_dir)


def _apply_patches(aiter_root: str, patches_dir: Optional[str]) -> None:
    """Apply every ``*.patch`` in *patches_dir* on top of *aiter_root*.

    Patches are applied with ``git apply --3way`` in sorted filename
    order. The first failing patch raises ``RuntimeError`` so the build
    surfaces the breakage instead of running with a half-patched tree.

    No-op when *patches_dir* is ``None``, missing, or contains no
    ``*.patch`` files.
    """
    if patches_dir is None:
        return
    patches_root = Path(patches_dir)
    if not patches_root.is_dir():
        return

    patches = sorted(patches_root.glob("*.patch"))
    if not patches:
        return

    print(f"[QoLA] Applying {len(patches)} patch(es) from {patches_dir}")
    for patch in patches:
        print(f"[QoLA]   - {patch.name}")
        # Resolve to absolute — git apply -C <dir> looks up relative paths
        # under <dir>, not under our CWD.
        patch_abs = str(patch.resolve())
        try:
            _git(aiter_root, "apply", "--3way", "--whitespace=nowarn", patch_abs)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to apply QoLA patch {patch} to AITER at "
                f"{aiter_root!r}.\n"
                f"git apply stderr:\n{exc.stderr}\n"
                f"Either rebase the patch against the current AITER "
                f"commit, or pin the manifest back to a compatible commit."
            ) from exc


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
