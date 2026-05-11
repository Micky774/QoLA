# QoLA AITER patches

Drop unified-diff `*.patch` files in this directory. On every build,
`qola.build_tools.submodule.ensure_aiter_commit` re-checks out the
manifest's pinned AITER commit, force-syncs submodules (CK lives at
AITER's `3rdparty/composable_kernel/` as a submodule), then applies
each patch in lexicographic filename order via `git apply --3way`.

A failing patch hard-aborts the build — there are no `--reject` files
and no silent skips. When a patch stops applying after an AITER bump,
either rebase the patch against the new tree or pin the manifest back
to a compatible AITER commit.

## Conventions

- **Numbered prefix** so order is explicit: `0001-foo.patch`, `0002-bar.patch`.
- **Generated with** `git format-patch -1 <sha>` from a working AITER
  checkout, or `git diff > NNNN-name.patch` for one-off fixes. Either
  format works — `git apply` accepts both.
- **Paths are relative to the AITER root** (the dir containing
  `aiter/`, `csrc/`, etc.). Patches against CK files therefore start
  with `3rdparty/composable_kernel/...`. Note: `3rdparty/composable_kernel/`
  is an AITER submodule, so `git apply --3way`'s merge fallback can't
  reach into it — clean CK patches apply, stale ones fail outright and
  must be rebased rather than auto-merged.
- **One logical change per patch.** Keeps the conflict surface small
  when AITER moves underneath us.

## Disentanglement intent

The point of this directory is to decouple "which AITER commit do we
build against" from "what fixes does TE need on top of AITER". The
manifest pins the upstream commit; this directory carries the deltas.
Bumping AITER is then a matter of rebasing patches, not maintaining a
parallel AITER fork branch.

## Skipping patches

To temporarily disable a patch, rename it so the suffix is no longer
`.patch` (e.g. `0003-foo.patch.disabled`). To skip the whole directory,
pass `--patches-dir /dev/null` or point `[qola].patches_dir` at an
empty directory.
