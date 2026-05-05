# QoLA — Quality of Life AITER

Manifest-driven ahead-of-time builder for AITER MHA kernels. Wraps AITER's `build_module()` JIT system with a declarative TOML manifest and structured output, producing torch-free C-linkable shared libraries (`libmha_fwd.so`, `libmha_bwd.so`).

## Architecture

```
qola/
  cli.py                          CLI entry point (qola build)
  build_tools/
    __init__.py                    Orchestrator: build_kernels()
    config.py                      TOML manifest parsing, BuildSpec, cpp_itfs source mapping
    resolver.py                    Reconstructs AITER's eval namespace without `import aiter`
    variant_matrix.py              MHA variant expansion from manifest declarations
  cpp_itfs/
    qola_common.h                  QOLA_NS_BEGIN/END/NS() macros for namespace collision prevention
    qola_mha_fwd.h                 Namespace wrapper for AITER's mha_fwd
    qola_mha_fwd.cu                Thin entry point — delegates to aiter::mha_fwd()
    qola_mha_bwd.h                 Namespace wrapper for AITER's mha_bwd
    qola_mha_bwd.cu                Thin entry point — delegates to aiter::mha_bwd()
    qola_exports.lds               Linker version script — exports only qola::* symbols
    registry.toml                  Maps module names to cpp_itfs source replacements
```

## Build modes

- **`pybind`** (default): `is_python_module=True, torch_exclude=False`. Produces standard pybind11 `.so` importable from Python. Requires torch at build and runtime.
- **`cpp_itfs`**: `is_python_module=False, torch_exclude=True`. Produces a plain C-linkable `.so` with no torch dependency. Requires only HIP/ROCm. Source replacement is driven by `cpp_itfs/registry.toml`.

## Configuration precedence (CLI vs manifest)

CLI flags win over manifest globals; per-module entries (most specific scope) still win over the CLI.

| Setting          | Highest → lowest precedence                                                              |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Build mode       | `[[modules]].mode` → CLI `--mode` → `[build].mode` → `"pybind"` default                  |
| GPU architectures| CLI `--arch` (repeatable) → `[build].architectures` → `$GPU_ARCHS` → `"native"`          |
| AITER commit     | CLI `--aiter-commit` → `[qola].aiter_commit` → currently checked-out HEAD                |

A flag is "set" only when explicitly passed; argparse defaults to `None` for `--mode` so it cannot accidentally override a manifest value. Per-module overrides are an opt-out from a CLI-wide blanket: `qola build --mode cpp_itfs` builds everything in `cpp_itfs` *except* modules that pin `mode = "pybind"` in `[[modules]]`.

## Key concepts

### Namespace resolution (`resolver.py`)

AITER's `optCompilerConfig.json` entries contain `eval()`-able f-strings referencing module-level globals from `core.py` (e.g. `AITER_CSRC_DIR`, `CK_DIR`, `AITER_CONFIGS`). QoLA reconstructs this eval namespace from just an AITER source tree path by:
1. Deriving path constants from the AITER root
2. `exec()`-ing the `# config_env start here`/`end here` block from `core.py` to get `AITER_CONFIGS`
3. Importing `get_gfx` from `chip_info.py` via `sys.path` injection

### cpp_itfs pattern

Each cpp_itfs wrapper replaces the pybind entry point + torch interface with a raw-pointer args struct and a function taking `hipStream_t`. The caller owns all device memory and stream lifecycle. This mirrors AITER's own `csrc/cpp_itfs/` pattern (`libmha_fwd`, `libmha_bwd`).

### Symbol collision prevention

The manifest's `[qola] namespace = "te"` causes:
- `.so` name prefix: `te_libmha_fwd.so`
- C++ namespace: `qola::te::mha_fwd()`
- Compile flag: `-DQOLA_NAMESPACE=te`

Use `QOLA_NS(sym)` macro in C++ to reference the correctly-namespaced symbol.

Namespace wrappers alone are not sufficient — AITER headers like `mha_fwd.h` declare functions with explicit `__attribute__((visibility("default")))`, which overrides `-fvisibility=hidden` and would leak `aiter::*` symbols into the final `.so`. All cpp_itfs modules **must** be linked with `qola_exports.lds` (`-Wl,--version-script,qola/cpp_itfs/qola_exports.lds`) to force all non-`qola::*` symbols local. The `[defaults]` section in `registry.toml` specifies this version script.

## Running builds

All builds and Python execution must happen inside the docker container (see parent repo CLAUDE.md). The host is for file reads, searches, and git only.

`--aiter-root` is optional and defaults to `<QoLA repo>/3rdparty/aiter` — a git-ignored directory that QoLA clones `https://github.com/ROCm/aiter.git` into on first use. On every build the system fetches and checks out the AITER commit resolved per the precedence table above; pass `--aiter-commit <sha>` to override the manifest's `[qola] aiter_commit`.

```bash
# pybind mode (default), clones/updates AITER per the manifest commit
docker exec <container> python -m qola.cli build \
  --manifest example/manifest.toml \
  --output-dir /tmp/qola-out

# cpp_itfs mode against an arbitrary AITER commit
docker exec <container> python -m qola.cli build \
  --manifest example/manifest.toml \
  --output-dir /tmp/qola-out \
  --aiter-commit <sha-or-branch> \
  --mode cpp_itfs
```

## Dependencies

- Build time: AITER source tree, ROCm/HIP, hipcc. Torch required for pybind mode only.
- Runtime (pybind): torch, ROCm
- Runtime (cpp_itfs): ROCm only. ASM blobs are baked into each `.so` at build time via `_generate_embedded_hsa`; there is no `AITER_ASM_DIR` filesystem fallback.

## AITER checkout

AITER is **not** a submodule of this repo — it is cloned on demand into `3rdparty/aiter/` (git-ignored) by `qola.build_tools.submodule.ensure_aiter_commit`. Consumer manifests (`[qola] aiter_commit`, or `--aiter-commit`) pin the SHA on every build. Any QoLA commit can target any AITER commit by editing the manifest; nothing in this repo references a fixed AITER SHA.

If `3rdparty/aiter/` does not yet exist, the first build clones `https://github.com/ROCm/aiter.git` into it (partial clone, `--filter=blob:none`) and checks out the manifest's commit. Subsequent builds reuse the same checkout, fetching new commits as needed. Delete the directory to force a fresh clone.

Dirty-tree policy: when the requested commit equals the current HEAD, the working copy is left alone (local hacks survive). When a *different* commit is requested and the tree is dirty, the build aborts rather than discarding work — commit, stash, or pin the manifest to the current HEAD to proceed. Do not modify AITER source here for permanent changes; maintain patches in a QoLA-specific AITER feature branch.
