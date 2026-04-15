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

Mode precedence: CLI `--mode` < manifest `[build] mode` < per-module `mode`.

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

```bash
# pybind mode (default)
docker exec <container> python -m qola.cli build \
  --manifest example/manifest.toml \
  --aiter-root 3rdparty/aiter \
  --output-dir /tmp/qola-out

# cpp_itfs mode
docker exec <container> python -m qola.cli build \
  --manifest example/manifest.toml \
  --aiter-root 3rdparty/aiter \
  --output-dir /tmp/qola-out \
  --mode cpp_itfs
```

## Dependencies

- Build time: AITER source tree, ROCm/HIP, hipcc. Torch required for pybind mode only.
- Runtime (pybind): torch, ROCm
- Runtime (cpp_itfs): ROCm only. `.co` ASM blobs must be available via `AITER_ASM_DIR`.

## Submodule

`3rdparty/aiter/` is AITER pinned at the commit specified in the manifest's `[qola] aiter_commit`. Do not modify AITER source here; maintain patches in a QoLA-specific AITER feature branch if needed.
