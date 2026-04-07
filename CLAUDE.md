# QoLA — Quality of Life AITER

Manifest-driven ahead-of-time builder for AITER kernels. Wraps AITER's `build_module()` JIT system with a declarative TOML manifest and structured output, producing either pybind11 Python modules or torch-free C-linkable shared libraries.

## Architecture

```
qola/
  cli.py                          CLI entry point (qola build)
  build/
    __init__.py                    Orchestrator: build_kernels()
    config.py                      TOML manifest parsing, BuildSpec, cpp_itfs source mapping
    resolver.py                    Reconstructs AITER's eval namespace without `import aiter`
    variant_matrix.py              MHA variant expansion from manifest declarations
  cpp_itfs/
    qola_common.h                  QOLA_NS_BEGIN/END/NS() macros for namespace collision prevention
    gemm_a4w4_asm.h                Args struct + C API for FP4 ASM GEMM
    gemm_a4w4_asm.cu               Torch-free ASM GEMM implementation (raw HIP)
    gemm_a4w4_blockscale.h         Args struct + C API for FP4 blockscale CK GEMM
    gemm_a4w4_blockscale.cu        Torch-free CK GEMM dispatch + entry point
    qola_gemm_a4w4_blockscale_common.cuh  Torch-free CK template wrapper (replaces AITER's torch version)
    gen_blockscale_instances.py    Torch-free codegen for CK kernel instances
    registry.toml                  Maps module names to cpp_itfs source replacements
  dispatch/
    gemm_a4w4_asm.py               Python dispatch: ctypes.CDLL + mangled symbol resolution
    gemm_a4w4_blockscale.py        Python dispatch for CK blockscale GEMM
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
- `.so` name prefix: `te_module_gemm_a4w4_asm.so`
- C++ namespace: `qola::te::gemm_a4w4_asm()`
- Compile flag: `-DQOLA_NAMESPACE=te`

Use `QOLA_NS(sym)` macro in C++ to reference the correctly-namespaced symbol.

### Python dispatch (`dispatch/`)

Loads the cpp_itfs `.so` via `ctypes.CDLL(path, RTLD_GLOBAL)` and resolves the C++ symbol by its Itanium ABI mangled name (same pattern as jax-aiter's `ffi/registry.py`). For production TE integration, prefer CMake-time linking into `libtransformer_engine.so` (like `ck_fused_attn` links `libmha_fwd.so`).

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

### CK module codegen (`gen_blockscale_instances.py`)

CK-based modules (like `gemm_a4w4_blockscale`) use compile-time template instantiation instead of runtime `.co` blob loading. AITER's `gen_instances.py` generates per-kernel `.cuh`/`.cpp` files with `torch::Tensor` signatures. QoLA ships its own `gen_blockscale_instances.py` that emits the same CK template instantiations but with raw-pointer function signatures (no torch dependency).

The registry's `blob_gen_cmd` field tells the build system to use QoLA's generator instead of AITER's. It is an eval-able f-string with `{QOLA_ROOT}` and `{AITER_CONFIGS.*}` available.

## Adding a new cpp_itfs module

1. Write `qola/cpp_itfs/<name>.h` (args struct) and `<name>.cu` (implementation). Include `qola_common.h` and use `QOLA_NS_BEGIN`/`QOLA_NS_END` for the namespace.
2. Add an entry to `qola/cpp_itfs/registry.toml` mapping the `optCompilerConfig.json` module name to the new sources.
3. Optionally add a Python dispatch wrapper in `qola/dispatch/`.

## Dependencies

- Build time: AITER source tree, ROCm/HIP, hipcc. Torch required for pybind mode only.
- Runtime (pybind): torch, ROCm
- Runtime (cpp_itfs): ROCm only. `.co` ASM blobs must be available via `AITER_ASM_DIR`.

## Submodule

`3rdparty/aiter/` is AITER pinned at the commit specified in the manifest's `[qola] aiter_commit`. Do not modify AITER source here; maintain patches in a QoLA-specific AITER feature branch if needed.
