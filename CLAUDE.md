# QoLA — Quality of Life AITER

Manifest-driven ahead-of-time builder for AITER kernels. Wraps AITER's `build_module()` JIT system with a declarative TOML manifest and structured output, producing either pybind11 Python modules or torch-free C-linkable shared libraries.

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
    qola_gemm_a4w4_asm.h           Args struct + C API for FP4 ASM GEMM
    qola_gemm_a4w4_asm.cu          Thin entry point — stubs → AITER's asm_gemm_a4w4.cu
    qola_gemm_a4w4_blockscale.h    Args struct + C API for FP4 blockscale CK GEMM
    qola_gemm_a4w4_blockscale.cu   Thin entry point — stubs → AITER's dispatch
    qola_mha_fwd.h                 Namespace wrapper for AITER's mha_fwd
    qola_mha_fwd.cu                Thin entry point — delegates to aiter::mha_fwd()
    qola_mha_bwd.h                 Namespace wrapper for AITER's mha_bwd
    qola_mha_bwd.cu                Thin entry point — delegates to aiter::mha_bwd()
    qola_exports.lds               Linker version script — exports only qola::* symbols
    registry.toml                  Maps module names to cpp_itfs source replacements
    torch_stubs/                   Minimal stub headers shadowing ATen/torch
      ATen/ATen.h                  Stub Tensor, ScalarType, TORCH_CHECK, size_to_dim_
      ATen/hip/HIPContext.h        Stub getCurrentHIPStream (thread-local hipStream_t)
      ATen/hip/impl/...            Empty stubs for transitive includes
      torch/extension.h            Forwards to stub ATen.h
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

Namespace wrappers alone are not sufficient — AITER headers like `mha_fwd.h` declare functions with explicit `__attribute__((visibility("default")))`, which overrides `-fvisibility=hidden` and would leak `aiter::*` symbols into the final `.so`. All cpp_itfs modules **must** be linked with `qola_exports.lds` (`-Wl,--version-script,qola/cpp_itfs/qola_exports.lds`) to force all non-`qola::*` symbols local. The `[defaults]` section in `registry.toml` specifies this version script.

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

### Stub torch headers (`torch_stubs/`)

CK-based modules (like `gemm_a4w4_blockscale`) use AITER's codegen and common headers unmodified. These headers `#include <ATen/ATen.h>` and use `torch::Tensor`, but the actual torch API surface is narrow: `data_ptr()`, `size()`, `stride()`, `dim()`, `sizes()`, `dtype()`, `TORCH_CHECK`, and `at::hip::getCurrentHIPStream()`.

QoLA provides minimal stub headers in `torch_stubs/` that implement this narrow surface. The stubs are placed first in the include path (`-I torch_stubs` before AITER's includes) so they shadow real torch headers. The stub `at::Tensor` carries pre-injected metadata (pointer, sizes, strides, dtype) — it does NOT manage storage, autograd, or device guards.

The entry point (`gemm_a4w4_blockscale.cu`) constructs stub Tensors from the raw-pointer args struct, injects the caller's `hipStream_t` via `at::hip::qola_set_stream()`, and calls AITER's generated kernel functions directly.

This approach eliminates per-module codegen duplication — when AITER updates kernel configs or CK templates, nothing changes on QoLA's side. The stubs cover the "narrow + zero" API tier (~44% of AITER kernels): kernels that only read tensor metadata and optionally zero output buffers.

## Adding a new cpp_itfs module

### Adding a new module (stub approach)

1. Write `qola/cpp_itfs/<name>.h` (args struct + C API). Include `qola_common.h`, use `QOLA_NS_BEGIN`/`QOLA_NS_END`.
2. Write `qola/cpp_itfs/qola_<name>.cu` — the thin entry point (~30 lines). It constructs stub `at::Tensor` objects from the args struct, calls `at::hip::qola_set_stream(stream)`, and invokes AITER's function. Prefix filename with `qola_` to avoid `.o` name collision with AITER's source of the same name. Follow `qola_gemm_a4w4_blockscale.cu` or `qola_gemm_a4w4_asm.cu` as examples.
3. Add a registry entry in `registry.toml`. Only drop the pybind entry point — keep AITER's dispatch `.cu` files. Include `torch_stubs` first in `add_includes`.
4. Verify the module's torch API surface stays within the stub's support (see `torch_stubs/ATen/ATen.h`).
5. Optionally add a Python dispatch wrapper in `qola/dispatch/`.

## Dependencies

- Build time: AITER source tree, ROCm/HIP, hipcc. Torch required for pybind mode only.
- Runtime (pybind): torch, ROCm
- Runtime (cpp_itfs): ROCm only. `.co` ASM blobs must be available via `AITER_ASM_DIR`.

## Submodule

`3rdparty/aiter/` is AITER pinned at the commit specified in the manifest's `[qola] aiter_commit`. Do not modify AITER source here; maintain patches in a QoLA-specific AITER feature branch if needed.
