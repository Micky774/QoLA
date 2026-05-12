# QoLA — Quality of Life AITER

Manifest-driven ahead-of-time (AOT) builder for [AITER](https://github.com/ROCm/aiter) kernels. QoLA wraps AITER's `build_module()` JIT compilation system with a declarative TOML manifest, producing either:

- **pybind11 Python modules** — standard `.so` files importable from Python (requires PyTorch)
- **torch-free C-linkable shared libraries** (`cpp_itfs` mode) — plain `.so` files linked via HIP/ROCm with no PyTorch dependency

QoLA is designed for [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) to pre-build AITER attention (MHA) kernels at package install time, replacing hours-long JIT compilation with a structured, reproducible build.

## Why QoLA?

- **Declarative manifests** — a single TOML file pins the AITER commit, target architectures, kernel modules, and MHA variant matrix
- **torch-free builds** — `cpp_itfs` mode eliminates the PyTorch build dependency for C-linkable libraries
- **Symbol isolation** — linker version scripts and C++ namespace wrapping prevent symbol collisions when multiple AITER-backed `.so` files coexist in one process
- **Patch-based AITER deltas** — `patches/aiter/*.patch` is reapplied on top of the pinned AITER commit on every build, so bumping AITER becomes a patch rebase rather than maintaining a parallel fork branch
- **No `import aiter`** — QoLA reconstructs AITER's build namespace from a source-tree path alone, avoiding `aiter/__init__.py` side effects and the torch import requirement

## Requirements

- Python >= 3.10
- ROCm / HIP toolchain (hipcc)
- AITER source tree (cloned on demand into `build/third_party/aiter/` — git-ignored, not a submodule; the manifest's `[qola] aiter_commit` pins the SHA)
- PyTorch (pybind mode only)

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Build all modules declared in a manifest (pybind mode).
# AITER is cloned into build/third_party/aiter/ on first use and checked out to the
# manifest's [qola] aiter_commit; --aiter-root is optional.
qola build \
  --manifest example/te-manifest.toml \
  --output-dir /tmp/qola-out

# Build in cpp_itfs mode (no PyTorch dependency)
qola build \
  --manifest example/te-manifest.toml \
  --output-dir /tmp/qola-out \
  --mode cpp_itfs
```

### CLI Options

| Option | Description |
|---|---|
| `--manifest` | Path to the TOML manifest file |
| `--aiter-root` | Path to the AITER source tree (default: `<QoLA repo>/build/third_party/aiter`, cloned on demand) |
| `--aiter-commit` | AITER SHA / tag / branch to fetch and checkout (overrides manifest's `[qola] aiter_commit`) |
| `--patches-dir` | Directory of `*.patch` files applied on top of the AITER checkout (overrides manifest's `[qola] patches_dir`; defaults to `<QoLA repo>/patches/aiter`) |
| `--output-dir` | Directory for build artifacts |
| `--arch` | Target GPU architecture (repeatable, e.g. `--arch gfx950`) |
| `--mode` | Build mode: `pybind` (default) or `cpp_itfs` |
| `--verbose` | Enable verbose build output |

## Manifest Format

The manifest is a TOML file that declares what to build. See [`example/te-manifest.toml`](example/te-manifest.toml) for a full example.

```toml
[qola]
aiter_commit = "33f2e6a..."   # Pinned AITER commit
namespace = "te"               # C++ namespace and .so prefix
patches_dir = "patches/aiter"  # Optional; overrides the default <QoLA repo>/patches/aiter
rocm_versions = ["7.2"]

[build]
architectures = ["gfx950"]

# Static modules from AITER's optCompilerConfig.json
[[modules]]
name = "libmha_fwd"
mode = "cpp_itfs"
receipt = 700                       # CK codegen filter (default: optCompilerConfig value, typically 600)
drop_srcs = ["mha_fwd_split.cu", "mha_fwd_batch_prefill.cu"]
drop_directions = ["fwd_splitkv", "batch_prefill"]

[[modules]]
name = "libmha_bwd"
mode = "cpp_itfs"
receipt = 700

# MHA variant matrix — Cartesian expansion of CK codegen filters
[[mha_fwd_variants]]
dtype = ["bf16", "fp16"]
has_lse = true
has_skip = false

[[mha_bwd_variants]]
dtype = ["bf16", "fp16"]
```

## Build Modes

### pybind (default)

Produces pybind11 `.so` modules importable from Python. Requires PyTorch at both build and runtime.

### cpp_itfs

Produces torch-free C-linkable shared libraries. Each module exposes a C++ API under the configured namespace:

```cpp
#include "qola_mha_fwd.h"

// With namespace = "te":
float ret = qola::te::mha_fwd(args, stream_config);
```

Source replacement is driven by [`cpp_itfs/registry.toml`](qola/cpp_itfs/registry.toml): pybind entry points are swapped for thin C wrappers that expose a namespace-guarded C++ API.

## Build Output

```
output-dir/
  lib/                    # Compiled .so files
    te_libmha_fwd.so
    te_libmha_bwd.so
  configs/                # AITER tuning CSVs
  manifest.json           # Build metadata and per-module results
```

## Available Kernel Modules

| Module | Description | cpp_itfs API |
|---|---|---|
| `libmha_fwd` | Multi-head attention forward | `qola::te::mha_fwd()` |
| `libmha_bwd` | Multi-head attention backward | `qola::te::mha_bwd()` |

## Architecture

### Namespace Resolution

QoLA reconstructs AITER's build-time eval namespace from a source tree path alone, without ever running `import aiter`. This avoids AITER's `__init__.py` side effects and torch import requirements. See [`resolver.py`](qola/build_tools/resolver.py).

### Symbol Collision Prevention

Two layers prevent symbol leaks when multiple `.so` files coexist:

1. **C++ namespace wrapping** — `QOLA_NS_BEGIN`/`QOLA_NS_END` macros place all public symbols under `qola::<namespace>::`
2. **Linker version script** — [`qola_exports.lds`](qola/cpp_itfs/qola_exports.lds) forces all non-`qola::*` symbols local, including AITER symbols with explicit `visibility("default")`

### MHA Variant Matrix

The manifest's `[[mha_fwd_variants]]` / `[[mha_bwd_variants]]` sections declare option dimensions (dtype, has_bias, has_mask, etc.) that are expanded into CK codegen filter patterns. This controls which of the ~34K possible kernel instances are actually compiled. See [`variant_matrix.py`](qola/build_tools/variant_matrix.py). This is currently only support for pybind11 output.

For `cpp_itfs` static modules, kernel pruning is instead controlled by the `receipt` manifest field, which overrides the `--receipt N` argument passed to CK's `generate.py`. AITER's `optCompilerConfig.json` defaults to receipt 600 (the generic `aiter::mha_*` C++ API filter); setting `receipt = 700` selects the TransformerEngine-specific filter (fp16/bf16, row vlayout, has_lse, no skip/sink/logits/qscale), shrinking forward codegen by roughly 48×. See `_rewrite_receipt` in [`config.py`](qola/build_tools/config.py) for the implementation.

### HSA Blob Embedding

[`generate_embedded_hsa.py`](qola/build_tools/generate_embedded_hsa.py) converts binary `.co` ASM blobs into a C++ header with compile-time byte arrays, enabling kernel distribution without a runtime `AITER_ASM_DIR`.

## Roadmap

- [ ] CI support for building and publishing pre-built libraries from manifests
- [ ] Kernel filtering for `libmha` — prune CK codegen instances based on manifest variant declarations in `cpp_itfs` mode (currently pybind-only)
- [ ] C-level JIT for `libmha` — compile MHA variant `.so` files on first use at the C layer, avoiding ahead-of-time compilation of the full variant matrix

## License

MIT License. See `LICENSE` for more details.
