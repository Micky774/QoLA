# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Torch-free codegen for QoLA's cpp_itfs gemm_a4w4_blockscale variant.

Generates the same CK template instantiation files as AITER's
gen_instances.py, but with raw-pointer function signatures instead of
torch::Tensor.  Reuses AITER's kernel config definitions from
gemm_a4w4_blockscale_common.py.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Import kernel configs from AITER's codegen directory.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
# Walk up from qola/cpp_itfs/ to QoLA root, then into 3rdparty/aiter.
_QOLA_ROOT = _SCRIPT_DIR.parent.parent
_AITER_CK_DIR = (
    _QOLA_ROOT / "3rdparty" / "aiter" / "csrc" / "ck_gemm_a4w4_blockscale"
)

sys.path.insert(0, str(_AITER_CK_DIR))
from gemm_a4w4_blockscale_common import (  # noqa: E402
    default_kernels_dict,
    kernelInstance,
    kernels_list,
)


# ---------------------------------------------------------------------------
# Torch-free CU count detection via rocminfo.
# ---------------------------------------------------------------------------

def _get_cu_count() -> int:
    """Return the number of compute units on the first GPU, or 0 on failure."""
    try:
        out = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            stripped = line.strip()
            if stripped.startswith("Compute Unit:"):
                return int(stripped.split(":")[1].strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    return 0


# ---------------------------------------------------------------------------
# Code-generation templates (torch-free versions)
# ---------------------------------------------------------------------------

# The function signature used throughout: raw pointers + dims.
_RAW_SIG = """\
    const void* a_ptr, const void* a_scale_ptr,
    const void* b_ptr, const void* b_scale_ptr,
    void* out_ptr,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_out,
    int stride_a_scale, int stride_b_scale,
    int splitK, hipStream_t stream"""


class gemm_a4w4_blockscale_codegen:
    def __init__(self, working_path: str, istune: bool = False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune

    # ---- per-kernel impl .cuh ----

    def gen_instance(self, k: kernelInstance) -> None:
        INSTANCE_CONTENT_nobias = f"""using DeviceGemmInstance = DeviceGemmHelperF4BlockScale<
            MFMA,
            CDataType,
            {k.BLOCK_SIZE},
            {k.MPerBLOCK}, {k.NPerBLOCK}, {k.KPerBLOCK},
            {k.AK1}, {k.BK1},
            {k.MPerXDL}, {k.NPerXDL},
            {k.WAVE_MAP_M}, {k.WAVE_MAP_N},
            S<{', '.join(str(x) for x in k.ABLOCK_TRANSFER)}>,
            S<{', '.join(str(x) for x in k.BBLOCK_TRANSFER)}>,
            {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
            {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
            S<{', '.join(str(x) for x in k.CBLOCK_TRANSFER)}>,
            {k.CBLOCK_SPV},
            ck::BlockGemmPipelineScheduler::{k.PIPELINE_Sched},
            ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
            ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;
        return gemm_a4w4_blockscale_impl<CDataType, DeviceGemmInstance>(
            a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, out_ptr,
            M, N, K, stride_a * 2, stride_b * 2, stride_out,
            stride_a_scale, stride_b_scale, splitK, stream);
"""

        impl_str = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "qola_gemm_a4w4_blockscale_common.cuh"
#include <hip/hip_runtime.h>

template <typename CDataType>
int
{k.name}(
{_RAW_SIG}
    )
{{{{
    int M_local = M;
    int N_local = N;
    int K_local = K;
    {{INSTANCE_CONTENT_nopad}}
}}}}

"""
        impl_str = impl_str.format(
            INSTANCE_CONTENT_pad=INSTANCE_CONTENT_nobias.format(GemmSpec="MNKPadding"),
            INSTANCE_CONTENT_nopad=INSTANCE_CONTENT_nobias.format(GemmSpec="Default"),
        )

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(impl_str)

        # ---- per-kernel instance .cpp files (explicit template instantiation) ----
        INSTANCE_template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/{name}.cuh"

template int
{name}<{dtypes}>(
{sig}
    );

"""
        bf16_inst = INSTANCE_template.format(name=k.name, dtypes="B16", sig=_RAW_SIG)
        fp16_inst = INSTANCE_template.format(name=k.name, dtypes="F16", sig=_RAW_SIG)

        Path(
            os.path.join(self.instances_path, f"{k.name}_dFP32_eBF16.cpp")
        ).write_text(bf16_inst)
        Path(
            os.path.join(self.instances_path, f"{k.name}_dFP32_eFP16.cpp")
        ).write_text(fp16_inst)

    # ---- lookup table header ----

    def gen_lookup_dict(self, kernels_dict: dict) -> None:
        LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(CTYPE)                                                                                      \\
   {                                                                                                                             \\"""

        LOOKUP_template = """
       {{{MNK},                                                                                                       \\
        {kernel_name}<CTYPE>}},                       \\"""

        LOOKUP_end = """
   }

#endif // USE_ROCM
"""
        with open(
            os.path.join(self.working_path, "gemm_a4w4_blockscale_lookup.h"), "w"
        ) as f:
            f.write(LOOKUP_head)
            for mnk, k in kernels_dict.items():
                if not self.istune and (isinstance(mnk, tuple) and mnk[0] > 0):
                    f.write(
                        LOOKUP_template.format(
                            MNK="{"
                            + ", ".join(str(x) for x in list(mnk))
                            + "}",
                            kernel_name=k.name,
                        )
                    )
                elif self.istune and isinstance(mnk, int):
                    f.write(LOOKUP_template.format(MNK=mnk, kernel_name=k.name))
            f.write(LOOKUP_end)

    # ---- manifest header (forward declarations) ----

    def gen_manifest_head(self, kernels_dict: dict) -> None:
        MANIFEST_head = f"""#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// QoLA torch-free variant — raw-pointer signatures.

#ifdef USE_ROCM

#include <cstdlib>
#include <hip/hip_runtime.h>
"""
        MANIFEST_template = """
template <typename CDataType>
int
{kernel_name}(
{sig}
    );
"""
        MANIFEST_end = """

#endif // USE_ROCM
"""

        with open(
            os.path.join(self.working_path, "gemm_a4w4_blockscale_manifest.h"), "w"
        ) as f:
            f.write(MANIFEST_head)
            seen = set()
            for mnk, k in kernels_dict.items():
                if k.name not in seen:
                    f.write(MANIFEST_template.format(kernel_name=k.name, sig=_RAW_SIG))
                    seen.add(k.name)
            f.write(MANIFEST_end)

    # ---- orchestrator ----

    def gen_instances(self, kernels_dict: dict) -> None:
        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        seen = set()
        for mnk, k in kernels_dict.items():
            if k.name not in seen:
                self.gen_instance(k)
                seen.add(k.name)

        self.gen_lookup_dict(kernels_dict)
        self.gen_manifest_head(kernels_dict)


# ---------------------------------------------------------------------------
# Tune-file loading (torch-free: uses rocminfo for CU count).
# ---------------------------------------------------------------------------

def get_tune_dict(tune_dict_csv: str) -> dict:
    tune_dict = dict(default_kernels_dict)
    if os.path.exists(tune_dict_csv):
        tune_df = pd.read_csv(tune_dict_csv)
        cu_num = _get_cu_count()
        if cu_num > 0 and "cu_num" in tune_df.columns:
            tune_df = tune_df[tune_df["cu_num"] == cu_num].reset_index()
        for i in range(len(tune_df)):
            M = tune_df.loc[i, "M"]
            N = tune_df.loc[i, "N"]
            K = tune_df.loc[i, "K"]
            kid = tune_df.loc[i, "kernelId"]
            if kid < 0 or kid > len(kernels_list):
                continue
            tune_dict[(M, N, K)] = kernels_list[kid]
    return tune_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="gen_blockscale_instances",
        description="QoLA torch-free codegen for CK gemm_a4w4_blockscale",
    )
    parser.add_argument(
        "-w", "--working_path", default="./", required=False,
        help="output directory for generated headers/sources",
    )
    parser.add_argument(
        "-f", "--tune_file",
        default="aiter/configs/a4w4_blockscale_tuned_gemm.csv",
        required=False,
        help="CSV file with tuned kernel mappings",
    )
    parser.add_argument(
        "--tune", action="store_true", required=False,
        help="generate tune instances (all kernels)",
    )

    args = parser.parse_args()
    codegen = gemm_a4w4_blockscale_codegen(args.working_path, args.tune)

    if args.tune:
        codegen.gen_instances(kernels_list)
    else:
        codegen.gen_instances(get_tune_dict(args.tune_file))
