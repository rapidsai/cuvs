# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Export fused 1-NN cuTile kernels to cubin or TileIR bytecode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import cuda.tile as ct
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    ConstantConstraint,
    KernelSignature,
    ScalarConstraint,
    export_kernel,
)

from fused_1nn_kernel import KERNELS, KERNEL_SYMBOLS, TILE_CONSTANTS

DEFAULT_TILEIR_BYTECODE_VERSION = "13.1"
# cuTile requires a gpu_code even for TileIR bytecode export: it selects the compilation
# target / feature set for lowering, not the runtime architecture (the driver JITs at load).
DEFAULT_TILEIR_EXPORT_GPU_CODE = "sm_80"


def _dtype_for(data_type: str):
    if data_type == "half":
        return ct.float16
    if data_type == "float":
        return ct.float32
    raise ValueError(f"Unsupported data_type {data_type!r}")


def _kernel_signature(data_type: str) -> KernelSignature:
    elem = _dtype_for(data_type)
    array = ArrayConstraint(
        elem,
        2,
        index_dtype=ct.int64,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
    )
    idx_array = ArrayConstraint(
        ct.int64,
        1,
        index_dtype=ct.int64,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(1,),
    )
    dist_array = ArrayConstraint(
        ct.float32,
        1,
        index_dtype=ct.int64,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(1,),
    )
    tm, tn, tk = TILE_CONSTANTS
    return KernelSignature(
        parameters=[
            array,
            array,
            idx_array,
            dist_array,
            ScalarConstraint(ct.int64),
            ScalarConstraint(ct.int64),
            ScalarConstraint(ct.int64),
            ConstantConstraint(tm),
            ConstantConstraint(tn),
            ConstantConstraint(tk),
        ],
        calling_convention=CallingConvention.cutile_python_v1(),
    ).with_symbol(KERNEL_SYMBOLS[data_type])


def export_binary(
    output_file: Path,
    *,
    output_format: Literal["cubin", "tileir_bytecode"],
    data_type: str,
    gpu_code: str,
    bytecode_version: str | None = None,
) -> str:
    kernel = KERNELS[data_type]
    signature = _kernel_signature(data_type)

    export_kwargs = {
        "kernel": kernel,
        "signatures": [signature],
        "output_file": str(output_file),
        "gpu_code": gpu_code,
        "output_format": output_format,
    }
    if output_format == "tileir_bytecode":
        export_kwargs["bytecode_version"] = (
            bytecode_version or DEFAULT_TILEIR_BYTECODE_VERSION
        )

    export_kernel(**export_kwargs)

    return signature.symbol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--format", choices=("cubin", "tileir_bytecode"), default="cubin"
    )
    parser.add_argument(
        "--data-type", choices=tuple(KERNELS.keys()), required=True
    )
    parser.add_argument(
        "--gpu-code",
        default=DEFAULT_TILEIR_EXPORT_GPU_CODE,
        help="Target SM for cubin export, or compile hint for TileIR bytecode export",
    )
    parser.add_argument(
        "--bytecode-version", default=DEFAULT_TILEIR_BYTECODE_VERSION
    )
    args = parser.parse_args()

    print(
        export_binary(
            args.output_file,
            output_format=args.format,
            data_type=args.data_type,
            gpu_code=args.gpu_code,
            bytecode_version=args.bytecode_version,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
