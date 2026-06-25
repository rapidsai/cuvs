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

from fused_1nn_kernel import METRICS, kernel_symbol, make_kernel, metric_abbrev

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


def _data_abbrev(data_type: str) -> str:
    return {"half": "h", "float": "f"}[data_type]


def _relaxed_matrix_constraint(elem_dtype):
    """Array constraints matching the relaxed TMA-friendly layout from gemm_nn_cutile."""
    return ArrayConstraint(
        elem_dtype,
        ndim=2,
        index_dtype=ct.int64,
        stride_lower_bound_incl=(0, None),
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(None, 1),
        stride_divisible_by=(8, 1),
        shape_divisible_by=(1, 1),
        base_addr_divisible_by=16,
    )


def _relaxed_vector_constraint(elem_dtype, *, tma_friendly: bool = False):
    base_div = 16 if tma_friendly else 1
    return ArrayConstraint(
        elem_dtype,
        ndim=1,
        index_dtype=ct.int64,
        stride_lower_bound_incl=(None,),
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(1,),
        stride_divisible_by=(1,),
        shape_divisible_by=(1,),
        base_addr_divisible_by=base_div,
    )


def _kernel_signature(
    data_type: str,
    metric: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> KernelSignature:
    elem = _dtype_for(data_type)
    matrix = _relaxed_matrix_constraint(elem)
    norm_array = _relaxed_vector_constraint(elem, tma_friendly=True)
    idx_array = _relaxed_vector_constraint(ct.int64)
    dist_array = _relaxed_vector_constraint(ct.float32)

    abbrev = _data_abbrev(data_type)
    symbol = kernel_symbol(abbrev, metric_abbrev(metric))

    return KernelSignature(
        parameters=[
            matrix,
            matrix,
            norm_array,
            norm_array,
            idx_array,
            dist_array,
            ScalarConstraint(ct.int64),
            ScalarConstraint(ct.int64),
            ScalarConstraint(ct.int64),
            ConstantConstraint(tile_m),
            ConstantConstraint(tile_n),
            ConstantConstraint(tile_k),
        ],
        calling_convention=CallingConvention.cutile_python_v1(),
    ).with_symbol(symbol)


def export_binary(
    output_file: Path,
    *,
    output_format: Literal["cubin", "tileir_bytecode"],
    data_type: str,
    metric: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    gpu_code: str,
    bytecode_version: str | None = None,
) -> str:
    kernel = make_kernel(data_type, metric, tile_m, tile_n, tile_k)
    signature = _kernel_signature(data_type, metric, tile_m, tile_n, tile_k)

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
        "--data-type", choices=("half", "float"), required=True
    )
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--tile-m", type=int, required=True)
    parser.add_argument("--tile-n", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
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
            metric=args.metric,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            gpu_code=args.gpu_code,
            bytecode_version=args.bytecode_version,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
