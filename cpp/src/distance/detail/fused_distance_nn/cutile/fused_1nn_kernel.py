# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""cuTile fused GEMM + inner-product 1-NN (argmax dot product) for cuVS."""

from __future__ import annotations

import cuda.tile as ct

ConstInt = ct.Constant[int]

TILE_M = 128
TILE_N = 256
TILE_K = 64


def _make_kernel(data_type: str):
    if data_type == "half":
        dtype = ct.float16
        acc_dtype = ct.float32
    elif data_type == "float":
        dtype = ct.float32
        acc_dtype = ct.float32
    else:
        raise ValueError(f"Unsupported data_type {data_type!r}")

    @ct.kernel
    def fused_1nn_kernel(A, B, OutIdx, OutDist, M, N, K, tm: ConstInt, tn: ConstInt, tk: ConstInt):
        bidm = ct.bid(0)

        best_dist = ct.full((tm,), -3.4e38, acc_dtype)
        best_idx = ct.zeros((tm,), ct.int64)

        num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
        num_tiles_n = ct.num_tiles(B, axis=0, shape=(tn, tk))
        zero_pad = ct.PaddingMode.ZERO

        for n in range(num_tiles_n):
            accumulator = ct.full((tm, tn), 0, dtype=acc_dtype)

            for k in range(num_tiles_k):
                a = ct.load(A, index=(bidm, k), shape=(tm, tk), padding_mode=zero_pad)
                b_T = ct.load(B, index=(n, k), shape=(tn, tk), padding_mode=zero_pad)
                accumulator = ct.mma(a, ct.transpose(b_T), accumulator)

            curr_max = ct.max(accumulator, axis=1)
            curr_idx = ct.argmax(accumulator, axis=1)

            update = curr_max > best_dist
            best_dist = ct.where(update, curr_max, best_dist)
            best_idx = ct.where(update, n * tn + curr_idx, best_idx)

        ct.store(OutIdx, index=(bidm,), tile=best_idx)
        ct.store(OutDist, index=(bidm,), tile=best_dist)

    return fused_1nn_kernel


KERNELS = {
    "half": _make_kernel("half"),
    "float": _make_kernel("float"),
}

KERNEL_SYMBOLS = {
    "half": "fused_1nn_half",
    "float": "fused_1nn_float",
}

TILE_CONSTANTS = (TILE_M, TILE_N, TILE_K)
