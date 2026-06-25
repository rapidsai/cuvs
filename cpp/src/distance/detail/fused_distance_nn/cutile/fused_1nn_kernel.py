# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""cuTile fused GEMM + 1-NN kernels (InnerProduct, L2Expanded, CosineExpanded)."""

from __future__ import annotations

import cuda.tile as ct

ConstInt = ct.Constant[int]

# Default tile geometry; overridden per export via make_kernel(..., tile_m, tile_n, tile_k).
DEFAULT_TILE_M = 128
DEFAULT_TILE_N = 128
DEFAULT_TILE_K = 64

METRICS = ("inner_product", "l2_expanded", "cosine_expanded")


def make_kernel(
    data_type: str,
    metric: str,
    tile_m: int = DEFAULT_TILE_M,
    tile_n: int = DEFAULT_TILE_N,
    tile_k: int = DEFAULT_TILE_K,
):
    """Build a cuTile kernel with metric and tile sizes baked in at compile time."""
    if data_type not in ("half", "float"):
        raise ValueError(f"Unsupported data_type {data_type!r}")
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}")

    acc_dtype = ct.float32
    is_ip = metric == "inner_product"
    is_l2 = metric == "l2_expanded"
    is_cos = metric == "cosine_expanded"

    @ct.kernel
    def fused_1nn_kernel(
        A,
        B,
        A_norm,
        B_norm,
        OutIdx,
        OutDist,
        M,
        N,
        K,
        tm: ConstInt,
        tn: ConstInt,
        tk: ConstInt,
    ):
        bidm = ct.bid(0)

        if is_ip:
            best_dist = ct.full((tm,), -3.4e38, acc_dtype)
        else:
            best_dist = ct.full((tm,), 3.4e38, acc_dtype)
        best_idx = ct.zeros((tm,), ct.int64)

        num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
        num_tiles_n = ct.num_tiles(B, axis=0, shape=(tn, tk))
        zero_pad = ct.PaddingMode.ZERO

        for n in range(num_tiles_n):
            accumulator = ct.full((tm, tn), 0, dtype=acc_dtype)

            for k in range(num_tiles_k):
                a = ct.load(
                    A, index=(bidm, k), shape=(tm, tk), padding_mode=zero_pad
                )
                b_T = ct.load(
                    B, index=(n, k), shape=(tn, tk), padding_mode=zero_pad
                )
                accumulator = ct.mma(a, ct.transpose(b_T), accumulator)

            if is_ip:
                score = accumulator
            elif is_l2 or is_cos:
                a_norm = ct.load(
                    A_norm, index=(bidm,), shape=(tm,), padding_mode=zero_pad
                )
                b_norm = ct.load(
                    B_norm, index=(n,), shape=(tn,), padding_mode=zero_pad
                )
                if is_l2:
                    # L2 expanded: ||x||^2 + ||y||^2 - 2 * dot(x, y); norms are squared.
                    score = (
                        a_norm[:, None] + b_norm[None, :] - (2.0 * accumulator)
                    )
                elif is_cos:
                    # Cosine expanded distance: 1 - dot / (||x|| * ||y||); norms are L2 (not squared).
                    # No sqrt during the reduction — only arithmetic on stored distance if needed.
                    denom = a_norm[:, None] * b_norm[None, :]
                    score = 1.0 - (accumulator / denom)

            # Only the final N-tile can include zero-padded centroid columns.
            if n == num_tiles_n - 1:
                col = ct.arange(tn, dtype=ct.int64)
                global_col = n * tn + col
                valid = global_col < N
                if is_ip:
                    score = ct.where(valid[None, :], score, -3.4e38)
                else:
                    score = ct.where(valid[None, :], score, 3.4e38)

            if is_ip:
                curr_best = ct.max(score, axis=1)
                curr_idx = ct.argmax(score, axis=1)
                update = curr_best > best_dist
                best_dist = ct.where(update, curr_best, best_dist)
            else:
                curr_best = ct.min(score, axis=1)
                curr_idx = ct.argmin(score, axis=1)
                update = curr_best < best_dist
                best_dist = ct.where(update, curr_best, best_dist)

            best_idx = ct.where(update, n * tn + curr_idx, best_idx)

        ct.store(OutIdx, index=(bidm,), tile=best_idx)
        ct.store(OutDist, index=(bidm,), tile=best_dist)

    return fused_1nn_kernel


def kernel_symbol(data_abbrev: str, metric_abbrev: str) -> str:
    """Must stay in sync with fused_1nn_kernel_entrypoint() in fused_1nn_planner.hpp."""
    return f"fused_1nn_{data_abbrev}_{metric_abbrev}"


def metric_abbrev(metric: str) -> str:
    return {
        "inner_product": "ip",
        "l2_expanded": "l2",
        "cosine_expanded": "cos",
    }[metric]
