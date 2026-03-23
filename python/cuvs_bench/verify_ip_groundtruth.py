#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone check: does groundtruth.neighbors.ibin match exact inner-product
# top-k on the full base? (No dependency on installed cuvs_bench version.)
#
# Usage:
#   python verify_ip_groundtruth.py base.fbin queries.fbin groundtruth.ibin [k] [query_row]
# Prints overlap, ids, and a side-by-side table of (idx, dot) for brute vs file (first 10 of k).
#
import struct
import sys

import numpy as np


def _read_shape(path):
    with open(path, "rb") as f:
        return struct.unpack("<II", f.read(8))


def mmap_fbin(path):
    nr, nc = _read_shape(path)
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(nr, nc))


def mmap_ibin(path):
    nr, nc = _read_shape(path)
    return np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(nr, nc))


def ip_scores_for_indices(dataset_mm, query_vec, indices):
    """Inner product base[idx]·q for each index (for diagnostics)."""
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    n = dataset_mm.shape[0]
    out = np.empty(len(indices), dtype=np.float64)
    for i, idx in enumerate(indices):
        idx = int(idx)
        if idx < 0 or idx >= n:
            out[i] = np.nan
        else:
            row = np.asarray(dataset_mm[idx], dtype=np.float32)
            out[i] = float(np.dot(row, q))
    return out


def brute_ip_topk_chunked(query_vec, dataset_mm, k, chunk_rows=65536):
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    n = dataset_mm.shape[0]
    top_sim = np.full(k, -np.inf, dtype=np.float64)
    top_idx = np.zeros(k, dtype=np.int64)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        block = np.asarray(dataset_mm[start:end], dtype=np.float32)
        sim = block @ q
        merged_sim = np.concatenate([top_sim, sim.astype(np.float64)])
        merged_idx = np.concatenate(
            [top_idx, np.arange(start, end, dtype=np.int64)]
        )
        pick = np.argsort(-merged_sim)[:k]
        top_sim = merged_sim[pick]
        top_idx = merged_idx[pick]
    return top_idx


def main():
    if len(sys.argv) < 4:
        print(
            "usage: python verify_ip_groundtruth.py "
            "base.fbin queries.fbin groundtruth.neighbors.ibin [k] [query_row]",
            file=sys.stderr,
        )
        sys.exit(2)
    base_p, q_p, gt_p = sys.argv[1:4]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    qi = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    base = mmap_fbin(base_p)
    queries = mmap_fbin(q_p)
    gt = mmap_ibin(gt_p)
    if base.shape[1] != queries.shape[1]:
        print(
            f"dim mismatch base {base.shape[1]} vs queries {queries.shape[1]}",
            file=sys.stderr,
        )
        sys.exit(1)
    kk = min(k, gt.shape[1])
    if qi >= queries.shape[0] or qi >= gt.shape[0]:
        print("query_row out of range", file=sys.stderr)
        sys.exit(1)

    print(
        f"shapes: base={base.shape} queries={queries.shape} gt={gt.shape} "
        f"(gt rows should match queries rows; gt cols >= k)"
    )

    truth = brute_ip_topk_chunked(queries[qi], base, kk)
    got = np.asarray(gt[qi, :kk], dtype=np.int64)
    n_base = base.shape[0]
    bad_got = np.logical_or(got < 0, got >= n_base)
    if bad_got.any():
        print(
            f"warning: {bad_got.sum()} neighbor id(s) out of range [0, {n_base}) "
            f"in file row {qi} — possible wrong dtype/endian or corrupt header"
        )

    inter = len(set(truth.tolist()) & set(got.tolist()))
    print(f"query_row={qi} k={kk} overlap true∩file = {inter}/{kk}")
    print(f"  brute IP top-{kk} ids: {truth.tolist()}")
    print(f"  file row ids:        {got.tolist()}")

    qv = np.asarray(queries[qi], dtype=np.float32).ravel()
    truth_dots = ip_scores_for_indices(base, qv, truth)
    got_dots = ip_scores_for_indices(base, qv, got)
    # Sort by dot descending so you see "best first" (same order as true IP ranking)
    t_order = np.argsort(-truth_dots)
    g_order = np.argsort(-got_dots)
    show = min(10, kk)
    print()
    print(
        f"Inner product (dot) scores for query_row={qi} "
        f"(showing first {show} of {kk}; sorted by dot desc within each list):"
    )
    print(f"  {'rank':>4}  {'brute idx':>12}  {'brute dot':>14}  |  {'file idx':>12}  {'file dot':>14}")
    for r in range(show):
        ti = t_order[r]
        gi = g_order[r]
        print(
            f"  {r + 1:4d}  {int(truth[ti]):12d}  {truth_dots[ti]:14.6g}  |  "
            f"{int(got[gi]):12d}  {got_dots[gi]:14.6g}"
        )
    if kk > show:
        print(f"  ... ({kk - show} more rows per column not shown)")
    print()
    print(
        f"  brute: min_dot={np.nanmin(truth_dots):.6g}  max_dot={np.nanmax(truth_dots):.6g}  "
        f"(true IP top-{kk} should have the k largest dots in the dataset)"
    )
    print(
        f"  file:  min_dot={np.nanmin(got_dots):.6g}  max_dot={np.nanmax(got_dots):.6g}  "
        f"(if file is IP GT, these should match brute up to ties)"
    )

    if inter < kk:
        print(
            "If not k/k, GT file is not raw IP top-k for this base/queries "
            "(or rows misaligned)."
        )


if __name__ == "__main__":
    main()
