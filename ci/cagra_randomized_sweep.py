# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[DO NOT MERGE] Replays the cuvs-java CagraRandomizedIT check on random cases.

Each case draws its parameters like CagraRandomizedIT does (or from an "edge" mode
with small datasets), builds a CAGRA index with NN-descent and the default search
parameters, and compares the result with an exact float64 search on the CPU. For
every case it records the old test check (the first 5 results by returned distance
must be in the exact top 2k+10), recall@k, and the properties that must always
hold. One JSON line per case goes to the output file; failures are also printed.
"""

import argparse
import json
import sys
import time

import cupy as cp
import numpy as np

import cuvs
from cuvs.neighbors import cagra

try:
    from cuvs.neighbors import filters
except ImportError:  # pragma: no cover
    filters = None


def draw_case(rng, mode):
    if mode == "java":
        n = int(rng.integers(0, 10_000)) + 2
        dim = int(rng.integers(0, 2048)) + 1
    else:  # edge: small datasets, k close to n
        n = int(rng.integers(2, 301))
        dim = int(rng.integers(1, 2049))
    nq = int(rng.integers(0, 10)) + 1
    k = min(int(rng.integers(0, 64)) + 1, n)
    use_filter = bool(rng.integers(0, 2)) and filters is not None
    return n, dim, nq, k, use_filter


def run_case(rng, mode, case_id, args):
    n, dim, nq, k, use_filter = draw_case(rng, mode)
    data = (rng.random((n, dim), dtype=np.float32) * 100).astype(np.float32)
    queries = (rng.random((nq, dim), dtype=np.float32) * 100).astype(
        np.float32
    )
    keep = None
    if use_filter:
        keep = rng.integers(0, 2, size=n).astype(bool)

    # Exact squared L2 in float64; filtered rows are +inf, ties broken by row index like
    # the Java test (a stable sort over a TreeMap keyed by row index).
    d64 = data.astype(np.float64)
    q64 = queries.astype(np.float64)
    exact = (
        (q64 * q64).sum(1)[:, None]
        - 2.0 * q64 @ d64.T
        + (d64 * d64).sum(1)[None, :]
    )
    exact = np.maximum(exact, 0.0)
    if keep is not None:
        exact[:, ~keep] = np.inf
    order = np.argsort(exact, axis=1, kind="stable")

    params = cagra.IndexParams(build_algo=args.build_algo)
    # Like the Java test, build from host memory or from device memory. Older cuVS releases
    # keep a non-owning view of an aligned device dataset, so it has to outlive the index.
    on_host = bool(rng.integers(0, 2))
    d_data = None if on_host else cp.asarray(data)
    index = cagra.build(params, data if on_host else d_data)
    sp = cagra.SearchParams()
    kwargs = {}
    if keep is not None:
        words = (n + 31) // 32
        bits = np.zeros(words * 32, dtype=np.uint8)
        bits[:n] = keep
        bitset = np.packbits(bits, bitorder="little").view("<u4")
        kwargs["filter"] = filters.from_bitset(cp.asarray(bitset))
    dist, nbr = cagra.search(sp, index, cp.asarray(queries), k, **kwargs)
    dist = cp.asnumpy(cp.asarray(dist))
    nbr = cp.asnumpy(cp.asarray(nbr)).astype(np.int64)

    window = min(2 * k + 10, n)
    rec = dict(
        id=case_id,
        mode=mode,
        host=on_host,
        n=n,
        dim=dim,
        nq=nq,
        k=k,
        filter=use_filter,
        n_pass=int(keep.sum()) if keep is not None else n,
    )
    old_fail_q = []
    recalls = []
    problems = []
    worst = None
    for q in range(nq):
        valid = dist[q] < np.finfo(np.float32).max
        ids = nbr[q][valid]
        dq = dist[q][valid]
        n_valid_expected = min(k, rec["n_pass"])
        invalid_ids = [int(i) for i in ids if i < 0 or i >= n]
        dup = len(ids) - len(set(ids.tolist()))
        filtered_out = [
            int(i)
            for i in ids
            if 0 <= i < n and keep is not None and not keep[i]
        ]
        if invalid_ids:
            problems.append(f"q{q}:invalid_ids={invalid_ids[:5]}")
        if dup:
            problems.append(f"q{q}:duplicates={dup}")
        if filtered_out:
            problems.append(f"q{q}:filtered_out_returned={filtered_out[:5]}")
        if len(ids) < n_valid_expected:
            problems.append(f"q{q}:short={len(ids)}/{n_valid_expected}")
        if len(dq) > 1 and np.any(np.diff(dq) < 0):
            problems.append(f"q{q}:unsorted")
        # Returned distances vs exact distances of the returned rows.
        ok_ids = ids[(ids >= 0) & (ids < n)]
        if len(ok_ids):
            ex = exact[q, ok_ids]
            fin = np.isfinite(ex)
            if fin.any():
                rel = np.abs(
                    dq[(ids >= 0) & (ids < n)][fin] - ex[fin]
                ) / np.maximum(ex[fin], 1.0)
                if rel.max() > 1e-2:
                    problems.append(f"q{q}:dist_mismatch={rel.max():.3g}")
        # Tie-aware recall@k against the exact k-th distance among passing rows.
        kk = n_valid_expected
        if kk > 0:
            kth = exact[q, order[q, kk - 1]]
            good = sum(
                1
                for i in set(ok_ids.tolist())
                if exact[q, i] <= kth * (1 + 1e-6)
            )
            recalls.append(min(good, kk) / kk)
        # The old test check.
        top_window = set(order[q, :window].tolist())
        srt = np.argsort(dist[q], kind="stable")
        first5 = [
            int(nbr[q][j])
            for j in srt[: min(5, k)]
            if dist[q][j] < np.finfo(np.float32).max
        ]
        missing = [i for i in first5 if i not in top_window]
        if missing:
            i = missing[0]
            rank = int(np.sum(exact[q] < exact[q, i])) if 0 <= i < n else -1
            old_fail_q.append(
                dict(
                    q=q,
                    id=i,
                    true_rank=rank,
                    window=window,
                    passes=bool(keep is None or (0 <= i < n and keep[i])),
                )
            )
            if worst is None or rank > worst:
                worst = rank
    rec["recall_mean"] = float(np.mean(recalls)) if recalls else 1.0
    rec["recall_min"] = float(np.min(recalls)) if recalls else 1.0
    rec["old_check_fail"] = old_fail_q
    rec["problems"] = problems
    del index, d_data
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edge-fraction", type=float, default=0.3)
    ap.add_argument("--build-algo", default="nn_descent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-below", type=float, default=0.7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    end = time.time() + args.minutes * 60
    n_cases = n_old_fail = n_prob = 0
    recall_means = []
    by_filter = {}
    with open(args.out, "w") as f:
        i = 0
        while time.time() < end:
            mode = "edge" if rng.random() < args.edge_fraction else "java"
            try:
                rec = run_case(rng, mode, i, args)
            except Exception as e:  # report and continue
                rec = dict(
                    id=i,
                    mode=mode,
                    error=f"{type(e).__name__}: {str(e)[:300]}",
                )
                print(
                    f"[{args.tag}] ERROR case {i}: {rec['error']}", flush=True
                )
            f.write(json.dumps(rec) + "\n")
            f.flush()
            i += 1
            if "error" in rec:
                continue
            n_cases += 1
            recall_means.append(rec["recall_mean"])
            by_filter.setdefault(rec["filter"], []).append(rec["recall_mean"])
            if (
                rec["old_check_fail"]
                or rec["problems"]
                or rec["recall_mean"] < args.print_below
            ):
                n_old_fail += bool(rec["old_check_fail"])
                n_prob += bool(rec["problems"])
                print(f"[{args.tag}] CASE {json.dumps(rec)}", flush=True)
    r = np.array(recall_means) if recall_means else np.array([1.0])
    print(
        f"[{args.tag}] SUMMARY cuvs={cuvs.__version__} cases={n_cases} "
        f"old_check_fail={n_old_fail} invariant_problems={n_prob} "
        f"recall_mean: min={r.min():.4f} p0.1={np.quantile(r, 0.001):.4f} "
        f"p1={np.quantile(r, 0.01):.4f} p5={np.quantile(r, 0.05):.4f} "
        f"median={np.median(r):.4f} mean={r.mean():.4f}",
        flush=True,
    )
    for f, vals in sorted(by_filter.items()):
        v = np.array(vals)
        print(
            f"[{args.tag}] RECALL filter={f} cases={len(v)} min={v.min():.4f} "
            f"p1={np.quantile(v, 0.01):.4f} p5={np.quantile(v, 0.05):.4f} "
            f"p25={np.quantile(v, 0.25):.4f} median={np.median(v):.4f} "
            f"mean={v.mean():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
