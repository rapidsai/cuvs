#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
r"""KMeans fit+predict benchmark: baseline / cuTile / flash-kmeans.

Single impl (activate the target conda env first):
  python benchmark_kmeans.py --impl baseline|cutile|flash --n N --d D --k K \\
    --max-iter 5 --tol 1e-4 --seed 42 \\
    --warmup-fit 1 --iters-fit 3 --warmup-pred 1 --iters-pred 3

Compare (subprocess per impl; export env vars, then --compare):
  export BENCH_CONDA=/path/to/miniforge3
  export BENCH_ENV_BASE=cuvs_2608_base
  export BENCH_ENV_CUTILE=cuvs_2608
  export BENCH_ENV_FLASH=cuvs_2608_base
  python benchmark_kmeans.py --compare --n 33554432 --d 32 --k 64 \\
    --max-iter 5 --tol 1e-4 --seed 42 \\
    --warmup-fit 1 --iters-fit 3 --warmup-pred 1 --iters-pred 3

Smoke test (small shape, single impl):
  conda activate cuvs_2608
  python benchmark_kmeans.py --impl cutile --n 10000 --d 32 --k 8 \\
    --max-iter 2 --tol 1e-4 --seed 42 \\
    --warmup-fit 0 --iters-fit 1 --warmup-pred 0 --iters-pred 1

Required for --compare (no defaults):
  BENCH_CONDA       path to miniforge/conda root
  BENCH_ENV_BASE    conda env name for baseline libcuvs
  BENCH_ENV_CUTILE  conda env name for cuTile libcuvs
  BENCH_ENV_FLASH   conda env name for flash-kmeans
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMPLS = ("baseline", "cutile", "flash")


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"required environment variable {name} is not set")
    return val


def _impl_config() -> dict[str, dict]:
    conda = Path(_require_env("BENCH_CONDA"))
    return {
        "baseline": {
            "bench_mode": "cuvs_base",
            "conda": conda,
            "conda_env": _require_env("BENCH_ENV_BASE"),
        },
        "cutile": {
            "bench_mode": "cuvs_cutile",
            "conda": conda,
            "conda_env": _require_env("BENCH_ENV_CUTILE"),
        },
        "flash": {
            "bench_mode": "flash",
            "conda": conda,
            "conda_env": _require_env("BENCH_ENV_FLASH"),
        },
    }


@dataclass
class BenchResult:
    impl: str
    fit_median_ms: float | None = None
    predict_median_ms: float | None = None
    n_iter: int | None = None
    inertia: float | None = None
    error: str | None = None


def median(xs: list[float]) -> float:
    import numpy as np

    return float(np.median(xs))


def run_benchmark(
    bench_mode: str,
    n: int,
    d: int,
    k: int,
    *,
    max_iter: int,
    tol: float,
    seed: int,
    warmup_fit: int,
    iters_fit: int,
    warmup_pred: int,
    iters_pred: int,
) -> BenchResult:
    import numpy as np

    rng = np.random.default_rng(seed)
    init_centroids_host = rng.standard_normal((k, d), dtype=np.float32)
    x_host = rng.standard_normal((n, d), dtype=np.float32)
    input_gib = n * d * 4 / (1024**3)

    label = {
        "cuvs_base": "baseline",
        "cuvs_cutile": "cutile",
        "flash": "flash",
    }[bench_mode]
    print(
        f"=== N={n:,} D={d} K={k:,} iters={max_iter} input={input_gib:.2f} GiB ===",
        flush=True,
    )

    if bench_mode in ("cuvs_base", "cuvs_cutile"):
        from cuda.bindings import runtime as cudart
        from pylibraft.common import device_ndarray

        from cuvs.cluster.kmeans import KMeansParams, fit, predict

        def sync():
            cudart.cudaDeviceSynchronize()

        x = device_ndarray(x_host)
        params = KMeansParams(
            n_clusters=k,
            max_iter=max_iter,
            tol=tol,
            metric="sqeuclidean",
            hierarchical=False,
            init_method="Array",
            n_init=1,
        )

        for _ in range(warmup_fit):
            fit(
                params, x, centroids=device_ndarray(init_centroids_host.copy())
            )
            sync()

        fit_times: list[float] = []
        n_iter = 0
        inertia = 0.0
        for _ in range(iters_fit):
            t0 = time.perf_counter()
            _, inertia, n_iter = fit(
                params, x, centroids=device_ndarray(init_centroids_host.copy())
            )
            sync()
            fit_times.append((time.perf_counter() - t0) * 1e3)

        centroids, _, _ = fit(
            params, x, centroids=device_ndarray(init_centroids_host.copy())
        )
        sync()

        for _ in range(warmup_pred):
            predict(params, x, centroids)
            sync()

        pred_times: list[float] = []
        for _ in range(iters_pred):
            t0 = time.perf_counter()
            predict(params, x, centroids)
            sync()
            pred_times.append((time.perf_counter() - t0) * 1e3)

        print(f"impl={label} init=Array", flush=True)
        print(f"fit_median_ms={median(fit_times):.2f}", flush=True)
        print(f"predict_median_ms={median(pred_times):.2f}", flush=True)
        print(f"n_iter={n_iter} inertia={inertia:.6g}", flush=True)
        return BenchResult(
            impl=label,
            fit_median_ms=median(fit_times),
            predict_median_ms=median(pred_times),
            n_iter=n_iter,
            inertia=inertia,
        )

    if bench_mode == "flash":
        import torch
        from flash_kmeans.assign_euclid_triton import euclid_assign_triton
        from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid

        def sync():
            torch.cuda.synchronize()

        x = torch.from_numpy(x_host).cuda()
        init_c = (
            torch.from_numpy(init_centroids_host.copy()).cuda().unsqueeze(0)
        )

        def run_fit(init):
            x_b = x.unsqueeze(0)
            _, centroids_b, _ = batch_kmeans_Euclid(
                x_b,
                k,
                max_iters=max_iter,
                tol=tol,
                init_centroids=init,
                verbose=False,
            )
            return centroids_b

        for _ in range(warmup_fit):
            run_fit(init_c.clone())
            sync()

        fit_times = []
        for _ in range(iters_fit):
            t0 = time.perf_counter()
            run_fit(init_c.clone())
            sync()
            fit_times.append((time.perf_counter() - t0) * 1e3)

        centroids_b = run_fit(init_c.clone())
        sync()

        x_b = x.unsqueeze(0)
        x_sq = (x_b**2).sum(dim=-1)

        for _ in range(warmup_pred):
            euclid_assign_triton(x_b, centroids_b, x_sq)
            sync()

        pred_times = []
        for _ in range(iters_pred):
            t0 = time.perf_counter()
            euclid_assign_triton(x_b, centroids_b, x_sq)
            sync()
            pred_times.append((time.perf_counter() - t0) * 1e3)

        print("impl=flash-kmeans init=Array", flush=True)
        print(f"fit_median_ms={median(fit_times):.2f}", flush=True)
        print(f"predict_median_ms={median(pred_times):.2f}", flush=True)
        return BenchResult(
            impl="flash",
            fit_median_ms=median(fit_times),
            predict_median_ms=median(pred_times),
        )

    raise ValueError(f"unknown bench_mode={bench_mode!r}")


def _parse_output(text: str, impl: str) -> BenchResult:
    fit_m = re.search(r"^fit_median_ms=([0-9.]+)", text, re.M)
    pred_m = re.search(r"^predict_median_ms=([0-9.]+)", text, re.M)
    if not fit_m or not pred_m:
        return BenchResult(impl=impl, error=text.strip() or "no output")
    n_iter_m = re.search(r"^n_iter=([0-9]+)", text, re.M)
    inertia_m = re.search(r"^inertia=([0-9.eE+-]+)", text, re.M)
    return BenchResult(
        impl=impl,
        fit_median_ms=float(fit_m.group(1)),
        predict_median_ms=float(pred_m.group(1)),
        n_iter=int(n_iter_m.group(1)) if n_iter_m else None,
        inertia=float(inertia_m.group(1)) if inertia_m else None,
    )


def _run_subprocess(
    impl: str,
    n: int,
    d: int,
    k: int,
    args: argparse.Namespace,
) -> BenchResult:
    cfg = _impl_config()[impl]
    conda = cfg["conda"]
    env_exports = " ".join(
        f'export {key}="{val}"'
        for key, val in (
            (
                "CUDA_VISIBLE_DEVICES",
                os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            ),
            ("MAX_ITER", args.max_iter),
            ("TOL", args.tol),
            ("SEED", args.seed),
            ("WARMUP_FIT", args.warmup_fit),
            ("ITERS_FIT", args.iters_fit),
            ("WARMUP_PRED", args.warmup_pred),
            ("ITERS_PRED", args.iters_pred),
        )
        if val != ""
    )
    cmd = f"""
set -eo pipefail
source "{conda}/etc/profile.d/conda.sh"
conda activate "{cfg["conda_env"]}"
{env_exports}
python3 "{ROOT / "benchmark_kmeans.py"}" --impl {impl} --n {n} --d {d} --k {k} \\
  --max-iter {args.max_iter} --tol {args.tol} --seed {args.seed} \\
  --warmup-fit {args.warmup_fit} --iters-fit {args.iters_fit} \\
  --warmup-pred {args.warmup_pred} --iters-pred {args.iters_pred}
"""
    proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return BenchResult(
            impl=impl, error=out.strip() or f"exit {proc.returncode}"
        )
    return _parse_output(out, impl)


def _speedup(base: float, other: float) -> str:
    if other <= 0:
        return "n/a"
    return f"{base / other:.2f}x"


def print_compare_table(
    results: list[BenchResult], n: int, d: int, k: int
) -> None:
    print(f"\n######## compare N={n} D={d} K={k} ########")
    print(f"{'impl':<10} {'fit_ms':>10} {'pred_ms':>10} {'notes'}")
    print("-" * 50)
    by_impl = {r.impl: r for r in results}
    for impl in IMPLS:
        r = by_impl.get(impl)
        if r is None:
            print(f"{impl:<10} {'—':>10} {'—':>10} missing")
            continue
        if r.error:
            print(
                f"{impl:<10} {'FAIL':>10} {'FAIL':>10} {r.error.splitlines()[-1][:40]}"
            )
            continue
        print(
            f"{impl:<10} {r.fit_median_ms:10.2f} {r.predict_median_ms:10.2f}"
        )

    flash = by_impl.get("flash")
    cutile = by_impl.get("cutile")
    if flash and cutile and flash.fit_median_ms and cutile.fit_median_ms:
        if not flash.error and not cutile.error:
            print(
                f"\nflash vs cutile fit: {_speedup(cutile.fit_median_ms, flash.fit_median_ms)}"
                f"  predict: {_speedup(cutile.predict_median_ms, flash.predict_median_ms)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare", action="store_true", help="run baseline, cutile, flash"
    )
    parser.add_argument("--impl", choices=IMPLS, help="single impl")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--tol", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup-fit", type=int, required=True)
    parser.add_argument("--iters-fit", type=int, required=True)
    parser.add_argument("--warmup-pred", type=int, required=True)
    parser.add_argument("--iters-pred", type=int, required=True)
    args = parser.parse_args()

    if args.compare:
        if args.impl:
            parser.error("--compare and --impl are mutually exclusive")
        _impl_config()  # validate required env before launching subprocesses
        results = [
            _run_subprocess(impl, args.n, args.d, args.k, args)
            for impl in IMPLS
        ]
        print_compare_table(results, args.n, args.d, args.k)
        return 0 if all(r.error is None for r in results) else 1

    if not args.impl:
        parser.error("set --impl for single-run mode, or use --compare")

    bench_mode = {
        "baseline": "cuvs_base",
        "cutile": "cuvs_cutile",
        "flash": "flash",
    }[args.impl]

    try:
        run_benchmark(
            bench_mode,
            args.n,
            args.d,
            args.k,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
            warmup_fit=args.warmup_fit,
            iters_fit=args.iters_fit,
            warmup_pred=args.warmup_pred,
            iters_pred=args.iters_pred,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
