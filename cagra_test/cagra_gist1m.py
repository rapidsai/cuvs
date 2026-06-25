import argparse
import csv
import time
import numpy as np
import cupy as cp
from cuvs.neighbors import cagra


def read_fvecs(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    dim = int(data[0])
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].view(np.float32)


def read_ivecs(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    dim = int(data[0])
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].copy()


def to_numpy(x):
    if hasattr(x, "copy_to_host"):
        return np.asarray(x.copy_to_host())
    return cp.asnumpy(x)


def recall_at_k(found: np.ndarray, gt: np.ndarray, k: int) -> float:
    found = found[:, :k]
    gt = gt[:, :k]

    hit = 0
    for i in range(found.shape[0]):
        hit += len(set(found[i]) & set(gt[i]))

    return hit / (found.shape[0] * k)


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--gt", required=True)

    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--base-limit", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)

    # GIST1M paper setting: d = 48
    # Initial/intermediate graph degree: 3d = 144
    parser.add_argument("--graph-degree", type=int, default=48)
    parser.add_argument("--intermediate-graph-degree", type=int, default=144)
    parser.add_argument("--build-algo", default="nn_descent")
    parser.add_argument("--nn-descent-niter", type=int, default=20)

    parser.add_argument("--itopk-list", default="100,128,160,192,256,320,384,448,512")
    parser.add_argument("--search-width", type=int, default=1)

    parser.add_argument(
        "--algo",
        default="single_cta",
        choices=["auto", "single_cta", "multi_cta"],
        help="CAGRA search algorithm: auto, single_cta, or multi_cta",
    )

    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", default="cagra_gist1m_sweep.csv")

    args = parser.parse_args()
    itopk_list = parse_int_list(args.itopk_list)

    print("==== Load dataset ====")
    t0 = time.perf_counter()
    base_np = read_fvecs(args.base).astype(np.float32)
    query_np = read_fvecs(args.query).astype(np.float32)
    gt_np = read_ivecs(args.gt)
    t1 = time.perf_counter()

    if args.base_limit > 0:
        base_np = base_np[:args.base_limit]

    if args.query_limit > 0:
        query_np = query_np[:args.query_limit]
        gt_np = gt_np[:args.query_limit]

    print(f"base shape       : {base_np.shape}")
    print(f"query shape      : {query_np.shape}")
    print(f"groundtruth shape: {gt_np.shape}")
    print(f"load time [s]    : {t1 - t0:.6f}")

    print()
    print("==== Transfer dataset to GPU ====")
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    dataset = cp.asarray(base_np)
    queries = cp.asarray(query_np)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    transfer_time = t1 - t0
    print(f"transfer time [s]: {transfer_time:.6f}")

    print()
    print("==== Build CAGRA index ====")
    print(f"metric                   : sqeuclidean")
    print(f"build_algo               : {args.build_algo}")
    print(f"graph_degree             : {args.graph_degree}")
    print(f"intermediate_graph_degree: {args.intermediate_graph_degree}")
    print(f"nn_descent_niter         : {args.nn_descent_niter}")

    index_params = cagra.IndexParams(
        metric="sqeuclidean",
        intermediate_graph_degree=args.intermediate_graph_degree,
        graph_degree=args.graph_degree,
        build_algo=args.build_algo,
        nn_descent_niter=args.nn_descent_niter,
    )

    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    index = cagra.build(index_params, dataset)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    build_time = t1 - t0

    print(f"graph build time [s]: {build_time:.6f}")

    results = []

    print()
    print("==== Sweep search parameters ====")
    print(f"k           : {args.k}")
    print(f"search_algo : {args.algo}")
    print(f"search_width: {args.search_width}")
    print(f"itopk_list  : {itopk_list}")
    print(f"repeat      : {args.repeat}")

    for itopk_size in itopk_list:
        if itopk_size < args.k:
            print(f"[SKIP] itopk_size={itopk_size} is smaller than k={args.k}")
            continue

        if args.algo == "single_cta" and itopk_size > 512:
            print(f"[SKIP] single_cta does not allow itopk_size > 512: itopk_size={itopk_size}")
            continue

        print()
        print(f"---- algo = {args.algo}, itopk_size = {itopk_size} ----")

        search_params = cagra.SearchParams(
            algo=args.algo,
            itopk_size=itopk_size,
            search_width=args.search_width,
        )

        # warmup
        distances, neighbors = cagra.search(search_params, index, queries, args.k)
        cp.cuda.Device().synchronize()

        search_times = []
        last_distances = None
        last_neighbors = None

        for r in range(args.repeat):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            distances, neighbors = cagra.search(search_params, index, queries, args.k)
            cp.cuda.Device().synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            search_times.append(elapsed)
            last_distances = distances
            last_neighbors = neighbors

            print(f"run {r + 1}: {elapsed:.6f} s")

        avg_search_time = sum(search_times) / len(search_times)
        qps = queries.shape[0] / avg_search_time

        search_time_sec_per_query = avg_search_time / queries.shape[0]
        search_time_ms_per_query = search_time_sec_per_query * 1000
        search_time_us_per_query = search_time_sec_per_query * 1_000_000

        neighbors_np = to_numpy(last_neighbors)
        distances_np = to_numpy(last_distances)

        rec = recall_at_k(neighbors_np, gt_np, args.k)

        print(f"avg search time for all queries [s]: {avg_search_time:.6f}")
        print(f"search time [ms/query]            : {search_time_ms_per_query:.6f}")
        print(f"search time [us/query]            : {search_time_us_per_query:.6f}")
        print(f"QPS [queries/sec]                 : {qps:.3f}")
        print(f"recall@{args.k}                     : {rec:.6f}")

        results.append({
            "dataset": "GIST1M",
            "dataset_size": base_np.shape[0],
            "dimension": base_np.shape[1],
            "num_queries": query_np.shape[0],
            "k": args.k,
            "graph_degree": args.graph_degree,
            "intermediate_graph_degree": args.intermediate_graph_degree,
            "build_algo": args.build_algo,
            "nn_descent_niter": args.nn_descent_niter,
            "build_time_sec": build_time,
            "algo": args.algo,
            "search_width": args.search_width,
            "itopk_size": itopk_size,
            "avg_search_time_sec_all_queries": avg_search_time,
            "search_time_sec_per_query": search_time_sec_per_query,
            "search_time_ms_per_query": search_time_ms_per_query,
            "search_time_us_per_query": search_time_us_per_query,
            "qps": qps,
            f"recall@{args.k}": rec,
        })

    print()
    print("==== Save results ====")

    fieldnames = [
        "dataset",
        "dataset_size",
        "dimension",
        "num_queries",
        "k",
        "graph_degree",
        "intermediate_graph_degree",
        "build_algo",
        "nn_descent_niter",
        "build_time_sec",
        "algo",
        "search_width",
        "itopk_size",
        "avg_search_time_sec_all_queries",
        "search_time_sec_per_query",
        "search_time_ms_per_query",
        "search_time_us_per_query",
        "qps",
        f"recall@{args.k}",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"saved: {args.output}")

    print()
    print("==== Summary ====")
    for row in results:
        print(
            f"dataset={row['dataset']}, "
            f"algo={row['algo']}, "
            f"itopk_size={row['itopk_size']:4d}, "
            f"time={row['search_time_us_per_query']:.3f} us/query, "
            f"time={row['search_time_ms_per_query']:.6f} ms/query, "
            f"QPS={row['qps']:.3f}, "
            f"recall@{args.k}={row[f'recall@{args.k}']:.6f}"
        )

    print()
    print("==== Example results from last parameter ====")
    if results:
        print("neighbors[:5]:")
        print(neighbors_np[:5])
        print()
        print("distances[:5]:")
        print(distances_np[:5])


if __name__ == "__main__":
    main()
