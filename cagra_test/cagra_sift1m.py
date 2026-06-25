import argparse
import csv
import math
import os
import time

import numpy as np
import cupy as cp
from cuvs.neighbors import cagra


def read_fvecs(filename, limit=None):
    """
    fvecs形式のベクトルを読み込む。
    各ベクトルは [dim(int32), data(float32)*dim] の形式。
    """
    data = np.fromfile(filename, dtype=np.int32)
    if data.size == 0:
        raise ValueError(f"Empty file: {filename}")

    dim = data[0]
    vector_size = dim + 1

    if data.size % vector_size != 0:
        raise ValueError(
            f"Invalid fvecs file: {filename}, "
            f"data.size={data.size}, dim={dim}"
        )

    num_vectors = data.size // vector_size
    data = data.reshape(num_vectors, vector_size)

    vectors = data[:, 1:].view(np.float32)

    if limit is not None:
        vectors = vectors[:limit]

    return np.ascontiguousarray(vectors)


def read_ivecs(filename, limit=None):
    """
    ivecs形式のベクトルを読み込む。
    各ベクトルは [dim(int32), data(int32)*dim] の形式。
    """
    data = np.fromfile(filename, dtype=np.int32)
    if data.size == 0:
        raise ValueError(f"Empty file: {filename}")

    dim = data[0]
    vector_size = dim + 1

    if data.size % vector_size != 0:
        raise ValueError(
            f"Invalid ivecs file: {filename}, "
            f"data.size={data.size}, dim={dim}"
        )

    num_vectors = data.size // vector_size
    data = data.reshape(num_vectors, vector_size)

    vectors = data[:, 1:]

    if limit is not None:
        vectors = vectors[:limit]

    return np.ascontiguousarray(vectors)


def parse_itopk_list(text):
    """
    "100,110,120" のような文字列を [100, 110, 120] に変換する。
    """
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def recall_at_k(result_neighbors, groundtruth, k):
    """
    recall@k を計算する。

    result_neighbors: shape = (num_queries, k)
    groundtruth      : shape = (num_queries, >=k)
    """
    result_neighbors = result_neighbors[:, :k]
    groundtruth = groundtruth[:, :k]

    total = result_neighbors.shape[0] * k
    correct = 0

    for i in range(result_neighbors.shape[0]):
        correct += len(set(result_neighbors[i]) & set(groundtruth[i]))

    return correct / total


def get_team_size(search_params):
    """
    cuVSのバージョンによって team_size が存在しない可能性があるため、
    安全に取得する。
    """
    return getattr(search_params, "team_size", "auto")


def get_cta_per_query(algo, search_width, itopk_size):
    """
    multi-CTA時の1クエリあたりCTA数を計算する。

    cuVS CAGRAのmulti-CTAでは、内部的に1 CTAあたりの
    itopkサイズが32として扱われるため、

        cta_per_query = max(search_width, ceil(itopk_size / 32))

    として計算する。

    single-CTAでは1クエリあたり1 CTAとして表示する。
    """
    if algo == "multi_cta":
        return max(search_width, math.ceil(itopk_size / 32))
    else:
        return 1


def make_index_params(args):
    """
    CAGRAのグラフ構築パラメータを作成する。
    cuVSのバージョン差を考慮して、まず通常の指定を試す。
    """
    try:
        return cagra.IndexParams(
            metric=args.metric,
            graph_degree=args.graph_degree,
            intermediate_graph_degree=args.intermediate_graph_degree,
            build_algo=args.build_algo,
            nn_descent_niter=args.nn_descent_niter,
        )
    except TypeError:
        # 古い/異なるcuVSバージョンで一部引数名が合わない場合の保険
        params = cagra.IndexParams()
        params.metric = args.metric
        params.graph_degree = args.graph_degree
        params.intermediate_graph_degree = args.intermediate_graph_degree
        params.build_algo = args.build_algo

        if hasattr(params, "nn_descent_niter"):
            params.nn_descent_niter = args.nn_descent_niter

        return params


def make_search_params(args, itopk_size):
    """
    CAGRAの探索パラメータを作成する。
    team_sizeを明示した場合は設定し、指定しない場合はcuVSの自動設定に任せる。
    """
    kwargs = {
        "algo": args.algo,
        "itopk_size": itopk_size,
        "search_width": args.search_width,
    }

    if args.team_size is not None:
        kwargs["team_size"] = args.team_size

    try:
        return cagra.SearchParams(**kwargs)
    except TypeError:
        # cuVSのバージョン差を考慮した保険
        params = cagra.SearchParams()
        params.algo = args.algo
        params.itopk_size = itopk_size
        params.search_width = args.search_width

        if args.team_size is not None and hasattr(params, "team_size"):
            params.team_size = args.team_size

        return params


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)

    parser.add_argument("--base-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)

    parser.add_argument("--k", type=int, default=100)

    parser.add_argument("--metric", type=str, default="sqeuclidean")
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument("--intermediate-graph-degree", type=int, default=96)
    parser.add_argument("--build-algo", type=str, default="nn_descent")
    parser.add_argument("--nn-descent-niter", type=int, default=20)

    parser.add_argument(
        "--algo",
        type=str,
        default="single_cta",
        choices=["single_cta", "multi_cta"],
    )
    parser.add_argument("--search-width", type=int, default=1)

    parser.add_argument(
        "--itopk-list",
        type=str,
        default=None,
        help="例: 100,110,120,130,140,150,160,170,180,190,200",
    )
    parser.add_argument(
        "--itopk-size",
        type=int,
        default=128,
        help="--itopk-list を指定しない場合に使用する itopk_size",
    )

    parser.add_argument(
        "--team-size",
        type=int,
        default=None,
        help="指定しない場合は cuVS のデフォルト/自動設定を使用する",
    )

    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.itopk_list is not None:
        itopk_list = parse_itopk_list(args.itopk_list)
    else:
        itopk_list = [args.itopk_size]

    print("==== Load dataset ====")
    t0 = time.perf_counter()

    base = read_fvecs(args.base, args.base_limit)
    query = read_fvecs(args.query, args.query_limit)
    gt = read_ivecs(args.gt, args.query_limit)

    load_time = time.perf_counter() - t0

    print(f"base shape       : {base.shape}")
    print(f"query shape      : {query.shape}")
    print(f"groundtruth shape: {gt.shape}")
    print(f"load time [s]    : {load_time:.6f}")

    print()
    print("==== Transfer dataset to GPU ====")
    t0 = time.perf_counter()

    base_gpu = cp.asarray(base)
    query_gpu = cp.asarray(query)

    cp.cuda.Stream.null.synchronize()
    transfer_time = time.perf_counter() - t0

    print(f"transfer time [s]: {transfer_time:.6f}")

    print()
    print("==== Build CAGRA index ====")
    print(f"metric                   : {args.metric}")
    print(f"build_algo               : {args.build_algo}")
    print(f"graph_degree             : {args.graph_degree}")
    print(f"intermediate_graph_degree: {args.intermediate_graph_degree}")
    print(f"nn_descent_niter         : {args.nn_descent_niter}")

    index_params = make_index_params(args)

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()

    index = cagra.build(index_params, base_gpu)

    cp.cuda.Stream.null.synchronize()
    build_time = time.perf_counter() - t0

    print(f"graph build time [s]: {build_time:.6f}")

    print()
    print("==== Sweep search parameters ====")
    print(f"k           : {args.k}")
    print(f"search_algo : {args.algo}")
    print(f"search_width: {args.search_width}")
    print(f"team_size   : {args.team_size if args.team_size is not None else 'auto'}")
    print(f"itopk_list  : {itopk_list}")
    print(f"repeat      : {args.repeat}")

    results = []

    for itopk_size in itopk_list:
        search_params = make_search_params(args, itopk_size)

        team_size = get_team_size(search_params)
        cta_per_query = get_cta_per_query(
            algo=args.algo,
            search_width=args.search_width,
            itopk_size=itopk_size,
        )

        print()
        print(
            f"---- algo = {args.algo}, "
            f"itopk_size = {itopk_size}, "
            f"team_size = {team_size}, "
            f"cta_per_query = {cta_per_query} ----"
        )

        # ウォームアップ
        distances, neighbors = cagra.search(
            search_params,
            index,
            query_gpu,
            args.k,
        )
        cp.cuda.Stream.null.synchronize()

        run_times = []

        for r in range(args.repeat):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()

            distances, neighbors = cagra.search(
                search_params,
                index,
                query_gpu,
                args.k,
            )

            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - t0

            run_times.append(elapsed)
            print(f"run {r + 1}: {elapsed:.6f} s")

        avg_time = sum(run_times) / len(run_times)

        num_queries = query.shape[0]
        time_ms_per_query = avg_time / num_queries * 1000.0
        time_us_per_query = avg_time / num_queries * 1_000_000.0
        qps = num_queries / avg_time

        neighbors_cpu = cp.asnumpy(neighbors)
        rec = recall_at_k(neighbors_cpu, gt, args.k)

        print(f"avg search time for all queries [s]: {avg_time:.6f}")
        print(f"search time [ms/query]            : {time_ms_per_query:.6f}")
        print(f"search time [us/query]            : {time_us_per_query:.6f}")
        print(f"QPS [queries/sec]                 : {qps:.3f}")
        print(f"recall@{args.k}                     : {rec:.6f}")
        print(f"team_size                         : {team_size}")
        print(f"cta_per_query                     : {cta_per_query}")

        results.append(
            {
                "dataset": "SIFT1M",
                "algo": args.algo,
                "k": args.k,
                "metric": args.metric,
                "graph_degree": args.graph_degree,
                "intermediate_graph_degree": args.intermediate_graph_degree,
                "build_algo": args.build_algo,
                "nn_descent_niter": args.nn_descent_niter,
                "search_width": args.search_width,
                "itopk_size": itopk_size,
                "team_size": team_size,
                "cta_per_query": cta_per_query,
                "repeat": args.repeat,
                "num_base": base.shape[0],
                "num_queries": query.shape[0],
                "dimension": base.shape[1],
                "graph_build_time_s": build_time,
                "avg_search_time_s": avg_time,
                "time_ms_per_query": time_ms_per_query,
                "time_us_per_query": time_us_per_query,
                "qps": qps,
                f"recall@{args.k}": rec,
            }
        )

    if args.output is not None:
        print()
        print("==== Save results ====")

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fieldnames = list(results[0].keys())

        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"saved: {args.output}")

    print()
    print("==== Summary ====")

    for row in results:
        print(
            f"algo={row['algo']}, "
            f"itopk_size={row['itopk_size']:4d}, "
            f"team_size={row['team_size']}, "
            f"cta_per_query={row['cta_per_query']}, "
            f"time={row['time_us_per_query']:.3f} us/query, "
            f"time={row['time_ms_per_query']:.6f} ms/query, "
            f"QPS={row['qps']:.3f}, "
            f"recall@{args.k}={row[f'recall@{args.k}']:.6f}"
        )


if __name__ == "__main__":
    main()
