/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Benchmark: full IVF-PQ build path and compare brute-force vs CAGRA cluster assignment.
 * For each scenario we run one benchmark that does both a brute and a CAGRA full build,
 * reports brute_ms and cagra_ms, and speedup = brute_ms / cagra_ms (>1 means CAGRA faster).
 *
 * n_lists = number of cluster centroids. We use at least 5 vectors per cluster
 * (n_vectors >= 5 * n_lists). "Time" = wall time for one iteration (brute + CAGRA build).
 *
 * Full build includes kmeans, PQ codebook training, and assignment. Threshold 200K in
 * ivf_pq_build.cuh was chosen from assignment-only benchmarks.
 */
#include <benchmark/benchmark.h>

#include <cuvs/neighbors/ivf_pq.hpp>

#include <chrono>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

namespace {

void init_random_dataset(raft::resources const& handle,
                         float* data,
                         int64_t n_rows,
                         int64_t dim)
{
  raft::random::RngState rng(12345ULL);
  raft::random::uniform(handle, rng, data, n_rows * dim, float(-1), float(1));
  raft::resource::sync_stream(handle);
}

}  // namespace

static void BM_IVFPQ_Build_Speedup(benchmark::State& state)
{
  int64_t n_rows   = static_cast<int64_t>(state.range(0));
  uint32_t n_lists = static_cast<uint32_t>(state.range(1));
  int64_t dim      = static_cast<int64_t>(state.range(2));

  raft::device_resources handle;
  rmm::device_uvector<float> dataset(static_cast<size_t>(n_rows) * static_cast<size_t>(dim),
                                     raft::resource::get_cuda_stream(handle));
  init_random_dataset(handle, dataset.data(), n_rows, dim);

  cuvs::neighbors::ivf_pq::index_params params_brute;
  params_brute.n_lists                        = n_lists;
  params_brute.kmeans_n_iters                 = 3;
  params_brute.kmeans_trainset_fraction       = 0.2;
  params_brute.add_data_on_build              = true;
  params_brute.metric                         = cuvs::distance::DistanceType::L2Expanded;
  params_brute.use_ann_for_extend = false;

  cuvs::neighbors::ivf_pq::index_params params_cagra = params_brute;
  params_cagra.use_ann_for_extend        = true;

  raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));

  auto dataset_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    dataset.data(), n_rows, dim);

  double total_brute_ms = 0.0, total_cagra_ms = 0.0;
  int64_t iterations = 0;

  for (auto _ : state) {
    auto start = std::chrono::steady_clock::now();
    auto idx_brute = cuvs::neighbors::ivf_pq::build(handle, params_brute, dataset_view);
    benchmark::DoNotOptimize(idx_brute.size());
    raft::resource::sync_stream(handle);
    auto end = std::chrono::steady_clock::now();
    total_brute_ms += 1e-6 * std::chrono::duration<double, std::nano>(end - start).count();

    start = std::chrono::steady_clock::now();
    auto idx_cagra = cuvs::neighbors::ivf_pq::build(handle, params_cagra, dataset_view);
    benchmark::DoNotOptimize(idx_cagra.size());
    raft::resource::sync_stream(handle);
    end = std::chrono::steady_clock::now();
    total_cagra_ms += 1e-6 * std::chrono::duration<double, std::nano>(end - start).count();

    ++iterations;
  }

  if (total_cagra_ms > 0) {
    state.counters["speedup"] = total_brute_ms / total_cagra_ms;
  }
  state.counters["brute_ms"] = benchmark::Counter(total_brute_ms,
                                                  benchmark::Counter::kAvgIterations);
  state.counters["cagra_ms"] = benchmark::Counter(total_cagra_ms,
                                                  benchmark::Counter::kAvgIterations);
}

constexpr int64_t kDim = 128;

// At least 5 vectors per cluster (n_vectors = 5 * n_lists). One row per config: brute_ms, cagra_ms, speedup.

// 1. 64K centroids, 5 vecs/cluster
BENCHMARK(BM_IVFPQ_Build_Speedup)
  ->Args({327680, 65536, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});

// 2. 200K centroids, 5 vecs/cluster
BENCHMARK(BM_IVFPQ_Build_Speedup)
  ->Args({1000000, 200000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});

// 3. 400K centroids, 5 vecs/cluster
BENCHMARK(BM_IVFPQ_Build_Speedup)
  ->Args({2000000, 400000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});

// 4. 600K centroids, 5 vecs/cluster
BENCHMARK(BM_IVFPQ_Build_Speedup)
  ->Args({3000000, 600000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});

// 5. 800K centroids, 5 vecs/cluster
BENCHMARK(BM_IVFPQ_Build_Speedup)
  ->Args({4000000, 800000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});

// 6. 1M centroids, 5 vecs/cluster
BENCHMARK(BM_IVFPQ_Build_Speedup)
  ->Args({5000000, 1000000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});

BENCHMARK_MAIN();
