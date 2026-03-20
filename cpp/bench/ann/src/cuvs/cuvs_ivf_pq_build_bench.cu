/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IVF-PQ benchmarks (two scenarios):
 *
 * 1) Full build — k-means *fit* ANN vs brute (shifting centroids). Both runs use brute cluster
 *    assignment for extend/add_data_on_build (`use_ann_for_extend = false`) so the comparison
 *    isolates the balanced k-means training path, not extend-time assignment.
 *
 * 2) Extend only — brute vs CAGRA for assigning vectors to *fixed* trained centroids. Empty
 *    trained indices are restored each iteration via deserialize (setup not timed). This matches
 *    the assumption that centroids do not move during extend, unlike k-means fit.
 */
#include <benchmark/benchmark.h>

#include <cuvs/neighbors/ivf_pq.hpp>

#include <chrono>
#include <memory>
#include <sstream>
#include <string>
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

/** Serialize an empty trained index to a string for benchmark reset. */
std::string serialize_index_blob(raft::resources const& handle,
                                   cuvs::neighbors::ivf_pq::index<int64_t> const& index)
{
  std::ostringstream os(std::ios::binary);
  cuvs::neighbors::ivf_pq::serialize(handle, os, index);
  raft::resource::sync_stream(handle);
  return os.str();
}

void deserialize_index_from_blob(raft::resources const& handle,
                                 std::string const& blob,
                                 cuvs::neighbors::ivf_pq::index<int64_t>* index)
{
  std::istringstream is(blob, std::ios::binary);
  cuvs::neighbors::ivf_pq::deserialize(handle, is, index);
  raft::resource::sync_stream(handle);
}

}  // namespace

/** Full IVF-PQ build: compare brute vs ANN for balanced k-means *fit* only. */
static void BM_IVFPQ_Build_KMeansFit_Speedup(benchmark::State& state)
{
  int64_t n_rows   = static_cast<int64_t>(state.range(0));
  uint32_t n_lists = static_cast<uint32_t>(state.range(1));
  int64_t dim      = static_cast<int64_t>(state.range(2));

  raft::device_resources handle;
  rmm::device_uvector<float> dataset(static_cast<size_t>(n_rows) * static_cast<size_t>(dim),
                                     raft::resource::get_cuda_stream(handle));
  init_random_dataset(handle, dataset.data(), n_rows, dim);

  cuvs::neighbors::ivf_pq::index_params params_bf_fit;
  params_bf_fit.n_lists                  = n_lists;
  params_bf_fit.kmeans_n_iters           = 3;
  params_bf_fit.kmeans_trainset_fraction = 0.2;
  params_bf_fit.add_data_on_build        = true;
  params_bf_fit.metric                   = cuvs::distance::DistanceType::L2Expanded;
  params_bf_fit.use_ann_for_extend       = false;
  params_bf_fit.use_ann_for_fit          = false;

  cuvs::neighbors::ivf_pq::index_params params_ann_fit = params_bf_fit;
  params_ann_fit.use_ann_for_fit                       = true;

  raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));

  auto dataset_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    dataset.data(), n_rows, dim);

  double total_bf_ms = 0.0, total_ann_ms = 0.0;

  for (auto _ : state) {
    auto start = std::chrono::steady_clock::now();
    auto idx_bf = cuvs::neighbors::ivf_pq::build(handle, params_bf_fit, dataset_view);
    benchmark::DoNotOptimize(idx_bf.size());
    raft::resource::sync_stream(handle);
    auto end = std::chrono::steady_clock::now();
    total_bf_ms += 1e-6 * std::chrono::duration<double, std::nano>(end - start).count();

    start = std::chrono::steady_clock::now();
    auto idx_ann = cuvs::neighbors::ivf_pq::build(handle, params_ann_fit, dataset_view);
    benchmark::DoNotOptimize(idx_ann.size());
    raft::resource::sync_stream(handle);
    end = std::chrono::steady_clock::now();
    total_ann_ms += 1e-6 * std::chrono::duration<double, std::nano>(end - start).count();
  }

  if (total_ann_ms > 0) {
    state.counters["speedup_fit"] = total_bf_ms / total_ann_ms;
  }
  state.counters["bf_fit_ms"] =
    benchmark::Counter(total_bf_ms, benchmark::Counter::kAvgIterations);
  state.counters["ann_fit_ms"] =
    benchmark::Counter(total_ann_ms, benchmark::Counter::kAvgIterations);
}

/**
 * extend() only: brute vs CAGRA cluster assignment with fixed centroids (trained empty index).
 * Deserialize is not timed.
 */
static void BM_IVFPQ_Extend_ClusterAssign_Speedup(benchmark::State& state)
{
  int64_t n_rows   = static_cast<int64_t>(state.range(0));
  uint32_t n_lists = static_cast<uint32_t>(state.range(1));
  int64_t dim      = static_cast<int64_t>(state.range(2));

  raft::device_resources handle;
  rmm::device_uvector<float> dataset(static_cast<size_t>(n_rows) * static_cast<size_t>(dim),
                                     raft::resource::get_cuda_stream(handle));
  init_random_dataset(handle, dataset.data(), n_rows, dim);

  cuvs::neighbors::ivf_pq::index_params params_common;
  params_common.n_lists                  = n_lists;
  params_common.kmeans_n_iters           = 3;
  params_common.kmeans_trainset_fraction = 0.2;
  params_common.add_data_on_build        = false;
  params_common.metric                   = cuvs::distance::DistanceType::L2Expanded;
  params_common.use_ann_for_fit          = false;

  cuvs::neighbors::ivf_pq::index_params params_bf_ext  = params_common;
  params_bf_ext.use_ann_for_extend                     = false;
  cuvs::neighbors::ivf_pq::index_params params_cagra_ext = params_common;
  params_cagra_ext.use_ann_for_extend                   = true;

  raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));

  auto dataset_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    dataset.data(), n_rows, dim);

  auto idx_bf  = cuvs::neighbors::ivf_pq::build(handle, params_bf_ext, dataset_view);
  auto idx_cag = cuvs::neighbors::ivf_pq::build(handle, params_cagra_ext, dataset_view);
  std::string blob_bf  = serialize_index_blob(handle, idx_bf);
  std::string blob_cag = serialize_index_blob(handle, idx_cag);

  double total_bf_ms = 0.0, total_cagra_ms = 0.0;

  for (auto _ : state) {
    state.PauseTiming();
    cuvs::neighbors::ivf_pq::index<int64_t> empty_bf(handle);
    cuvs::neighbors::ivf_pq::index<int64_t> empty_cag(handle);
    deserialize_index_from_blob(handle, blob_bf, &empty_bf);
    deserialize_index_from_blob(handle, blob_cag, &empty_cag);
    state.ResumeTiming();

    auto start = std::chrono::steady_clock::now();
    cuvs::neighbors::ivf_pq::extend(handle, dataset_view, std::nullopt, &empty_bf);
    raft::resource::sync_stream(handle);
    auto end = std::chrono::steady_clock::now();
    total_bf_ms += 1e-6 * std::chrono::duration<double, std::nano>(end - start).count();

    start = std::chrono::steady_clock::now();
    cuvs::neighbors::ivf_pq::extend(handle, dataset_view, std::nullopt, &empty_cag);
    raft::resource::sync_stream(handle);
    end = std::chrono::steady_clock::now();
    total_cagra_ms += 1e-6 * std::chrono::duration<double, std::nano>(end - start).count();
  }

  if (total_cagra_ms > 0) {
    state.counters["speedup_extend"] = total_bf_ms / total_cagra_ms;
  }
  state.counters["bf_extend_ms"] =
    benchmark::Counter(total_bf_ms, benchmark::Counter::kAvgIterations);
  state.counters["cagra_extend_ms"] =
    benchmark::Counter(total_cagra_ms, benchmark::Counter::kAvgIterations);
}

constexpr int64_t kDim = 128;

// Full build: k-means fit brute vs ANN (same problem sizes as before).
BENCHMARK(BM_IVFPQ_Build_KMeansFit_Speedup)
  ->Args({327680, 65536, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Build_KMeansFit_Speedup)
  ->Args({1000000, 200000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Build_KMeansFit_Speedup)
  ->Args({2000000, 400000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Build_KMeansFit_Speedup)
  ->Args({3000000, 600000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Build_KMeansFit_Speedup)
  ->Args({4000000, 800000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Build_KMeansFit_Speedup)
  ->Args({5000000, 1000000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});

// extend(): fixed-centroid assignment brute vs CAGRA.
BENCHMARK(BM_IVFPQ_Extend_ClusterAssign_Speedup)
  ->Args({327680, 65536, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Extend_ClusterAssign_Speedup)
  ->Args({1000000, 200000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Extend_ClusterAssign_Speedup)
  ->Args({2000000, 400000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Extend_ClusterAssign_Speedup)
  ->Args({3000000, 600000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Extend_ClusterAssign_Speedup)
  ->Args({4000000, 800000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});
BENCHMARK(BM_IVFPQ_Extend_ClusterAssign_Speedup)
  ->Args({5000000, 1000000, kDim})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ArgNames({"n_vectors", "n_lists", "dim"});

BENCHMARK_MAIN();
