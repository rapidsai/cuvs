/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark/benchmark.h>
#include <iostream>

#include <raft/core/resource/cuda_stream.hpp>

#include "../../../../src/distance/fused_distance_nn.cuh"
#include "../../../../src/distance/unfused_distance_nn.cuh"

#include <raft/core/device_resources.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/norm_types.hpp>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>

using cuvs::distance::DistanceType;
using cuvs::distance::fusedDistanceNNMinReduce;
using cuvs::distance::pairwise_distance_gemm;
using cuvs::distance::reduce_min;
using cuvs::distance::unfusedDistanceNNMinReduce;

enum class AlgorithmType { gemm, unfused, fused };

__global__ void fill_int8(int8_t* buff, int len)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Fill the buffer with pseudo-random int8_t values using a simple LCG
  if (tid < len) {
    // Simple LCG: x_n+1 = (a * x_n + c) % m
    // Use tid as seed, constants chosen for decent distribution
    int seed  = tid * 1103515245 + 12345;
    buff[tid] = static_cast<int8_t>((seed >> 16) & 0xFF);
  }
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          AlgorithmType algo,
          bool sqrt,
          DistanceType metric>
void benchmark_distance_nn(benchmark::State& state)
{
  const int m = state.range(0);
  const int n = state.range(1);
  const int k = state.range(2);

  raft::device_resources handle;
  rmm::cuda_stream_view stream;

  stream = raft::resource::get_cuda_stream(handle);

  auto x      = raft::make_device_matrix<DataT, IdxT>(handle, m, k);
  auto y      = raft::make_device_matrix<DataT, IdxT>(handle, n, k);
  auto x_norm = raft::make_device_vector<AccT, IdxT>(handle, m);
  auto y_norm = raft::make_device_vector<AccT, IdxT>(handle, n);
  auto out    = raft::make_device_vector<OutT, IdxT>(handle, m);

  raft::random::RngState rng{1234};
  if constexpr (std::is_same_v<DataT, int8_t>) {
    fill_int8<<<1000, 256, 0, stream>>>(x.data_handle(), m * k);
    fill_int8<<<1000, 256, 0, stream>>>(y.data_handle(), n * k);
  } else {
    raft::random::uniform(handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
    raft::random::uniform(handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));
  }

  // Pre-compute norms
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    x_norm.data_handle(), x.data_handle(), k, m, stream);
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    y_norm.data_handle(), y.data_handle(), k, n, stream);

  // Calculate the workspace size
  // for fused it is m * sizeof(IdxT)
  // for unfused and gemm it is m * n * sizeof(AccT);

  size_t workspace_size = (algo == AlgorithmType::fused) ? m * sizeof(IdxT) : m * n * sizeof(AccT);

  raft::device_vector<char, size_t> workspace =
    raft::make_device_vector<char, size_t>(handle, workspace_size);

  // Warm up
  if constexpr (algo != AlgorithmType::fused) {
    unfusedDistanceNNMinReduce<DataT, AccT, OutT, IdxT>(handle,
                                                        out.data_handle(),
                                                        x.data_handle(),
                                                        y.data_handle(),
                                                        x_norm.data_handle(),
                                                        y_norm.data_handle(),
                                                        static_cast<IdxT>(m),
                                                        static_cast<IdxT>(n),
                                                        static_cast<IdxT>(k),
                                                        (AccT*)workspace.data_handle(),
                                                        sqrt,
                                                        true,
                                                        true,
                                                        metric,
                                                        float(0.0),
                                                        stream);
  }

  raft::matrix::fill(
    handle,
    raft::make_device_matrix_view(workspace.data_handle(), workspace_size, size_t(1)),
    char(0));
  if constexpr (std::is_same_v<OutT, raft::KeyValuePair<IdxT, AccT>>) {
    // OutT is a RAFT KeyValuePair
    raft::matrix::fill(handle, raft::make_device_matrix_view(out.data_handle(), m, 1), OutT{0, 0});
  } else {
    // OutT is a scalar type
    raft::matrix::fill(handle, raft::make_device_matrix_view(out.data_handle(), m, 1), OutT{0});
  }
  raft::resource::sync_stream(handle, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (auto _ : state) {
    cudaEventRecord(start, stream);
    if constexpr (algo == AlgorithmType::fused) {
      fusedDistanceNNMinReduce<DataT, OutT, IdxT>(out.data_handle(),
                                                  x.data_handle(),
                                                  y.data_handle(),
                                                  x_norm.data_handle(),
                                                  y_norm.data_handle(),
                                                  static_cast<IdxT>(m),
                                                  static_cast<IdxT>(n),
                                                  static_cast<IdxT>(k),
                                                  (void*)workspace.data_handle(),
                                                  sqrt,
                                                  true,
                                                  true,
                                                  metric,
                                                  0.0,
                                                  stream);
    }

    if constexpr (algo == AlgorithmType::unfused) {
      unfusedDistanceNNMinReduce<DataT, AccT, OutT, IdxT>(handle,
                                                          out.data_handle(),
                                                          x.data_handle(),
                                                          y.data_handle(),
                                                          x_norm.data_handle(),
                                                          y_norm.data_handle(),
                                                          static_cast<IdxT>(m),
                                                          static_cast<IdxT>(n),
                                                          static_cast<IdxT>(k),
                                                          (AccT*)workspace.data_handle(),
                                                          sqrt,
                                                          true,
                                                          true,
                                                          metric,
                                                          float(0.0),
                                                          stream);
    }

    if constexpr (algo == AlgorithmType::gemm) {
      pairwise_distance_gemm<DataT, AccT, OutT, IdxT>(handle,
                                                      (AccT*)workspace.data_handle(),
                                                      x.data_handle(),
                                                      y.data_handle(),
                                                      static_cast<IdxT>(m),
                                                      static_cast<IdxT>(n),
                                                      static_cast<IdxT>(k),
                                                      x_norm.data_handle(),
                                                      y_norm.data_handle(),
                                                      stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);  // wait until kernel is done
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    state.SetIterationTime(ms / 1000.0);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  state.counters["M"] = m;
  state.counters["N"] = n;
  state.counters["K"] = k;
  int64_t total_ops   = int64_t(2) * m * n * k;
  state.counters["FLOP/s"] =
    benchmark::Counter(total_ops, benchmark::Counter::kIsIterationInvariantRate);
}

template <typename IdxT>
static void register_configs(IdxT M, IdxT N, IdxT K)
{
  benchmark::internal::Benchmark* bench;

  // In the following instances IdxT is int64_t, it does not seem to have much impact on the
  // performance OutT is always <IdxT, AccT>

  constexpr bool sqrt         = false;
  constexpr DistanceType dist = DistanceType::L2Expanded;
  // Method: Fused, DataT: float, AccT: float
  bench = benchmark::RegisterBenchmark("fused/float/float",
                                       benchmark_distance_nn<float,
                                                             float,
                                                             raft::KeyValuePair<int64_t, float>,
                                                             int64_t,
                                                             AlgorithmType::fused,
                                                             sqrt,
                                                             dist>);
  bench->Args({M, N, K})->UseManualTime();

  // Method: unfused, DataT: float, AccT: float
  bench = benchmark::RegisterBenchmark("unfused/float/float",
                                       benchmark_distance_nn<float,
                                                             float,
                                                             raft::KeyValuePair<int64_t, float>,
                                                             int64_t,
                                                             AlgorithmType::unfused,
                                                             sqrt,
                                                             dist>);
  bench->Args({M, N, K})->UseManualTime();

  // Method: gemm, DataT: float, AccT: float
  bench = benchmark::RegisterBenchmark("gemm/float/float",
                                       benchmark_distance_nn<float,
                                                             float,
                                                             raft::KeyValuePair<int64_t, float>,
                                                             int64_t,
                                                             AlgorithmType::gemm,
                                                             sqrt,
                                                             dist>);
  bench->Args({M, N, K})->UseManualTime();

  // Method: unfused, DataT: half, AccT: float
  bench = benchmark::RegisterBenchmark("unfused/half/float",
                                       benchmark_distance_nn<half,
                                                             float,
                                                             raft::KeyValuePair<int64_t, float>,
                                                             int64_t,
                                                             AlgorithmType::unfused,
                                                             sqrt,
                                                             dist>);
  bench->Args({M, N, K})->UseManualTime();

  // Method: gemm, DataT: half, AccT: float
  bench = benchmark::RegisterBenchmark("gemm/half/float",
                                       benchmark_distance_nn<half,
                                                             float,
                                                             raft::KeyValuePair<int64_t, float>,
                                                             int64_t,
                                                             AlgorithmType::gemm,
                                                             sqrt,
                                                             dist>);
  bench->Args({M, N, K})->UseManualTime();

  // Method: unfused, DataT: half, AccT: half
  bench = benchmark::RegisterBenchmark("unfused/half/half",
                                       benchmark_distance_nn<half,
                                                             half,
                                                             raft::KeyValuePair<int64_t, half>,
                                                             int64_t,
                                                             AlgorithmType::unfused,
                                                             sqrt,
                                                             dist>);
  bench->Args({M, N, K})->UseManualTime();

  // Method: gemm, DataT: half, AccT: half
  bench = benchmark::RegisterBenchmark("gemm/half/half",
                                       benchmark_distance_nn<half,
                                                             half,
                                                             raft::KeyValuePair<int64_t, half>,
                                                             int64_t,
                                                             AlgorithmType::gemm,
                                                             sqrt,
                                                             dist>);
  bench->Args({M, N, K})->UseManualTime();

  // Method: unfused, DataT: int8_t, AccT: int32_t
  bench = benchmark::RegisterBenchmark("unfused/int8_t/int32_t",
                                       benchmark_distance_nn<int8_t,
                                                             int32_t,
                                                             raft::KeyValuePair<int64_t, int32_t>,
                                                             int64_t,
                                                             AlgorithmType::unfused,
                                                             sqrt,
                                                             dist>);
  bench->Args({M, N, K})->UseManualTime();

  // Method: gemm, DataT: int8_t, AccT: int32_t
  bench = benchmark::RegisterBenchmark("gemm/int8_t/int32_t",
                                       benchmark_distance_nn<int8_t,
                                                             int32_t,
                                                             raft::KeyValuePair<int64_t, int32_t>,
                                                             int64_t,
                                                             AlgorithmType::gemm,
                                                             sqrt,
                                                             dist>);
  bench->Args({M, N, K})->UseManualTime();
}

int main(int argc, char** argv)
{
  using IdxT = int64_t;
  IdxT M     = 0;
  IdxT N     = 0;
  IdxT K     = 0;

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-M") == 0) {
      M = std::stoi(argv[i + 1]);
      for (int j = i; j < argc - 2; j++) {
        argv[j] = argv[j + 2];
      }
      argc -= 2;
      i -= 2;
    } else if (strcmp(argv[i], "-N") == 0) {
      N = std::stoi(argv[i + 1]);
      for (int j = i; j < argc - 2; j++) {
        argv[j] = argv[j + 2];
      }
      argc -= 2;
      i -= 2;
    } else if (strcmp(argv[i], "-K") == 0) {
      K = std::stoi(argv[i + 1]);
      for (int j = i; j < argc - 2; j++) {
        argv[j] = argv[j + 2];
      }
      argc -= 2;
      i -= 2;
    }
  }

  if (M == 0 && N == 0 && K == 0) {
    register_configs<IdxT>(16 * 1024, 4 * 1024, 128);
    register_configs<IdxT>(16 * 1024, 4 * 1024, 64);
    register_configs<IdxT>(8 * 1024, 2 * 1024, 64);
    register_configs<IdxT>(4 * 1024, 1024, 64);
  } else {
    register_configs<IdxT>(M, N, K);
  }

  // Initialize benchmark
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
