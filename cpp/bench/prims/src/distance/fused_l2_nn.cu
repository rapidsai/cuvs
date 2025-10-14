#include "common.cuh"
#include <benchmark/benchmark.h>
#include <iostream>

#include <raft/core/resource/cuda_stream.hpp>

#include "../../../../src/distance/fused_distance_nn.cuh"
#include "../../../../src/distance/unfused_distance_nn.cuh"

#include <raft/linalg/norm.cuh>
#include <raft/linalg/norm_types.hpp>
#include <raft/random/rng.cuh>

using cuvs::distance::DistanceType;
using cuvs::distance::fusedDistanceNNMinReduce;
using cuvs::distance::reduce_min;
using cuvs::distance::unfused_distance_nn;
using cuvs::distance::unfusedDistanceNNMinReduce;

enum class AlgorithmType { gemm, unfused, fused };

template <typename DataT, typename AccT, typename OutT, typename IdxT, AlgorithmType algo>
void benchmark_fusedl2nn(benchmark::State& state)
{
  const int m               = state.range(0);
  const int n               = state.range(1);
  const int k               = state.range(2);
  const bool sqrt           = state.range(3);
  const DistanceType metric = DistanceType(state.range(4));

  raft::device_resources handle;
  rmm::cuda_stream_view stream;

  stream = raft::resource::get_cuda_stream(handle);

  auto x       = raft::make_device_matrix<DataT, IdxT>(handle, m, k);
  auto y       = raft::make_device_matrix<DataT, IdxT>(handle, n, k);
  auto x_norm  = raft::make_device_vector<AccT, IdxT>(handle, m);
  auto y_norm  = raft::make_device_vector<AccT, IdxT>(handle, n);
  auto out     = raft::make_device_vector<OutT, IdxT>(handle, m);
  auto out_ref = raft::make_device_vector<OutT, IdxT>(handle, m);

  raft::random::RngState rng{1234};
  raft::random::uniform(handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
  raft::random::uniform(handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));

  // Pre-compute norms
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    x_norm.data_handle(), x.data_handle(), k, m, stream);
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    y_norm.data_handle(), y.data_handle(), k, n, stream);

  // Calculate the workspace size
  // for fused it is m * sizeof(IdxT)
  // for unfused, gemm and tensor it is m * n * sizeof(AccT);

  size_t workspace_size = (algo == AlgorithmType::fused) ? m * sizeof(IdxT) : m * n * sizeof(AccT);

  raft::device_vector<char, IdxT> workspace =
    raft::make_device_vector<char, IdxT>(handle, workspace_size);

  CudaEventTimer timer(stream);

  // Reference calculation
  ref_l2nn_api<DataT, AccT, OutT, IdxT>(
    out_ref.data_handle(), x.data_handle(), y.data_handle(), m, n, k, sqrt, metric, stream);

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
                                                        0.0,
                                                        stream);
  }

  RAFT_CUDA_TRY(cudaMemsetAsync(workspace.data_handle(), 0, workspace_size, stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(out.data_handle(), 0, m * sizeof(OutT), stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  // launch_memcpy();
  timer.start();
  for (auto _ : state) {
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
                                                          0.0,
                                                          stream);
    }

    if constexpr (algo == AlgorithmType::gemm) {
      unfusedDistanceNNMinReduce<DataT, AccT, OutT, IdxT, false>(handle,
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
                                                                 0.0,
                                                                 stream);
    }
  }
  timer.stop();
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  if constexpr (algo == AlgorithmType::gemm) {
    if (metric == DistanceType::L2Expanded) {
      reduce_min<DataT, AccT, OutT, IdxT, DistanceType::L2Expanded>(out.data_handle(),
                                                                    (AccT*)workspace.data_handle(),
                                                                    x_norm.data_handle(),
                                                                    y_norm.data_handle(),
                                                                    m,
                                                                    n,
                                                                    stream,
                                                                    sqrt,
                                                                    true);
    } else if (metric == DistanceType::L2SqrtExpanded) {
      reduce_min<DataT, AccT, OutT, IdxT, DistanceType::L2SqrtExpanded>(
        out.data_handle(),
        (AccT*)workspace.data_handle(),
        x_norm.data_handle(),
        y_norm.data_handle(),
        m,
        n,
        stream,
        sqrt,
        true);
    } else if (metric == DistanceType::CosineExpanded) {
      reduce_min<DataT, AccT, OutT, IdxT, DistanceType::CosineExpanded>(
        out.data_handle(),
        (AccT*)workspace.data_handle(),
        x_norm.data_handle(),
        y_norm.data_handle(),
        m,
        n,
        stream,
        sqrt,
        true);
    }
  }

  ComparisonSummary* global_summary = nullptr;
  RAFT_CUDA_TRY(cudaMallocManaged(&global_summary, sizeof(ComparisonSummary)));
  global_summary->init();

  vector_compare(global_summary, out_ref.data_handle(), out.data_handle(), m, stream);
  global_summary->print();

  state.counters["M"]         = m;
  state.counters["N"]         = n;
  state.counters["K"]         = k;
  state.counters["iter_time"] = timer.elapsed_seconds() / state.iterations();
  state.counters["FLOP/s"] =
    (int64_t(state.iterations()) * 2 * m * n * k) / timer.elapsed_seconds();
  state.counters["total_missed"] = global_summary->n_misses;
  state.counters["max_diff"]     = global_summary->max_diff;
  /*
       int64_t num_flops = int64_t(2) * m * n * k;

       int64_t read_elts = int64_t(n) * k + m * k;
       int64_t write_elts = m;

       state.counters["M"] = m;
       state.counters["N"] = n;
       state.counters["K"] = k;

       state.counters["FLOP/s"] = benchmark::Counter(
         num_flops, benchmark::Counter::kIsIterationInvariantRate,
     benchmark::Counter::OneK::kIs1000);

       state.counters["BW Wr"] = benchmark::Counter(write_elts * sizeof(OutT),
                                                   benchmark::Counter::kIsIterationInvariantRate,
                                                   benchmark::Counter::OneK::kIs1000);
       state.counters["BW Rd"] = benchmark::Counter(read_elts * sizeof(DataT),
                                                   benchmark::Counter::kIsIterationInvariantRate,
                                                   benchmark::Counter::OneK::kIs1000);*/
}

template <typename IdxT>
static void CustomArguments(benchmark::internal::Benchmark* b)
{
  /*constexpr int K             = 1024;
  std::vector<int64_t> m_list = {4 * K, 8 * K};
  std::vector<int64_t> n_list = {4 * K, 8 * K, 16 * K};
  // std::vector<int64_t> k_list = {128, 512, 1024, 1536};
  std::vector<int64_t> k_list = {128, 256, 512};
  for (auto k : k_list) {
    for (auto m : m_list) {
      for (auto n : n_list) {
        b->Args({m, n, k});
      }
    }
  }*/
  b->Args({65536, 256, 128, true, DistanceType::L2Expanded});
  b->Args({65536, 256, 128, false, DistanceType::CosineExpanded});
  // b->Args({65536, 10000, 768, false});
}

int main(int argc, char** argv)
{
  benchmark::internal::Benchmark* bench;
  /*int64_t M = 1024;
  int64_t N = 1024;
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--M") == 0) {
      M = std::stoi(argv[i + 1]);
      for(int j = i; j < argc - 2; j++) {
        argv[j] = argv[j + 2];
      }
      argc -= 2;
      i -= 2;
    } else if (strcmp(argv[i], "--N") == 0) {
      N = std::stoi(argv[i + 1]);
      for(int j = i; j < argc - 2; j++) {
        argv[j] = argv[j + 2];
      }
      argc -= 2;
      i -= 2;
    } else if (strcmp(argv[i], "--method") == 0) {
      N = std::stoi(argv[i + 1]);
      for(int j = i; j < argc - 2; j++) {
        argv[j] = argv[j + 2];
      }
      argc -= 2;
      i -= 2;
    }
  }*/

  // fused path
  /*bench = benchmark::RegisterBenchmark(
    "fusedl2nn/float/float/<int64_t, float>",
    benchmark_fusedl2nn<float, float, raft::KeyValuePair<int64_t, float>, int64_t,
  AlgorithmType::fused>); bench->Apply(CustomArguments<int64_t>);*/

  bench = benchmark::RegisterBenchmark("unfused/float/int/<int, float>",
                                       benchmark_fusedl2nn<float,
                                                           float,
                                                           raft::KeyValuePair<int64_t, float>,
                                                           int64_t,
                                                           AlgorithmType::gemm>);
  bench->Apply(CustomArguments<int64_t>);
  // unfused path
  // half -> half
  /*bench = benchmark::RegisterBenchmark("unfused/half/int/<int, half>",
                                       benchmark_fusedl2nn<half,
                                                           half,
                                                           raft::KeyValuePair<int64_t, half>,
                                                           int64_t,
                                                           AlgorithmType::unfused>);
  bench->Apply(CustomArguments<int64_t>);

  // half -> float
  bench = benchmark::RegisterBenchmark("unfused/half/int/<int, float>",
                                       benchmark_fusedl2nn<half,
                                                           float,
                                                           raft::KeyValuePair<int64_t, float>,
                                                           int64_t,
                                                           AlgorithmType::unfused>);
  bench->Apply(CustomArguments<int64_t>);

  // float -> float
  bench = benchmark::RegisterBenchmark("unfused/float/int/<int, float>",
                                       benchmark_fusedl2nn<float,
                                                           float,
                                                           raft::KeyValuePair<int64_t, float>,
                                                           int64_t,
                                                           AlgorithmType::unfused>);

  bench->Apply(CustomArguments<int64_t>);

  // just gemm
  // half -> half
  bench = benchmark::RegisterBenchmark(
    "gemm/half/int/<int, half>",
    benchmark_fusedl2nn<half, half, raft::KeyValuePair<int64_t, half>, int64_t,
  AlgorithmType::gemm>); bench->Apply(CustomArguments<int64_t>);
  // half -> float
  bench = benchmark::RegisterBenchmark(
    "gemm/half/int/<int, float>",
    benchmark_fusedl2nn<half, float, raft::KeyValuePair<int64_t, float>, int64_t,
  AlgorithmType::gemm>); bench->Apply(CustomArguments<int64_t>);
  // float -> float
  bench = benchmark::RegisterBenchmark(
    "gemm/float/int/<int, float>",
    benchmark_fusedl2nn<float, float, raft::KeyValuePair<int64_t, float>, int64_t,
  AlgorithmType::gemm>); bench->Apply(CustomArguments<int64_t>);*/

  // Initialize benchmark
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
