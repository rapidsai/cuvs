#include <benchmark/benchmark.h>
#include <iostream>
#include "common.cuh"
#include "cublas_sample.cu"
#include "tensor_core.cu"

#include <raft/core/resource/cuda_stream.hpp>

#include "../../../../src/distance/fused_distance_nn.cuh"

#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>

  template <typename DataT>
  __global__ void rescale(DataT* arr, int scale_factor, int len) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < len; i+=blockDim.x*gridDim.x) {
      arr[i] = hrint(arr[i]*DataT(scale_factor));
    }
  }

  enum class AlgorithmType {
    gemm,
    gemm_reduce,
    fused,
    tensor
  };

  template <typename DataT, typename AccT, typename OutT, typename IdxT, AlgorithmType algo>
  void benchmark_fusedl2nn(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);
    const int k = state.range(2);

    raft::device_resources handle;
    rmm::cuda_stream_view stream;

    if constexpr (std::is_scalar<OutT>()) {
      static_assert(std::is_same<OutT, AccT>::value, "When OutT is of scalar type, OutT and AccT must be of same type");
    } else {
      static_assert(is_pair<OutT, IdxT, AccT>(), "When OutT is RAFT key value pair, it must be of raft::KeyValuePair<IdxT, AccT> type");
    }

    stream = raft::resource::get_cuda_stream(handle);

    auto x     = raft::make_device_matrix<DataT, IdxT>(handle, m, k);
    auto y     = raft::make_device_matrix<DataT, IdxT>(handle, n, k);
    auto x_norm = raft::make_device_vector<AccT, IdxT>(handle, m);
    auto y_norm = raft::make_device_vector<AccT, IdxT>(handle, n);
    auto out   = raft::make_device_vector<OutT, IdxT>(handle, m);
    auto out_ref = raft::make_device_vector<OutT, IdxT>(handle, m);


    raft::random::RngState rng{1234};
    raft::random::uniform(
       handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
    raft::random::uniform(
       handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));

    //CHECK_CUDA(cudaDeviceSynchronize());
    //rescale<DataT><<<(m*k)/128, 128, 0, stream>>>(x.data_handle(), 4, m*k);
    //rescale<DataT><<<(n*k)/128, 128, 0, stream>>>(y.data_handle(), 4, n*k);

    // Pre-compute norms
    raft::linalg::rowNorm(x_norm.data_handle(),
                          x.data_handle(),
                          k,
                          m,
                          raft::linalg::L2Norm,
                          true,
                          stream);
    raft::linalg::rowNorm(y_norm.data_handle(),
                          y.data_handle(),
                          k,
                          n,
                          raft::linalg::L2Norm,
                          true,
                          stream);

    // Calculate the workspace size
    // for fused it is m * sizeof(IdxT)
    // for gemm_reduce, gemm and tensor it is m * n * sizeof(AccT);
    size_t workspace_size = m * n * sizeof(AccT) > n * sizeof(IdxT) ? m * n * sizeof(AccT) : n * sizeof(IdxT);

    raft::device_vector<char, IdxT> workspace = raft::make_device_vector<char, IdxT>(handle, workspace_size);

    CudaEventTimer timer(stream);

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));

    // Reference calculation
    ref_l2nn_api<DataT, AccT, OutT, IdxT>(out_ref.data_handle(), x.data_handle(), y.data_handle(), m, n, k, stream);

    // Warm up
    cublas_l2nn<DataT, AccT, OutT, IdxT, false, false>(out.data_handle(), x.data_handle(),
          y.data_handle(), m, n, k, x_norm.data_handle(), y_norm.data_handle(), (AccT*)workspace.data_handle(), cublas_handle, stream);


    CHECK_CUDA(cudaMemsetAsync(workspace.data_handle(), 0, workspace_size, stream));
    CHECK_CUDA(cudaMemsetAsync(out.data_handle(), 0, m * sizeof(OutT)));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    //launch_memcpy();
    timer.start();
    for (auto _ : state) {
      if constexpr (algo == AlgorithmType::fused) {
        cuvs::distance::fusedDistanceNNMinReduce<DataT, OutT, IdxT>(out.data_handle(),
                                                                    x.data_handle(),
                                                                    y.data_handle(),
                                                                    x_norm.data_handle(),
                                                                    y_norm.data_handle(),
                                                                    static_cast<IdxT>(m),
                                                                    static_cast<IdxT>(n),
                                                                    static_cast<IdxT>(k),
                                                                    (void*)workspace.data_handle(),
                                                                    false,
                                                                    true,
                                                                    true,
                                                                    cuvs::distance::DistanceType::L2Expanded,
                                                                    0.0,
                                                                    stream);
      }

      if constexpr (algo == AlgorithmType::gemm_reduce) {
        cublas_l2nn<DataT, AccT, OutT, IdxT, false, false>(out.data_handle(), x.data_handle(),
          y.data_handle(), m, n, k, x_norm.data_handle(), y_norm.data_handle(), (AccT*)workspace.data_handle(), cublas_handle, stream);
      }

      if constexpr (algo == AlgorithmType::gemm) {
        cublas_l2nn<DataT, AccT, OutT, IdxT, false, true>(out.data_handle(), x.data_handle(),
          y.data_handle(), m, n, k, x_norm.data_handle(), y_norm.data_handle(), (AccT*)workspace.data_handle(), cublas_handle, stream);
      }

      if constexpr (algo == AlgorithmType::tensor) {
        tensor_l2nn<DataT, AccT, DataT, IdxT>((AccT*)workspace.data_handle(), x.data_handle(),
          y.data_handle(), m, n, k, x_norm.data_handle(), y_norm.data_handle(), stream);
      }
    }
    timer.stop();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if constexpr (algo == AlgorithmType::gemm || algo == AlgorithmType::tensor) {
      reduce_min<DataT, AccT, OutT, IdxT>(out.data_handle(),
                                    (AccT*)workspace.data_handle(),
                                    x_norm.data_handle(),
                                    y_norm.data_handle(),
                                    m, n, stream);
    }

    ComparisonSummary* global_summary;
    CHECK_CUDA(cudaMallocManaged(&global_summary, sizeof(ComparisonSummary)));
    global_summary->init();

    vector_compare(global_summary, out_ref.data_handle(), out.data_handle(), m, stream);
    // global_summary->print();

    state.counters["M"] = m;
    state.counters["N"] = n;
    state.counters["K"] = k;
    state.counters["iter_time"] =  timer.elapsed_seconds() / state.iterations();
    state.counters["FLOP/s"] =  (int64_t(state.iterations()) * 2 * m * n * k) / timer.elapsed_seconds();
    state.counters["total_missed"] = global_summary->n_misses;
    state.counters["max_diff"] = global_summary->max_diff;
/*
     int64_t num_flops = int64_t(2) * m * n * k;

     int64_t read_elts = int64_t(n) * k + m * k;
     int64_t write_elts = m;

     state.counters["M"] = m;
     state.counters["N"] = n;
     state.counters["K"] = k;

     state.counters["FLOP/s"] = benchmark::Counter(
       num_flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);

     state.counters["BW Wr"] = benchmark::Counter(write_elts * sizeof(OutT),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);
     state.counters["BW Rd"] = benchmark::Counter(read_elts * sizeof(DataT),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);*/
   }

template <typename IdxT>
static void CustomArguments(benchmark::internal::Benchmark* b) {

  constexpr int K = 1024;
  std::vector<int64_t> m_list = {4*K, 8*K};
  std::vector<int64_t> n_list = {4*K, 8*K, 16*K};
  //std::vector<int64_t> k_list = {128, 512, 1024, 1536};
  std::vector<int64_t> k_list = {128, 256, 512};
  for (auto k : k_list) {
    for (auto m : m_list) {
      for (auto n : n_list) {
        b->Args({m, n, k});
      }
    }
  }
  //b->Args({256*128, 128*128, 128});
}


int main(int argc, char** argv) {

  benchmark::internal::Benchmark* bench;

  // fused path
  bench = benchmark::RegisterBenchmark("fusedl2nn/float/float/<int, float>",
                                      benchmark_fusedl2nn<float, float, raft::KeyValuePair<int, float>, int, AlgorithmType::fused>);
  bench->Apply(CustomArguments<int>);

  // unfused path
  // half -> half
  bench = benchmark::RegisterBenchmark("gemm_reduce/half/int/<int, half>",
                                            benchmark_fusedl2nn<half, half, raft::KeyValuePair<int, half>, int, AlgorithmType::gemm_reduce>);
  bench->Apply(CustomArguments<int>);

  // half -> float
  bench = benchmark::RegisterBenchmark("gemm_reduce/half/int/<int, float>",
                                            benchmark_fusedl2nn<half, float, raft::KeyValuePair<int, float>, int, AlgorithmType::gemm_reduce>);
  bench->Apply(CustomArguments<int>);

  // float -> float
  bench = benchmark::RegisterBenchmark("gemm_reduce/float/int/<int, float>",
                                            benchmark_fusedl2nn<float, float, raft::KeyValuePair<int, float>, int, AlgorithmType::gemm_reduce>);

  bench->Apply(CustomArguments<int>);

  // just gemm
  // half -> half
  bench = benchmark::RegisterBenchmark("gemm/half/int/<int, half>",
                                      benchmark_fusedl2nn<half, half, raft::KeyValuePair<int, half>, int, AlgorithmType::gemm>);
  bench->Apply(CustomArguments<int>);
  // half -> float
  bench = benchmark::RegisterBenchmark("gemm/half/int/<int, float>",
                                      benchmark_fusedl2nn<half, float, raft::KeyValuePair<int, float>, int, AlgorithmType::gemm>);
  bench->Apply(CustomArguments<int>);
  // float -> float
  bench = benchmark::RegisterBenchmark("gemm/float/int/<int, float>",
                                      benchmark_fusedl2nn<float, float, raft::KeyValuePair<int, float>, int, AlgorithmType::gemm>);
  bench->Apply(CustomArguments<int>);

  // hand coded tensor core MMA
  // half -> half
  bench = benchmark::RegisterBenchmark("tensor/half/int/<int, half>",
                                      benchmark_fusedl2nn<half, half, raft::KeyValuePair<int, half>, int, AlgorithmType::tensor>);
  bench->Apply(CustomArguments<int>);

  // Initialize benchmark
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}

