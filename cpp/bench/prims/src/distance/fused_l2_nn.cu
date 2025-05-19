#include <benchmark/benchmark.h>
#include <iostream>
#include "common.cuh"
#include "cublas_sample.cu"

#include <raft/core/resource/cuda_stream.hpp>

#include "../../../../src/distance/fused_distance_nn.cuh"

#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>


#ifdef DEV_EXP

template <typename DataT, typename outT>
__global__ void naive_l2_nn(outT* out, DataT* A, DataT* B, size_t M, size_t N, size_t K) {
  // A is M x K
  // B is N x K
  size_t tx = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  size_t ty = threadIdx.y + size_t(blockIdx.y) * blockDim.y;

  for (size_t x = tx; x < M; x += size_t(blockDim.x) * gridDim.x) {
    for (size_t y = ty; y < N; y += size_t(blockDim.y) * gridDim.y) {
      DataT d = 0;
      for (size_t z = 0; z < K; z++) {
        DataT tmp = A[x * K + z] - B[y * K + z];
        d += (tmp * tmp);
      }
      atomicMin(&out[x], d);
    }
  }
}
#endif

  enum class AlgorithmType {
    gemm,
    gemm_reduce,
    fused,
  };

  template <typename DataT, typename IdxT, typename OutT, AlgorithmType algo>
  void benchmark_fusedl2nn(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);
    const int k = state.range(2);

    raft::device_resources handle;
    rmm::cuda_stream_view stream;
    using AccT = raft::KeyValuePair<int, float>::Value;

    stream = raft::resource::get_cuda_stream(handle);
    auto x_h = raft::make_host_matrix<DataT, IdxT>(handle, m, k);
    auto y_h = raft::make_host_matrix<DataT, IdxT>(handle, n, k);
    auto out_h = raft::make_host_vector<OutT, IdxT>(handle, m);

    auto x     = raft::make_device_matrix<DataT, IdxT>(handle, m, k);
    auto y     = raft::make_device_matrix<DataT, IdxT>(handle, n, k);
    auto x_norm = raft::make_device_vector<DataT, IdxT>(handle, m);
    auto y_norm = raft::make_device_vector<DataT, IdxT>(handle, n);
    auto out   = raft::make_device_vector<OutT, IdxT>(handle, m);
    auto out_exp = raft::make_device_vector<OutT, IdxT>(handle, m);


    raft::random::RngState rng{1234};
    raft::random::uniform(
       handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
    raft::random::uniform(
       handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));

    //ref_l2nn(out_h.data_handle(), x_h.data_handle(), y_h.data_handle(), m, n, k);
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

    raft::device_matrix<DataT, IdxT> z = raft::make_device_matrix<DataT, IdxT>(handle, m, n);

    raft::device_vector<char, IdxT> workspace = raft::make_device_vector<char, IdxT>(handle, m * sizeof(IdxT));

    CudaEventTimer timer(stream);

    size_t ws_size = m * n * sizeof(8);

    float* workspace_blas;

    OutT* workspace_blas2;

    CHECK_CUDA(cudaMalloc(&workspace_blas, ws_size));
    CHECK_CUDA(cudaMalloc(&workspace_blas2, 1600000));

    //const DataT alpha = 1.0f;
    //const DataT beta = 0.0f;
    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
    ref_l2nn_api(out_exp.data_handle(), x.data_handle(), y.data_handle(), m, n, k, stream);

    cudaDeviceSynchronize();
    //print_kernel<<<1, 1, 0, stream>>>(out_exp.data_handle(), 1, 10);
    //cudaDeviceSynchronize();
    CHECK_CUDA(cudaStreamSynchronize(stream));

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
        cublas_l2nn<OutT, DataT, IdxT, false>(out.data_handle(), x.data_handle(),
          y.data_handle(), m, n, k, x_norm.data_handle(), y_norm.data_handle(), workspace_blas, ws_size, workspace_blas2, cublas_handle, stream);
      }
      if constexpr (algo == AlgorithmType::gemm) {
        cublas_l2nn<OutT, DataT, IdxT, true>(out.data_handle(), x.data_handle(),
          y.data_handle(), m, n, k, x_norm.data_handle(), y_norm.data_handle(), workspace_blas, ws_size, workspace_blas2, cublas_handle, stream);
      }
    }
    timer.stop();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if constexpr (algo == AlgorithmType::gemm) {
      reduce_min<OutT, DataT, IdxT>(out.data_handle(),
                                    workspace_blas,
                                    x_norm.data_handle(),
                                    y_norm.data_handle(),
                                    m, n, workspace_blas2, stream);
    }
    vector_compare(out_exp.data_handle(), out.data_handle(), m, 1e-4, stream);
    state.counters["M"] = m;
    state.counters["N"] = n;
    state.counters["K"] = k;
    state.counters["iter_time"] =  timer.elapsed_seconds() / state.iterations();
    state.counters["FLOP/s"] =  (int64_t(state.iterations()) * 2 * m * n * k) / timer.elapsed_seconds();
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

  /*std::vector<int64_t> m_list = {1024, 2048, 4096, 8192, 16384};
  std::vector<int64_t> n_list = {1024, 2048, 4096, 8192, 16384};
  std::vector<int64_t> k_list = {8, 16, 32, 64, 128, 256, 512};
  for (auto k : k_list) {
    for (auto m : m_list) {
      for (auto n : n_list) {
        if (m > n) continue;
        b->Args({m, n, k});
        if (m != n) {
          b->Args({n, m, k});
        }
      }
    }
  }*/
  b->Args({6000, 3000, 3000});
}

/*   template <typename IdxT>
   static void CustomArguments(benchmark::internal::Benchmark* b) {

//     std::vector<int64_t> m_list = {1, 100000, 1000000};
     std::vector<int64_t> m_list = {1, 100'000};
     //if constexpr (sizeof(IdxT) == 8) { m_list.push_back(10000000); }
     //std::vector<int64_t> n_list = {100, 1000, 10000};
     std::vector<int64_t> n_list = {8, 100'000};
     //std::vector<int64_t> k_list = {64, 128, 256};
     std::vector<int64_t> k_list = {8, 128};
     for (auto m : m_list) {
       for (auto n : n_list) {
         for (auto k : k_list) {
           b->Args({m, n, k});
         }
       }
     }
   }*/

   int main(int argc, char** argv) {

     benchmark::internal::Benchmark* bench;

     // IdxT = int
     //bench = benchmark::RegisterBenchmark("fusedl2nn/float/int/float", benchmark_fusedl2nn<float, int, float>);
     //bench->Apply(CustomArguments<int>);

     //bench = benchmark::RegisterBenchmark("fusedl2nn/double/int/double", benchmark_fusedl2nn<double, int, double>);
     //bench->Apply(CustomArguments<int>);

     bench = benchmark::RegisterBenchmark("gemm_reduce/float/int/<int, float>",
                                          benchmark_fusedl2nn<float, int, raft::KeyValuePair<int, float>, AlgorithmType::gemm_reduce>);
     bench->Apply(CustomArguments<int>);
     bench = benchmark::RegisterBenchmark("fusedl2nn/float/int/<int, float>",
                                          benchmark_fusedl2nn<float, int, raft::KeyValuePair<int, float>, AlgorithmType::fused>);
     bench->Apply(CustomArguments<int>);

     bench = benchmark::RegisterBenchmark("gemm/float/int/<int, float>",
                                          benchmark_fusedl2nn<float, int, raft::KeyValuePair<int, float>, AlgorithmType::gemm>);
     bench->Apply(CustomArguments<int>);
     //bench = benchmark::RegisterBenchmark("fusedl2nn/double/int/<int, double>", benchmark_fusedl2nn<double, int, raft::KeyValuePair<int, double>>);
     //bench->Apply(CustomArguments<int>);

     // IdxT = in64_t
     //bench = benchmark::RegisterBenchmark("fusedl2nn/float/int64_t/float", benchmark_fusedl2nn<float, int64_t, float>);
     //bench->Apply(CustomArguments<int>);

     //bench = benchmark::RegisterBenchmark("fusedl2nn/double/int64_t/double", benchmark_fusedl2nn<double, int64_t, double>);
     //bench->Apply(CustomArguments<int>);

     //bench = benchmark::RegisterBenchmark("fusedl2nn/float/int64_t/<int64_t, float>", benchmark_fusedl2nn<float, int64_t, raft::KeyValuePair<int64_t, float>>);
     //bench->Apply(CustomArguments<int>);

     //bench = benchmark::RegisterBenchmark("fusedl2nn/double/int64_t/<int64_t, double>", benchmark_fusedl2nn<double, int64_t, raft::KeyValuePair<int64_t, double>>);
     //bench->Apply(CustomArguments<int>);

     // Initialize benchmark
     ::benchmark::Initialize(&argc, argv);
     if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
     ::benchmark::RunSpecifiedBenchmarks();
     ::benchmark::Shutdown();
     return 0;
   }
