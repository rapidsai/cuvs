#include <benchmark/benchmark.h>
#include <iostream>
#include "common.cuh"
#include "cublas_sample.cu"

#include <raft/core/resource/cuda_stream.hpp>

#include "../../../../src/distance/fused_distance_nn.cuh"

#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>

__global__ void gpu_memcpy_kernel(const float4* src, float4* dest, size_t len, int itr=1) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (int i = 0; i < itr; i++) {
      // Grid-stride loop for arbitrary buffer sizes
      for(; idx < len; idx += stride) {
          dest[idx] = src[idx];
      }
    }
}


void launch_memcpy() {
    
    float4* dst;
    float4* src;
    const size_t len = 1024 * 1024 * 1024 / sizeof(float4);
    unsigned int blocks = 1;
    const unsigned int block_size = 1024;
    unsigned int grid_size = blocks;
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    CHECK_CUDA(cudaMallocAsync(&src, len * sizeof(float4), stream));
    CHECK_CUDA(cudaMallocAsync(&dst, len * sizeof(float4), stream));
    
    gpu_memcpy_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, len, 5);
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaFreeAsync(src, stream));
    CHECK_CUDA(cudaFreeAsync(dst, stream));
    
    CHECK_CUDA(cudaStreamDestroy(stream));
}


  enum class AlgorithmType {
    gemm,
    gemm_reduce,
    fused,
  };

  template <typename DataT, typename AccT, typename OutT, typename IdxT, AlgorithmType algo>
  void benchmark_fusedl2nn(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);
    const int k = state.range(2);

    raft::device_resources handle;
    rmm::cuda_stream_view stream;

    if constexpr (std::is_fundamental<OutT>()) {
      static_assert(std::is_same<OutT, AccT>::value, "OutT and AccT are not the same");
    } else {
      static_assert(std::is_same<OutT, raft::KeyValuePair<IdxT, AccT>>::value, 
          "OutT is not raft::KeyValuePair<IdxT, AccT> type");
    }

    stream = raft::resource::get_cuda_stream(handle);
    //auto x_h = raft::make_host_matrix<DataT, IdxT>(handle, m, k);
    //auto y_h = raft::make_host_matrix<DataT, IdxT>(handle, n, k);
    //auto out_h = raft::make_host_vector<OutT, IdxT>(handle, m);

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
    //cudaMemcpy(x_h.data_handle(), x.data_handle(), m*k*sizeof(DataT), cudaMemcpyDeviceToHost);
    //printf("%f", __half2float(x_h.data_handle()[0]));
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

    raft::device_vector<char, IdxT> workspace = raft::make_device_vector<char, IdxT>(handle, m * sizeof(IdxT));

    CudaEventTimer timer(stream);

    size_t ws_size = m * n * sizeof(AccT);

    raft::device_vector<AccT, IdxT> workspace_blas = raft::make_device_vector<AccT, IdxT>(handle, ws_size);

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
    ref_l2nn_api<DataT, AccT, OutT, IdxT>(out_ref.data_handle(), x.data_handle(), y.data_handle(), m, n, k, stream);
    // Warm up
    //cublas_l2nn<DataT, AccT, OutT, IdxT, false, false>(out.data_handle(), x.data_handle(),
     //     y.data_handle(), m, n, k, x_norm.data_handle(), y_norm.data_handle(), workspace_blas.data_handle(), cublas_handle, stream);
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
          y.data_handle(), m, n, k, x_norm.data_handle(), y_norm.data_handle(), workspace_blas.data_handle(), cublas_handle, stream);
      }
      if constexpr (algo == AlgorithmType::gemm) {
        cublas_l2nn<DataT, AccT, OutT, IdxT, false, true>(out.data_handle(), x.data_handle(),
          y.data_handle(), m, n, k, x_norm.data_handle(), y_norm.data_handle(), workspace_blas.data_handle(), cublas_handle, stream);
      }
    }
    timer.stop();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if constexpr (algo == AlgorithmType::gemm) {
      reduce_min<DataT, AccT, OutT, IdxT>(out.data_handle(),
                                    workspace_blas.data_handle(),
                                    x_norm.data_handle(),
                                    y_norm.data_handle(),
                                    m, n, stream);
    }
    vector_compare(out_ref.data_handle(), out.data_handle(), m, 1e-1, stream);
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

  constexpr int K = 1024;
  std::vector<int64_t> m_list = {8*K, 16*K};
  std::vector<int64_t> n_list = {8*K, 16*K};
  //std::vector<int64_t> k_list = {128, 512, 1024, 1536};
  std::vector<int64_t> k_list = {128};
  for (auto k : k_list) {
    for (auto m : m_list) {
      for (auto n : n_list) {
        b->Args({m, n, k});
      }
    }
  }
  //b->Args({6000, 3000, 3000});
}


   int main(int argc, char** argv) {

     benchmark::internal::Benchmark* bench;

//     bench = benchmark::RegisterBenchmark("fusedl2nn/float/int/<int, float>",
//                                          benchmark_fusedl2nn<float, int, raft::KeyValuePair<int, float>, AlgorithmType::fused>);
//     bench->Apply(CustomArguments<int>);

     bench = benchmark::RegisterBenchmark("gemm_reduce/half/int/<int, float>",
                                          benchmark_fusedl2nn<half, float, raft::KeyValuePair<int, float>, int, AlgorithmType::gemm_reduce>);
     bench->Apply(CustomArguments<int>);

     bench = benchmark::RegisterBenchmark("gemm/half/int/<int, float>",
                                          benchmark_fusedl2nn<half, float, raft::KeyValuePair<int, float>, int, AlgorithmType::gemm>);
     bench->Apply(CustomArguments<int>);

     // Initialize benchmark
     ::benchmark::Initialize(&argc, argv);
     if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
     ::benchmark::RunSpecifiedBenchmarks();
     ::benchmark::Shutdown();
     return 0;
   }

/*
int main()
{
  half2 *d;
  cudaMalloc((void **)&d, 128 * sizeof(half));
  kernel<<<10240,128>>> (d);
  cudaDeviceSynchronize();
}
*/
