#include <benchmark/benchmark.h>
#include <raft/core/resource/cuda_stream.hpp>

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#include "../../../../src/distance/fused_distance_nn.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>

#ifdef DEV_EXP

template <typename T>
__global__ void print_kernel(T* f, size_t n_rows, size_t n_cols) {
  for (size_t r = 0; r < n_rows; r++) {
    for (size_t c = 0; c < n_cols; c++) {
      if constexpr (std::is_same_v<T, raft::KeyValuePair<int, float>>) {
        printf("<%d, %f>, ", f[r * n_cols + c].key, f[r * n_cols + c].value);
      } else {
        printf("%f, ", f[r * n_cols + c]);
      }
    }
    printf("\n");
  }
}


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

 template <typename DataT, typename IdxT, typename OutT>
 void benchmark_fusedl2nn(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);
    const int k = state.range(2);

    raft::device_resources handle;
    rmm::cuda_stream_view stream;

    stream = raft::resource::get_cuda_stream(handle);
    auto x     = raft::make_device_matrix<DataT, int>(handle, m, k);
    auto y     = raft::make_device_matrix<DataT, IdxT>(handle, n, k);
    auto x_norm = raft::make_device_vector<DataT, IdxT>(handle, m);
    auto y_norm = raft::make_device_vector<DataT, IdxT>(handle, n);
    auto out   = raft::make_device_vector<OutT, IdxT>(handle, m);

    raft::random::RngState rng{1234};
    raft::random::uniform(
       handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
    raft::random::uniform(
       handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));

#ifdef NVDEV
    printf("x = \n");
    cudaDeviceSynchronize();
    print_kernel<<<1, 1>>>(x.data_handle(), m, k);
    cudaDeviceSynchronize();

    printf("y = \n");
    cudaDeviceSynchronize();
    print_kernel<<<1, 1>>>(y.data_handle(), n, k);
    cudaDeviceSynchronize();
#endif
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

     for (auto _ : state) {
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
#ifdef NVDEV
     printf("out = \n");
     cudaDeviceSynchronize();
     print_kernel<<<1, 1>>>(out.data_handle(), m, 1);
     cudaDeviceSynchronize();
#endif
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
                                                 benchmark::Counter::OneK::kIs1000);
   }

   template <typename IdxT>
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
   }

   int main(int argc, char** argv) {

     benchmark::internal::Benchmark* bench;

     // IdxT = int
     bench = benchmark::RegisterBenchmark("fusedl2nn/float/int/float", benchmark_fusedl2nn<float, int, float>);
     bench->Apply(CustomArguments<int>);

     //bench = benchmark::RegisterBenchmark("fusedl2nn/double/int/double", benchmark_fusedl2nn<double, int, double>);
     //bench->Apply(CustomArguments<int>);

     bench = benchmark::RegisterBenchmark("fusedl2nn/float/int/<int, float>", benchmark_fusedl2nn<float, int, raft::KeyValuePair<int, float>>);
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
