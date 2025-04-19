/**
 * @file cublas_sample.cu
 * @brief Sample code demonstrating cuBLAS GEMM (General Matrix Multiplication)
 */

#include <cublas_v2.h>
#include <benchmark/benchmark.h>
#include "common.cuh"

#define CHECK_CUBLAS(call)                                               \
  do {                                                                   \
    cublasStatus_t status = call;                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                               \
      std::cerr << "cuBLAS Error at line " << __LINE__ << ": "           \
                << status << std::endl;                                  \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


//
// @brief Benchmark for matrix multiplication using cuBLAS
//  
// This benchmark computes C = alpha*A*B + beta*C using cuBLAS GEMM
// where A, B, and C are matrices
//
template <typename DataT>
static void benchmark_cublasgemm(benchmark::State& state) {
  // Get matrix dimensions from benchmark parameters
  const size_t m = state.range(0);  // rows of A and C
  const size_t n = state.range(1);  // cols of B and C
  const size_t k = state.range(2);  // cols of A and rows of B

  // Host matrices
  std::vector<DataT> h_A(m * k, 0.0);  // Initialize with 0.0
  std::vector<DataT> h_B(k * n, 0.0);  // Initialize with 0.0
  std::vector<DataT> h_C(m * n, 0.0);  // Initialize with 0.0

  PCG rng(1234, 0);
  rng.fill_buffer(h_A.data(), m * k);
  rng.fill_buffer(h_B.data(), k * n);
  // Device matrices
  DataT *d_A, *d_B, *d_C;
 
  // Allocate device memory
  CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(DataT)));
  CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(DataT)));
  CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(DataT)));

  // Copy matrices from host to device
  CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(DataT), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(DataT), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), m * n * sizeof(DataT), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Create cuBLAS handle
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  // Enable TF32 mode
  //CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  CHECK_CUBLAS(cublasSetStream(handle, stream));


  // Create a CUDA event timer
  CudaEventTimer timer(stream);
  // Set up scaling factors
  const DataT alpha = 1.0f;
  const DataT beta = 0.0f;

  // Warm up - use the appropriate GEMM function based on data type
  if constexpr (std::is_same_v<DataT, float>) {
    CHECK_CUBLAS(cublasSgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,  // No transpositions
      n, m, k,                   // Dimensions (swapped due to row/col-major difference)
      &alpha,                    // alpha
      d_B, n,                    // B and its leading dimension
      d_A, k,                    // A and its leading dimension
      &beta,                     // beta
      d_C, n                     // C and its leading dimension
    ));
  } else if constexpr (std::is_same_v<DataT, double>) {
    CHECK_CUBLAS(cublasDgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,  // No transpositions
      n, m, k,                   // Dimensions (swapped due to row/col-major difference)
      &alpha,                    // alpha
      d_B, n,                    // B and its leading dimension
      d_A, k,                    // A and its leading dimension
      &beta,                     // beta
      d_C, n                     // C and its leading dimension
    ));
  }

  // Synchronize before benchmarking
  CHECK_CUDA(cudaStreamSynchronize(stream));

    timer.start();
  // Benchmark loop
  for (auto _ : state) {
    // Perform matrix multiplication: C = alpha*A*B + beta*C
    // Note: cuBLAS uses column-major ordering, but we're using row-major
    // So we compute B*A instead of A*B (i.e., we swap the order)
    if constexpr (std::is_same_v<DataT, float>) {
      CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpositions
        n, m, k,                   // Dimensions (swapped due to row/col-major difference)
        &alpha,                    // alpha
        d_B, n,                    // B and its leading dimension
        d_A, k,                    // A and its leading dimension
        &beta,                     // beta
        d_C, n                     // C and its leading dimension
      ));
    } else if constexpr (std::is_same_v<DataT, double>) {
      CHECK_CUBLAS(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpositions
        n, m, k,                   // Dimensions (swapped due to row/col-major difference)
        &alpha,                    // alpha
        d_B, n,                    // B and its leading dimension
        d_A, k,                    // A and its leading dimension
        &beta,                     // beta
        d_C, n                     // C and its leading dimension
      ));
    }
  }
  timer.stop();  
  // Synchronize to make sure the kernel is done
  CHECK_CUDA(cudaStreamSynchronize(stream));
  // Calculate and report stats
  //state.SetBytesProcessed(int64_t(state.iterations()) * 
   //                      (m * k + k * n + m * n) * sizeof(DataT));
  //state.SetItemsProcessed(int64_t(state.iterations()) * m * n * k * 2); // 2 FLOPs per element
     state.counters["M"] = m;
     state.counters["N"] = n;
     state.counters["K"] = k;
     state.counters["iter_time"] =  timer.elapsed_seconds() / state.iterations();
  state.counters["iter_time"] =  timer.elapsed_seconds() / state.iterations();
  state.counters["FLOP/s"] = (m * n * k * 2 * int64_t(state.iterations())) / timer.elapsed_seconds();

  // Clean up
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
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
  std::vector<int64_t> m_list = {6*1000};
  std::vector<int64_t> n_list = {3*1000};
  std::vector<int64_t> k_list = {3*1000};
  for (auto k : k_list) {
    for (auto m : m_list) {
      for (auto n : n_list) {
        b->Args({m, n, k});
      }
    }
  }
}

int main(int argc, char* argv[]) {

  benchmark::internal::Benchmark* bench;

  //bench = benchmark::RegisterBenchmark("cublas_gemm/float", benchmark_cublasgemm<float>);
  //bench->Apply(CustomArguments<int>);

  bench = benchmark::RegisterBenchmark("cublas_gemm/double", benchmark_cublasgemm<double>);
  bench->Apply(CustomArguments<int>);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
