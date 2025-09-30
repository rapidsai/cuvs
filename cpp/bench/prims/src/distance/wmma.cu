/**
 * @file wmma_benchmark.cu
 * @brief Benchmark demonstrating NVIDIA CUDA WMMA (Warp Matrix Multiply Accumulate) operations
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <vector>

// Error checking macro
#define CHECK_CUDA(call)                                                                   \
  do {                                                                                     \
    cudaError_t status = call;                                                             \
    if (status != cudaSuccess) {                                                           \
      std::cerr << "CUDA Error at line " << __LINE__ << ": " << cudaGetErrorString(status) \
                << std::endl;                                                              \
      exit(EXIT_FAILURE);                                                                  \
    }                                                                                      \
  } while (0)

class CudaEventTimer {
 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaStream_t stream_;
  float elapsed_ms_;
  bool timing_started_;

 public:
  CudaEventTimer(cudaStream_t stream = nullptr)
    : stream_(stream), elapsed_ms_(0.0f), timing_started_(false)
  {
    CHECK_CUDA(cudaEventCreate(&start_));
    CHECK_CUDA(cudaEventCreate(&stop_));
  }

  ~CudaEventTimer()
  {
    CHECK_CUDA(cudaEventDestroy(start_));
    CHECK_CUDA(cudaEventDestroy(stop_));
  }

  void start()
  {
    CHECK_CUDA(cudaEventRecord(start_, stream_));
    timing_started_ = true;
  }

  void stop()
  {
    if (!timing_started_) {
      std::cerr << "Warning: Timer stopped without being started" << std::endl;
      return;
    }
    CHECK_CUDA(cudaEventRecord(stop_, stream_));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms_, start_, stop_));
    timing_started_ = false;
  }

  float elapsed_millis() const { return elapsed_ms_; }

  float elapsed_seconds() const { return elapsed_ms_ / 1000.0f; }
};

// Using namespace for WMMA operations
using namespace nvcuda;

// WMMA dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

/**
 * @brief CUDA kernel that performs matrix multiplication using WMMA
 *
 * This kernel computes C = A * B using Tensor Cores via WMMA API
 *
 * @param a Input matrix A
 * @param b Input matrix B
 * @param c Output matrix C
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
template <typename InT, typename ArithT, typename AccT>
__global__ void wmma_matrix_multiply(
  const InT* a, const InT* b, AccT* c, int m, int n, int k, bool debug)
{
  // Block index
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Check if this warp is within bounds
  if (warpM >= m / WMMA_M || warpN >= n / WMMA_N) return;

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, ArithT, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, ArithT, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccT> c_frag;

  // Initialize the output to zero
  wmma::fill_fragment(c_frag, 0.0f);

  // Loop over k dimension
  for (int i = 0; i < k; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a + aRow * k + aCol, k);
    wmma::load_matrix_sync(b_frag, b + bRow * n + bCol, n);

    // if (i == 0 && debug == true) {
    //   printf("%d,%f,%f,%f,%f\n", threadIdx.x, a_frag.x[0], a_frag.x[1], a_frag.x[2],
    //   a_frag.x[3]);
    // }
    //  Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // Store the output
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;
  wmma::store_matrix_sync(c + cRow * n + cCol, c_frag, n, wmma::mem_row_major);
}

/**
 * @brief Benchmark for matrix multiplication using WMMA Tensor Cores
 *
 * This benchmark computes C = A * B using NVIDIA's Tensor Cores via WMMA API
 */
template <typename InT, typename ArithT, typename AccT>
static void benchmark_wmma(benchmark::State& state)
{
  // Get matrix dimensions from benchmark parameters
  // Ensure dimensions are multiples of WMMA tile sizes
  const int m = state.range(0);
  const int n = state.range(1);
  const int k = state.range(2);

  // Check if dimensions are valid for WMMA
  if (m % WMMA_M != 0 || n % WMMA_N != 0 || k % WMMA_K != 0) {
    state.SkipWithError("Matrix dimensions must be multiples of WMMA tile sizes");
    return;
  }

  // Allocate host memory
  std::vector<InT> h_a(m * k);
  std::vector<InT> h_b(k * n);
  std::vector<AccT> h_c(m * n, 0.0f);

  // Initialize input matrices with values
  for (int i = 0; i < m * k; ++i) {
    h_a[i] = InT(i);  // Initialize with 1.0
  }

  for (int i = 0; i < k * n; ++i) {
    h_b[i] = InT(1.0f);  // Initialize with 1.0
  }

  // Allocate device memory
  InT *d_a, *d_b;
  AccT* d_c;

  CHECK_CUDA(cudaMalloc(&d_a, m * k * sizeof(InT)));
  CHECK_CUDA(cudaMalloc(&d_b, k * n * sizeof(InT)));
  CHECK_CUDA(cudaMalloc(&d_c, m * n * sizeof(AccT)));

  // Copy data from host to device
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), m * k * sizeof(InT), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), k * n * sizeof(InT), cudaMemcpyHostToDevice));

  // Create CUDA stream
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Set up grid and block dimensions
  dim3 gridDim((m + (WMMA_M * 2 - 1)) / (WMMA_M * 2), (n + WMMA_N - 1) / WMMA_N);
  dim3 blockDim(64, 2);

  // Create timer
  CudaEventTimer timer(stream);

  // Warm-up run
  wmma_matrix_multiply<InT, ArithT, AccT><<<1, 32, 0, stream>>>(d_a, d_b, d_c, m, n, k, true);

  // Synchronize before benchmarking
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Benchmark loop
  timer.start();
  for (auto _ : state) {
    wmma_matrix_multiply<InT, ArithT, AccT>
      <<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, m, n, k, false);
  }
  timer.stop();

  // Synchronize to make sure the kernel is done
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Calculate and report stats
  state.counters["M"]         = m;
  state.counters["N"]         = n;
  state.counters["K"]         = k;
  state.counters["iter_time"] = timer.elapsed_seconds() / state.iterations();
  state.counters["FLOP/s"] =
    (int64_t(state.iterations()) * 2 * m * n * k) / timer.elapsed_seconds();

  // Clean up
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaStreamDestroy(stream));
}

/**
 * @brief Custom argument generator for the benchmark
 *
 * Generates matrix dimensions that are compatible with WMMA tile sizes
 */
static void CustomArguments(benchmark::internal::Benchmark* b)
{
  // Matrix dimensions must be multiples of WMMA tile sizes
  std::vector<int64_t> m_list = {128, 256, 512, 1024, 2048};
  std::vector<int64_t> n_list = {128, 256, 512, 1024, 2048};
  std::vector<int64_t> k_list = {1024, 2048};

  for (auto k : k_list) {
    for (auto m : m_list) {
      for (auto n : n_list) {
        if (m > n) continue;
        b->Args({m, n, k});
        if (m != n) { b->Args({n, m, k}); }
      }
    }
  }
}

int main(int argc, char* argv[])
{
  // Register the benchmark
  benchmark::internal::Benchmark* bench;
  bench = benchmark::RegisterBenchmark("wmma_matmul/fp16", benchmark_wmma<half, half, float>);
  bench->Apply(CustomArguments);

  // bench = benchmark::RegisterBenchmark("wmma_matmul/tf32", benchmark_wmma<float,
  // wmma::precision::tf32, float>); bench->Apply(CustomArguments);

  // Initialize benchmark
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
