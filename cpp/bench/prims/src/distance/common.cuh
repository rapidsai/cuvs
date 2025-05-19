#pragma once

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#include <iostream>
#include <cuda_runtime.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/kvp.hpp>
// Error checking macro
#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status = call;                                           \
    if (status != cudaSuccess) {                                         \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "             \
                << cudaGetErrorString(status) << std::endl;              \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CHECK_CUBLAS(call)                                               \
  do {                                                                   \
    cublasStatus_t status = call;                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                               \
      std::cerr << "cuBLAS Error at line " << __LINE__ << ": "           \
                << status << std::endl;                                  \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


template <typename DataT>
__host__ __device__ DataT max_val() {
  if constexpr (std::is_same<DataT, half>::value) {
    return CUDART_MAX_NORMAL_FP16;
  } else {
    return cuda::std::numeric_limits<DataT>::max();
  }
}

template <typename DataT>
DataT min_val() {
  if constexpr (std::is_same<DataT, half>::value) {
    return CUDART_MIN_DENORM_FP16;
  } else {
    return cuda::std::numeric_limits<DataT>::min();
  }
}

template <typename T>
__global__ void print_kernel(T* f, size_t n_rows, size_t n_cols, bool is_row_major=true) {
  for (size_t r = 0; r < n_rows; r++) {
    printf("[");
    for (size_t c = 0; c < n_cols; c++) {
      size_t index = 0;
      if (is_row_major) {
        index = r * n_cols + c;
      } else {
        index = r + c * n_rows;
      }
      if constexpr (std::is_floating_point<T>::value) {
        double val;
        if constexpr (std::is_same<T, half>::value) {
          val = double(float(f[index]));
        } else {
          val = double(f[index]);
        }
        printf("%f, ", val);
      } else {
        double val;
        if (std::is_same<decltype(f[index].value), half>::value) {
          val = double(float(f[index].value));
        } else {
          val = double(f[index].value);
        }
        printf("<%ld, %e>, ", int64_t(f[index].key), val);
      }
    }
    printf("]\n");
  }
}

class CudaEventTimer {
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaStream_t stream_;
    float elapsed_ms_;
    bool timing_started_;

public:
    CudaEventTimer(cudaStream_t stream = nullptr) : stream_(stream), elapsed_ms_(0.0f), timing_started_(false) {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~CudaEventTimer() {
        CHECK_CUDA(cudaEventDestroy(start_));
        CHECK_CUDA(cudaEventDestroy(stop_));
    }

    void start() {
        CHECK_CUDA(cudaEventRecord(start_, stream_));
        timing_started_ = true;
    }

    void stop() {
        if (!timing_started_) {
            std::cerr << "Warning: Timer stopped without being started" << std::endl;
            return;
        }
        CHECK_CUDA(cudaEventRecord(stop_, stream_));
        CHECK_CUDA(cudaEventSynchronize(stop_));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms_, start_, stop_));
        timing_started_ = false;
    }

    float elapsed_millis() const {
        return elapsed_ms_;
    }

    float elapsed_seconds() const {
        return elapsed_ms_ / 1000.0f;
    }
};

// Host RNG for filling a buffer
class PCG {
public:
  // The constructor
  PCG(uint64_t seed, uint64_t subsequence)
  {
    pcg_state = uint64_t(0);
    inc       = (subsequence << 1u) | 1u;
    clock();
    pcg_state += seed;
    clock();
  }

  // Get a single uint32_t value based on the state of the PCG
  void next(uint32_t& ret) {
    uint32_t xorshifted = ((pcg_state >> 18u) ^ pcg_state) >> 27u;
    uint32_t rot        = pcg_state >> 59u;
    ret                 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    clock();
    return;
  }

  void next(float& ret) {
    uint32_t val = 0;
    next(val);
    val >>= 8;
    ret   = static_cast<float>(val) / float(uint32_t(1) << 24);
  }

  void next(double& ret) {
    uint32_t val_lo = 0;
    uint32_t val_hi = 0;
    next(val_lo);
    next(val_hi);
    uint64_t val = uint64_t(val_hi) << 32 | uint64_t(val_lo);
    val >>= 11;
    ret   = static_cast<double>(val) / double(uint64_t(1) << 53);
  }

  template <typename T>
  void fill_buffer(T* buff, size_t n_items) {
    for (size_t i = 0; i < n_items; i++) {
      T rng_val;
      next(rng_val);
      buff[i] = rng_val;
    }
  }
  // Move ahead PCG state by a single step
  void clock() {
    pcg_state = pcg_state * 6364136223846793005ULL + inc;
  }

private:
  uint64_t pcg_state;
  uint64_t inc;
};

template <typename DataT, typename OutT, typename IdxT>
void ref_l2nn(OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K) {

  for (IdxT m = 0; m < M; m++) {
    IdxT min_index = N + 1;
    DataT min_dist = std::numeric_limits<DataT>::max();
    for (IdxT n = 0; n < N; n++) {
      DataT dist = DataT(0.0);
      for (IdxT k = 0; k < K; k++) {
        DataT diff = A[m * K + k] - B[n * K + k];
        dist += (diff * diff);
      }
      if (dist < min_dist) {
        min_dist = dist;
        min_index = n;
      }
    }
    if constexpr (std::is_floating_point<OutT>::value) {
      out[m] = min_dist;
    } else {
      out[m].key = min_index;
      out[m].value = min_dist;
    }
  }
}

// This is a naive implementation of l2-distance finding nearest neighbour
template <typename DataT, typename OutT, typename IdxT>
__global__ void ref_l2nn_dev(OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K) {
  IdxT tid = threadIdx.x + blockIdx.x * size_t(blockDim.x);
  IdxT n_warps = (size_t(blockDim.x) * gridDim.x) / 32;

  IdxT warp_id = tid / 32;
  IdxT warp_lane = threadIdx.x % 32;
  const int warp_size = 32;

  for (IdxT m = warp_id; m < M; m+=n_warps) {
    __shared__ DataT dist[4];

    IdxT min_index = N + 1;
    DataT min_dist = max_val<DataT>();
    /*if constexpr (std::is_same<DataT, half>::value) {
      min_dist = CUDART_MAX_NORMAL_FP16;
    } else {
      min_dist = cuda::std::numeric_limits<DataT>::max();
    }*/
    //if (tid == 0) printf("max value = %e %ld\n", __half2float(min_dist), sizeof(DataT));
    for (IdxT n = 0; n < N; n++) {
      if (warp_lane == 0) {
        dist[warp_id % 4] = DataT(0.0);
      }
      DataT th_dist = DataT(0.0);
      for (IdxT k = warp_lane; k < K; k+=warp_size) {
        DataT diff = A[m * K + k] - B[n * K + k];
        th_dist += (diff * diff);
      }
      __syncwarp();
      atomicAdd(&dist[warp_id % 4], th_dist);
      __syncwarp();
    /*if (m == 0 && n == 0 && warp_lane == 0) {
      for (int i = 0; i < K; i++) {
        printf("%f, ", A[i]);
      }
      printf("\n");

      for (int i = 0; i < K; i++) {
        printf("%f, ", B[i]);
      }
      printf("\nref dist = %f\n", dist[warp_id % 4]);
    }
      __syncwarp();*/
      if (warp_lane == 0 && dist[warp_id % 4] < min_dist) {
        min_dist = dist[warp_id % 4];
        min_index = n;
      }
    }
    if constexpr (std::is_floating_point<OutT>::value) {
      if (warp_lane == 0) {
        out[m] = min_dist;
      }
    } else {
      // output is a raft::KeyValuePair
      if (warp_lane == 0) {
        out[m].key = min_index;
        out[m].value = min_dist;
      }
    }
  }
}

template <typename DataT, typename OutT, typename IdxT>
void ref_l2nn_api(OutT* out, const DataT* A, const DataT* B, IdxT m, IdxT n, IdxT k, cudaStream_t stream) {

  //constexpr int block_dim = 128;
  //static_assert(block_dim % 32 == 0, "blockdim must be divisible by 32");
  //constexpr int warps_per_block = block_dim / 32;
  //int num_blocks = m ;
  ref_l2nn_dev<<<m/4, 128, 0, stream>>>(out, A, B, m, n, k);
  return;
}

// Structure to track comparison failures
struct ComparisonFailure {
  int failed;       // Flag indicating if comparison failed
  int64_t first_index; // First index where comparison failed
  double diff;   // Maximum difference found
  int mutex;        // Simple mutex lock for thread synchronization
};

template <typename DataT, typename IdxT>
__global__ void vector_compare_kernel(const DataT* a, const DataT* b, IdxT n, double tolerance = 1e-6,
                                      ComparisonFailure* global_failure = nullptr) {
  // Shared memory for tracking failures within a block
  __shared__ ComparisonFailure block_failure;

  // Initialize shared memory variable
  if (threadIdx.x == 0) {
    block_failure.failed = 0;
    block_failure.first_index = -1;
    block_failure.diff = 0;
    block_failure.mutex = 0;
  }
  __syncthreads();

  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n) {

    double diff;
    if constexpr (std::is_floating_point<DataT>::value) {
      diff = abs(a[tid], b[tid]);
      //printf("Expected = %f vs actual = %f\n", a[tid], b[tid]);
    } else {
      diff = abs(a[tid].value - b[tid].value);
      //if (tid == 0) printf("Expected = %f vs actual = %f\n", a[tid].value, b[tid].value);
      //if (tid == 0) printf("Expected = %d vs actual = %d\n", a[tid].key, b[tid].key);
    }
    if (diff > tolerance) {
      // Acquire mutex lock using atomic compare-and-swap
      while (atomicCAS(&block_failure.mutex, 0, 1) != 0) {
        // Spin wait
      }

      // Critical section: update first_index if this is the earliest failure
      if (block_failure.first_index == IdxT(-1) || tid < block_failure.first_index) {
        block_failure.failed = 1;
        block_failure.first_index = tid;
        block_failure.diff = diff;
      }

      // Release mutex
      atomicExch(&block_failure.mutex, 0);
    }
  }

  __syncthreads();

  // First thread in the block can report the failure if needed
  if (threadIdx.x == 0 && block_failure.failed == 1) {
    // Acquire mutex lock using atomic compare-and-swap
    while (atomicCAS(&global_failure->mutex, 0, 1) != 0) {
      // Spin wait
    }

    // Critical section: update first_index if this is the earliest failure
    if (global_failure->first_index == IdxT(-1) || block_failure.first_index < global_failure->first_index) {
      global_failure->failed = 1;
      global_failure->first_index = block_failure.first_index;
      global_failure->diff = block_failure.diff;
    }

    // Release mutex
    atomicExch(&global_failure->mutex, 0);
  }
}

template <typename DataT, typename IdxT>
void vector_compare(const DataT* a, const DataT* b, const IdxT n, double tolerance = double(1e-6), cudaStream_t stream = nullptr) {
  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  ComparisonFailure* global_failure;
  CHECK_CUDA(cudaMallocManaged(&global_failure, sizeof(ComparisonFailure)));

  global_failure->failed = 0;
  global_failure->first_index = -1;
  global_failure->diff = 0.0;
  global_failure->mutex = 0;

  vector_compare_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n, tolerance, global_failure);

  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (global_failure->failed) {
    std::cout << "Vector comparison failed: values differ by more than tolerance " << tolerance << std::endl;
    std::cout << "First failure at index " << global_failure->first_index
              << " with difference " << global_failure->diff << std::endl;
  } else {
    std::cout << "Vector comparison passed: all values within tolerance " << tolerance << std::endl;
  }

  CHECK_CUDA(cudaFree(global_failure));
}
