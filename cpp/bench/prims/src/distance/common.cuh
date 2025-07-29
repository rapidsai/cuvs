#pragma once

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#include <iostream>
#include <cmath>
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

template <typename T>
constexpr bool is_scalar() {
      return std::is_same<T, double>() || std::is_same<T, float>() ||
             std::is_same<T, half>() || std::is_same<T, int8_t>();
}

template <typename T, typename IdxT, typename AccT>
constexpr static bool is_pair() {
  return std::is_same<T, raft::KeyValuePair<IdxT, AccT>>();
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
class OutAccessor {
    __host__ __device__
    constexpr static bool is_out_scalar() {
      return std::is_same<OutT, double>() || std::is_same<OutT, float>() ||
             std::is_same<OutT, half>() || std::is_same<OutT, int8_t>();
    }

    constexpr static bool is_out_pair() {
      return std::is_same<OutT, raft::KeyValuePair<IdxT, AccT>>();
    }

  public:

      // Check that the OutT type is either scalar (float, int etc.) or of type raft::KeyValuePair
      static_assert(is_out_scalar() || is_out_pair(), "Type of out variable is unsupported");

    __host__ __device__
    inline static IdxT get_key(OutT out) {
      if constexpr (is_out_scalar()) {
        return IdxT(0);
      }

      if constexpr (is_out_pair()) {
        return out.key;
      }
    }

    __host__ __device__
    inline static void set_key(OutT& out, IdxT key) {

      if constexpr (is_out_scalar()) {
        // we are not storing key information
      }

      if constexpr (is_out_pair()) {
        out.key = key;
      }
    }

    __host__ __device__
    inline static AccT get_value(OutT out) {

      if constexpr (is_out_scalar()) {
        return out;
      }

      if constexpr (is_out_pair()) {
        return out.value;
      }
    }

    __host__ __device__
    inline static void set_value(OutT& out, AccT value) {

      if constexpr (is_out_scalar()) {
        out = value;
      }

      if constexpr (is_out_pair()) {
        out.value = value;
      }
    }

};

template <typename T>
__host__ __device__ T max_val() {
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MAX_NORMAL_FP16;
  } else {
    return cuda::std::numeric_limits<T>::max();
  }
}

template <typename T>
T min_val() {
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MIN_DENORM_FP16;
  } else {
    return cuda::std::numeric_limits<T>::min();
  }
}

template <typename T>
__global__ void print_kernel(const T* f, size_t n_rows, size_t n_cols, bool is_row_major=true) {
  for (size_t r = 0; r < n_rows; r++) {
    printf("%ld [", sizeof(T));
    for (size_t c = 0; c < n_cols; c++) {
      size_t index = 0;
      if (is_row_major) {
        index = r * n_cols + c;
      } else {
        index = r + c * n_rows;
      }
      /*if constexpr (std::is_fundamental<T>::value) {
        double val;
        if constexpr (std::is_same<T, const half>::value) {
          val = double(float(f[index]));
        } else {
          val = double(f[index]);
        }
        printf("..\n");
      } else {*/

        printf("%f, ", float(f[index]));
        /*double val;
        if (std::is_same<decltype(f[index].value), half>::value) {
          val = double(float(f[index].value));
        } else {
          val = double(f[index].value);
        }
        printf("<%ld, %e>, ", int64_t(f[index].key), val);*/
      //}
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


template <typename DataT, typename AccT, typename OutT, typename IdxT>
void ref_l2nn(OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K) {

  for (IdxT m = 0; m < M; m++) {
    IdxT min_index = N + 1;
    double min_dist = std::numeric_limits<double>::max();
    for (IdxT n = 0; n < N; n++) {
      double dist = double(0.0);
      for (IdxT k = 0; k < K; k++) {
        double diff = double(A[m * K + k]) - double(B[n * K + k]);
        dist += (diff * diff);
      }
      if (dist < min_dist) {
        min_dist = dist;
        min_index = n;
      }
    }
    if constexpr (std::is_fundamental<OutT>::value) {
      out[m] = AccT(min_dist);
    } else {
      out[m].key = IdxT(min_index);
      out[m].value = AccT(min_dist);
    }
  }
}

// This is a naive implementation of l2-distance finding nearest neighbour
template <typename DataT, typename AccT, typename OutT, typename IdxT>
__global__ void ref_l2nn_dev(OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K) {
  IdxT tid = threadIdx.x + blockIdx.x * size_t(blockDim.x);
  IdxT n_warps = (size_t(blockDim.x) * gridDim.x) / 32;

  IdxT warp_id = tid / 32;
  IdxT warp_lane = threadIdx.x % 32;
  const int warp_size = 32;

  for (IdxT m = warp_id; m < M; m+=n_warps) {
    __shared__ AccT dist[4];

    IdxT min_index = N + 1;
    AccT min_dist = max_val<AccT>();

    for (IdxT n = 0; n < N; n++) {
      if (warp_lane == 0) {
        dist[warp_id % 4] = AccT(0.0);
      }
      AccT th_dist = AccT(0.0);
      for (IdxT k = warp_lane; k < K; k+=warp_size) {
        AccT diff = AccT(A[m * K + k]) - AccT(B[n * K + k]);
        th_dist += (diff * diff);
      }
      __syncwarp();
      atomicAdd(&dist[warp_id % 4], th_dist);
      __syncwarp();

      if (warp_lane == 0 && dist[warp_id % 4] < min_dist) {
        min_dist = dist[warp_id % 4];
        min_index = n;
      }
    }
    if constexpr (std::is_fundamental<OutT>::value) {
      if (warp_lane == 0) {
        static_assert(std::is_same<OutT, AccT>::value, "OutT and AccT are not same type");
        out[m] = AccT(min_dist);
      }
    } else {
      // output is a raft::KeyValuePair
      if (warp_lane == 0) {
        static_assert(std::is_same<OutT, raft::KeyValuePair<IdxT, AccT>>::value, "OutT is not raft::KeyValuePair<> type");
        out[m].key = IdxT(min_index);
        out[m].value = AccT(min_dist);
      }
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void ref_l2nn_api(OutT* out, const DataT* A, const DataT* B, IdxT m, IdxT n, IdxT k, cudaStream_t stream) {

  //constexpr int block_dim = 128;
  //static_assert(block_dim % 32 == 0, "blockdim must be divisible by 32");
  //constexpr int warps_per_block = block_dim / 32;
  //int num_blocks = m ;
  ref_l2nn_dev<DataT, AccT, OutT, IdxT><<<m/4, 128, 0, stream>>>(out, A, B, m, n, k);
  return;
}

// Structure to track comparison failures
class ComparisonSummary {
public:
  double max_diff;          // Maximum difference found
  uint64_t max_diff_index;   // where does the maximim difference occur
  double max_a;
  double max_b;
  double acc_diff;          // sum of all the diffs
  uint64_t n;               // How many items are compared
  uint64_t n_misses;
  int mutex;                // Simple mutex lock for thread synchronization

  __device__ __host__
  void init() {
    max_diff = 0.0;
    max_diff_index = 0;
    max_a = 0.0;
    max_b = 0.0;
    acc_diff = 0.0;
    n = 0;
    n_misses = 0;
  }

  __device__ __host__
  void update(double diff, uint64_t index, double a_val, double b_val, bool missed) {
    if ( max_diff < diff ) {
      max_diff = diff;
      max_diff_index = index;
      max_a = a_val;
      max_b = b_val;
    }
    acc_diff += diff;
    n++;
    n_misses = missed ? n_misses + 1 : n_misses;
  }

  __device__ __host__
  void update(ComparisonSummary& op2) {
    if ( max_diff < op2.max_diff ) {
      max_diff = op2.max_diff;
      max_diff_index = op2.max_diff_index;
      max_a = op2.max_a;
      max_b = op2.max_b;
    }
    acc_diff += op2.acc_diff;
    n += op2.n;
    n_misses += op2.n_misses;
  }

  __device__ __host__
  void print() {
    if (max_diff > 0.0) {
      printf("Total compared %lu\n", n);
      printf("Total missed %lu\n", n_misses);
      printf("Average diff: %e\n", acc_diff / n);
      printf("max_diff: %e (%e - %e)\n", max_diff, max_a, max_b);
      printf("max_diff_index: %lu\n", max_diff_index);
    }
  }
};

template <typename OutT, typename IdxT>
__global__ void vector_compare_kernel(const OutT* a, const OutT* b, IdxT n,
                                      ComparisonSummary* global_summary) {
  // Shared memory for tracking failures within a block
  __shared__ ComparisonSummary block_summary;

  // Initialize shared memory variable
  if (threadIdx.x == 0) {
    block_summary.init();
  }
  __syncthreads();

  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (IdxT i = 0; i < n; i++) {
    tid = i;
    double diff, a_val, b_val;
    bool missed = false;
    if constexpr (std::is_fundamental_v<OutT> || std::is_same_v<OutT, half>) {
      diff = std::abs(double(a[tid]) - double(b[tid]));
      //printf("Expected = %f vs actual = %f\n", a[tid], b[tid]);
      a_val = double(a[tid]);
      b_val = double(b[tid]);

    } else {
      diff = std::abs(double(a[tid].value) - double(b[tid].value));
      //if (tid == 0) printf("Expected = %f vs actual = %f, %d, %d\n", a[tid].value, b[tid].value, a[tid].key, b[tid].key);
      a_val = double(a[tid].value);
      b_val = double(b[tid].value);

      missed = a[tid].key != b[tid].key;
    }
      // Acquire mutex lock using atomic compare-and-swap
    /*while (atomicCAS(&block_summary.mutex, 0, 1) != 0) {
      // Spin wait
    }*/

    // Critical section: update first_index if this is the earliest failure
    block_summary.update(diff, tid, a_val, b_val, missed);

    // Release mutex
    //atomicExch(&block_summary.mutex, 0);
  }

  __syncthreads();

  // First thread in the block can report the failure if needed
  if (threadIdx.x == 0) {
    // Acquire mutex lock using atomic compare-and-swap
    /*while (atomicCAS(&global_summary->mutex, 0, 1) != 0) {
      // Spin wait
    }*/

    // Critical section: update first_index if this is the earliest failure
    global_summary->update(block_summary);

    // Release mutex
    //atomicExch(&global_summary->mutex, 0);
  }
}

template <typename OutT, typename IdxT>
void vector_compare(ComparisonSummary* global_summary, const OutT* a, const OutT* b, const IdxT n, cudaStream_t stream = nullptr) {
  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;


  //vector_compare_kernel<OutT, IdxT><<<grid_size, block_size, 0, stream>>>(a, b, n, global_summary);
  // Not thread safe right now, so launch only single thread
  vector_compare_kernel<OutT, IdxT><<<1, 1, 0, stream>>>(a, b, n, global_summary);

  CHECK_CUDA(cudaStreamSynchronize(stream));

  //global_summary->print();

  //CHECK_CUDA(cudaFree(global_summary));
}
