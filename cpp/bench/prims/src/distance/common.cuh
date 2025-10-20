#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>

#include <cuvs/distance/distance.hpp>
using cuvs::distance::DistanceType;

template <typename T>
__host__ __device__ T max_val()
{
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MAX_NORMAL_FP16;
  } else {
    return cuda::std::numeric_limits<T>::max();
  }
}

template <typename T>
T min_val()
{
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MIN_DENORM_FP16;
  } else {
    return cuda::std::numeric_limits<T>::min();
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
  CudaEventTimer(cudaStream_t stream = nullptr)
    : stream_(stream), elapsed_ms_(0.0f), timing_started_(false)
  {
    RAFT_CUDA_TRY(cudaEventCreate(&start_));
    RAFT_CUDA_TRY(cudaEventCreate(&stop_));
  }

  ~CudaEventTimer()
  {
    RAFT_CUDA_TRY(cudaEventDestroy(start_));
    RAFT_CUDA_TRY(cudaEventDestroy(stop_));
  }

  void start()
  {
    RAFT_CUDA_TRY(cudaEventRecord(start_, stream_));
    timing_started_ = true;
  }

  void stop()
  {
    if (!timing_started_) {
      std::cerr << "Warning: Timer stopped without being started" << std::endl;
      return;
    }
    RAFT_CUDA_TRY(cudaEventRecord(stop_, stream_));
    RAFT_CUDA_TRY(cudaEventSynchronize(stop_));
    RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed_ms_, start_, stop_));
    timing_started_ = false;
  }

  float elapsed_millis() const { return elapsed_ms_; }

  float elapsed_seconds() const { return elapsed_ms_ / 1000.0f; }
};

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void ref_l2nn(OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K, bool sqrt)
{
  for (IdxT m = 0; m < M; m++) {
    IdxT min_index  = N + 1;
    double min_dist = std::numeric_limits<double>::max();
    for (IdxT n = 0; n < N; n++) {
      double dist = double(0.0);
      for (IdxT k = 0; k < K; k++) {
        double diff = double(A[m * K + k]) - double(B[n * K + k]);
        dist += (diff * diff);
      }
      if (dist < min_dist) {
        min_dist  = dist;
        min_index = n;
      }
    }
    if constexpr (std::is_fundamental<OutT>::value) {
      out[m] = AccT(min_dist);
    } else {
      out[m].key   = IdxT(min_index);
      out[m].value = AccT(min_dist);
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
__device__ AccT l2_distance(const DataT* v1, const DataT* v2, IdxT K)
{
  AccT th_dist = AccT(0.0);
  for (IdxT k = 0; k < K; k++) {
    AccT diff = AccT(v1[k]) - AccT(v2[k]);
    th_dist += (diff * diff);
  }
  return th_dist;
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
__device__ AccT cosine_distance(const DataT* v1, const DataT* v2, IdxT K)
{
  AccT v1_norm = AccT(0.0);
  AccT v2_norm = AccT(0.0);
  AccT v1v2    = AccT(0.0);

  for (IdxT k = 0; k < K; k++) {
    v1_norm += (AccT(v1[k]) * AccT(v1[k]));
    v2_norm += (AccT(v2[k]) * AccT(v2[k]));
    v1v2 += (AccT(v1[k]) * AccT(v2[k]));
  }

  return AccT(1.0) - (v1v2 / (v1_norm * v2_norm));
}
// This is a naive implementation of l2-distance finding nearest neighbour
template <typename DataT, typename AccT, typename OutT, typename IdxT>
__global__ void ref_l2nn_dev(
  OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K, bool sqrt, DistanceType metric)
{
  IdxT tid = threadIdx.x + blockIdx.x * size_t(blockDim.x);

  for (IdxT m = tid; m < M; m += (blockDim.x * gridDim.x)) {
    IdxT min_index = N + 1;
    AccT min_dist  = max_val<AccT>();

    for (IdxT n = 0; n < N; n++) {
      AccT dist;
      if (metric == DistanceType::L2SqrtExpanded || metric == DistanceType::L2Expanded) {
        dist = l2_distance<DataT, AccT, OutT, IdxT>(&A[m * K], &B[n * K], K);
      } else if (metric == DistanceType::CosineExpanded) {
        dist = cosine_distance<DataT, AccT, OutT, IdxT>(&A[m * K], &B[n * K], K);
      }
      if (dist < min_dist) {
        min_dist  = dist;
        min_index = n;
      }
    }

    if constexpr (std::is_fundamental<OutT>::value) {
      static_assert(std::is_same<OutT, AccT>::value, "OutT and AccT are not same type");
      out[m] = AccT(min_dist);
    } else {
      // output is a raft::KeyValuePair
      static_assert(std::is_same<OutT, raft::KeyValuePair<IdxT, AccT>>::value,
                    "OutT is not raft::KeyValuePair<> type");
      out[m].key = IdxT(min_index);
      if (sqrt) {
        out[m].value = raft::sqrt(AccT(min_dist));
      } else {
        out[m].value = AccT(min_dist);
      }
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void ref_l2nn_api(OutT* out,
                  const DataT* A,
                  const DataT* B,
                  IdxT m,
                  IdxT n,
                  IdxT k,
                  bool sqrt,
                  DistanceType metric,
                  cudaStream_t stream)
{
  // constexpr int block_dim = 128;
  // static_assert(block_dim % 32 == 0, "blockdim must be divisible by 32");
  // constexpr int warps_per_block = block_dim / 32;
  // int num_blocks = m ;
  ref_l2nn_dev<DataT, AccT, OutT, IdxT>
    <<<(m + 127) / 128, 128, 0, stream>>>(out, A, B, m, n, k, sqrt, metric);
  return;
}

// Structure to track comparison failures
class ComparisonSummary {
 public:
  double max_diff;          // Maximum difference found
  uint64_t max_diff_index;  // where does the maximum difference occur
  double max_a;
  double max_b;
  double acc_diff;  // sum of all the diffs
  uint64_t n;       // How many items are compared
  uint64_t n_misses;
  int mutex;  // Simple mutex lock for thread synchronization

  __device__ __host__ void init()
  {
    max_diff       = 0.0;
    max_diff_index = 0;
    max_a          = 0.0;
    max_b          = 0.0;
    acc_diff       = 0.0;
    n              = 0;
    n_misses       = 0;
  }

  __device__ __host__ void update(
    double diff, uint64_t index, double a_val, double b_val, bool missed)
  {
    if (max_diff < diff) {
      max_diff       = diff;
      max_diff_index = index;
      max_a          = a_val;
      max_b          = b_val;
    }
    acc_diff += diff;
    n++;
    n_misses = missed ? n_misses + 1 : n_misses;
  }

  __device__ __host__ void update(ComparisonSummary& op2)
  {
    if (max_diff < op2.max_diff) {
      max_diff       = op2.max_diff;
      max_diff_index = op2.max_diff_index;
      max_a          = op2.max_a;
      max_b          = op2.max_b;
    }
    acc_diff += op2.acc_diff;
    n += op2.n;
    n_misses += op2.n_misses;
  }

  __device__ __host__ void print()
  {
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
__global__ void vector_compare_kernel(const OutT* a,
                                      const OutT* b,
                                      IdxT n,
                                      ComparisonSummary* global_summary)
{
  // Shared memory for tracking failures within a block
  __shared__ ComparisonSummary block_summary;

  // Initialize shared memory variable
  if (threadIdx.x == 0) { block_summary.init(); }
  __syncthreads();

  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (IdxT i = 0; i < n; i++) {
    tid = i;
    double diff, a_val, b_val;
    bool missed = false;
    if constexpr (std::is_fundamental_v<OutT> || std::is_same_v<OutT, half>) {
      diff = std::abs(double(a[tid]) - double(b[tid]));
      // printf("Expected = %f vs actual = %f\n", a[tid], b[tid]);
      a_val = double(a[tid]);
      b_val = double(b[tid]);

    } else {
      diff = std::abs(double(a[tid].value) - double(b[tid].value));
      // if (tid == 0) printf("Expected = %f vs actual = %f, %d, %d\n", a[tid].value, b[tid].value,
      // a[tid].key, b[tid].key);
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
    // atomicExch(&block_summary.mutex, 0);
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
    // atomicExch(&global_summary->mutex, 0);
  }
}

template <typename OutT, typename IdxT>
void vector_compare(ComparisonSummary* global_summary,
                    const OutT* a,
                    const OutT* b,
                    const IdxT n,
                    cudaStream_t stream = nullptr)
{
  constexpr int block_size = 256;
  const int grid_size      = (n + block_size - 1) / block_size;

  // vector_compare_kernel<OutT, IdxT><<<grid_size, block_size, 0, stream>>>(a, b, n,
  // global_summary);
  //  Not thread safe right now, so launch only single thread
  vector_compare_kernel<OutT, IdxT><<<1, 1, 0, stream>>>(a, b, n, global_summary);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  // global_summary->print();

  // RAFT_CUDA_TRY(cudaFree(global_summary));
}
