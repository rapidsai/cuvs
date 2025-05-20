/**
 * @file cublas_sample.cu
 * @brief Sample code demonstrating cuBLAS GEMM (General Matrix Multiplication)
 */

#include <cublas_v2.h>
#include <benchmark/benchmark.h>
#include "common.cuh"

template <typename DataT, typename IdxT>
struct Reducer {
    typedef raft::KeyValuePair<IdxT, DataT> KVType;

    __device__ KVType operator()(const KVType& a, const KVType& b) {
        if ((a.value < b.value) || (a.value == b.value && a.key < b.key)) {
          return a;
        } else {
          return b;
        }
    }

    __device__ DataT operator()(const DataT& a, const DataT& b) {
      return a < b? a : b;
    }
};

template <typename OutT, typename DataT, typename IdxT, int TPB>
__global__ void reduce_min_kernel(OutT* out, const DataT* z, const DataT* x_norm, const DataT* y_norm, IdxT m, IdxT n) {
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT row = blockIdx.x;

  DataT x_norm_row = x_norm[row];

  raft::KeyValuePair<IdxT, DataT> thread_min;

  thread_min.value = max_val<DataT>();
  thread_min.key= max_val<IdxT>();

  for (IdxT col = threadIdx.x; col < n; col+=TPB) {
      auto dist = x_norm_row + y_norm[col] - 2*z[row*n + col];
      if (dist < thread_min.value) {
        thread_min.value = dist;
        thread_min.key = col;
      }
  }
  typedef cub::BlockReduce<OutT, TPB> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  auto block_result = BlockReduceT(temp_storage).Reduce(thread_min, Reducer<DataT, IdxT>{});

  if (threadIdx.x == 0) {
    out[row] = block_result;
  }
}


template <typename OutT, typename DataT, typename IdxT>
void neo_reduce_min(OutT* out, const DataT* z, const DataT* x_norm, const DataT* y_norm, IdxT m, IdxT n, cudaStream_t stream) {
  const int TPB = 128;

  int blocks = m;
  reduce_min_kernel<OutT, DataT, IdxT, TPB><<<blocks, TPB, 0, stream>>>(out, z, x_norm, y_norm, m, n);
  CHECK_CUDA(cudaDeviceSynchronize());
}
template <typename OutT, typename DataT, typename IdxT, bool GEMM_ONLY>
void cublas_l2nn(OutT* out, const DataT* x, const DataT* y, IdxT M, IdxT N, IdxT K,
                 DataT* x_norm, DataT* y_norm,
                 DataT* z, size_t ws_size, cublasHandle_t& handle, cudaStream_t stream) {
  // Set up scaling factors
  const DataT alpha = 1.0f;
  const DataT beta = 0.0f;

  // Enable TF32 mode
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  if constexpr (std::is_same_v<DataT, float>) {
    CHECK_CUBLAS(cublasSgemm(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      N, M, K,                   // Dimensions (swapped due to row/col-major difference)
      &alpha,                    // alpha
      y, K,                      // B and its leading dimension
      x, K,                      // A and its leading dimension
      &beta,                     // beta
      z, N                       // C and its leading dimension
    ));
  } else if constexpr (std::is_same_v<DataT, double>) {
    CHECK_CUBLAS(cublasDgemm(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      N, M, K,                   // Dimensions (swapped due to row/col-major difference)
      &alpha,                    // alpha
      y, K,                      // B and its leading dimension
      x, K,                      // A and its leading dimension
      &beta,                     // beta
      z, N                       // C and its leading dimension
    ));
  }

  if constexpr(!GEMM_ONLY) {
    reduce_min<OutT, DataT, IdxT>(out, z, x_norm, y_norm, M, N, stream);
  }
}

