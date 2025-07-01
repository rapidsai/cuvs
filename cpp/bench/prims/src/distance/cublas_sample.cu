/**
 * @file cublas_sample.cu
 * @brief Sample code demonstrating cuBLAS GEMM (General Matrix Multiplication)
 */

#include <cublas_v2.h>
#include <benchmark/benchmark.h>
#include "common.cuh"

template <typename AccT, typename IdxT>
struct Reducer {
    typedef raft::KeyValuePair<IdxT, AccT> KVType;

    __device__ KVType operator()(const KVType& a, const KVType& b) {
        if ((a.value < b.value) || (a.value == b.value && a.key < b.key)) {
          return a;
        } else {
          return b;
        }
    }

    __device__ AccT operator()(const AccT& a, const AccT& b) {
      return a < b ? a : b;
    }
};

template <typename DataT, typename AccT, typename OutT, typename IdxT, int TPB>
__global__ void reduce_min_kernel(OutT* out, const AccT* z, const AccT* x_norm, const AccT* y_norm, IdxT m, IdxT n) {
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT row = blockIdx.x;

  AccT x_norm_row = x_norm[row];

  raft::KeyValuePair<IdxT, AccT> thread_min;

  thread_min.value = max_val<AccT>();
  thread_min.key= max_val<IdxT>();

  for (IdxT col = threadIdx.x; col < n; col+=TPB) {
      auto dist = x_norm_row + y_norm[col] - AccT(2)*z[row*n + col];
      //if (row == 0 && (col == 112 || col == 74)) {
      //  printf("row = %d, col = %d, dist = %f, %d\n", row, col, z[row*n+col], threadIdx.x);
      //}
      if (dist < thread_min.value) {
        thread_min.value = dist;
        thread_min.key = col;
      }
  }
  typedef cub::BlockReduce<OutT, TPB> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  auto block_result = BlockReduceT(temp_storage).Reduce(thread_min, Reducer<AccT, IdxT>{});

  if (threadIdx.x == 0) {
    out[row] = block_result;
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void reduce_min(OutT* out, const AccT* z, const AccT* x_norm, const AccT* y_norm, IdxT m, IdxT n, cudaStream_t stream) {
  const int TPB = 128;

  int blocks = m;
  reduce_min_kernel<DataT, AccT, OutT, IdxT, TPB><<<blocks, TPB, 0, stream>>>(out, z, x_norm, y_norm, m, n);
}

template <typename DataT, typename AccT, typename OutT, typename IdxT, bool FAST_MODE, bool GEMM_ONLY>
void cublas_l2nn(OutT* out, const DataT* x, const DataT* y, IdxT M, IdxT N, IdxT K,
                 AccT* x_norm, AccT* y_norm, AccT* z, cublasHandle_t& handle, cudaStream_t stream) {

  // Enable TF32 mode
  //CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  cudaDataType_t xyType, zType;
  cublasComputeType_t computeType;

  const half h_alpha = 1.0, h_beta = 0.0;
  const float f_alpha = 1.0, f_beta = 0.0;
  const double d_alpha = 1.0, d_beta = 0.0;
  const int32_t i32_alpha = 1, i32_beta = 0;
  const int8_t i8_alpha = 1, i8_beta = 0;

  const void* alpha = nullptr;
  const void* beta = nullptr;
  if constexpr (std::is_same_v<DataT, float>) {
    xyType = CUDA_R_32F;
    zType = CUDA_R_32F;
    computeType = CUBLAS_COMPUTE_32F;
    // computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    // computeType = CUBLAS_COMPUTE_32F_FAST_16F;
    // computeType = CUBLAS_COMPUTE_32F_FAST_16BF;
    // computeType = CUBLAS_COMPUTE_32F_EMULATED_16BFX9;
    alpha = reinterpret_cast<const void*>(&f_alpha);
    beta = reinterpret_cast<const void*>(&f_beta);
  } else if constexpr (std::is_same_v<DataT, double>) {
    xyType = CUDA_R_64F;
    zType = CUDA_R_64F;
    computeType = CUBLAS_COMPUTE_64F;
    alpha = reinterpret_cast<const void*>(&d_alpha);
    beta = reinterpret_cast<const void*>(&d_beta);
  } else if constexpr (std::is_same_v<DataT, half>) {
    xyType = CUDA_R_16F;
    if constexpr (std::is_same_v<AccT, half>) {
      zType = CUDA_R_16F;
      computeType = CUBLAS_COMPUTE_16F;
      //computeType = CUBLAS_COMPUTE_32F;
      alpha = reinterpret_cast<const void*>(&h_alpha);
      beta = reinterpret_cast<const void*>(&h_beta);
    } else if constexpr (std::is_same_v<AccT, float>) {
      zType = CUDA_R_32F;
      computeType = CUBLAS_COMPUTE_32F;
      alpha = reinterpret_cast<const void*>(&f_alpha);
      beta = reinterpret_cast<const void*>(&f_beta);
    }
  } else if constexpr (std::is_same_v<DataT, int8_t>) {
    xyType = CUDA_R_8I;
    if constexpr (std::is_same_v<AccT, int32_t>) {
      zType = CUDA_R_32I;
      computeType = CUBLAS_COMPUTE_32I;
      alpha = reinterpret_cast<const void*>(&i32_alpha);
      beta = reinterpret_cast<const void*>(&i32_beta);
    } else if constexpr (std::is_same_v<AccT, float>) {
      zType = CUDA_R_32F;
      computeType = CUBLAS_COMPUTE_32F;
      alpha = reinterpret_cast<const void*>(&i8_alpha);
      beta = reinterpret_cast<const void*>(&i8_beta);
    }
  }

  CHECK_CUBLAS(cublasGemmEx(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      N, M, K,                   // Dimensions (swapped due to row/col-major difference)
      alpha,                    // alpha
      y, xyType, K,            // B, its data type, and leading dimension
      x, xyType, K,            // A, its data type, and leading dimension
      beta,                     // beta
      z, zType, N,            // C, its data type, and leading dimension
      computeType,               // Computation type
      CUBLAS_GEMM_DEFAULT        // Algorithm selection
    ));

  /*print_kernel<half><<<1, 1>>>(x, 1, 10);
  print_kernel<half><<<1, 1>>>(y, 1, 10);
  print_kernel<float><<<1, 1>>>(z, 1, 10);
  CHECK_CUDA(cudaDeviceSynchronize());*/
  if constexpr(!GEMM_ONLY) {
    reduce_min<DataT, AccT, OutT, IdxT>(out, z, x_norm, y_norm, M, N, stream);
  }
}
// 
