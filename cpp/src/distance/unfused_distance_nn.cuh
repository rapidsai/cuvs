/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __UNFUSED_DISTANCE_NN_H
#define __UNFUSED_DISTANCE_NN_H

#pragma once

//#include "detail/fused_distance_nn.cuh"
//#include "fused_distance_nn_helpers.cuh"
#include <raft/core/resources.hpp>
#include <raft/linalg/contractions.cuh>
#include <raft/util/cuda_utils.cuh>

#include <cub/cub.cuh>

#include <stdint.h>

#include <limits>
#include <type_traits>
#include <cublas_v2.h>

namespace cuvs {
namespace distance {

/**
 * \ingroup unfused_l2_nn
 * @{
 */
/**
 * @brief Unfused L2 distance and 1-nearest-neighbor computation in a single call.
 *
 * Unlike the fused call, here we do explicit gemm call followed by a reduction call.
 * This code path exists because the fused path is sometime not the optimal due to
 * GEMM-like kernel not being optimized yet.
 *
 *
 * @tparam DataT      data type
 * @tparam OutT       output type to either store 1-NN indices and their minimum
 *                    distances or store only the min distances. Accordingly, one
 *                    has to pass an appropriate `ReduceOpT`
 * @tparam IdxT       indexing arithmetic type
 * @tparam ReduceOpT  A struct to perform the final needed reduction operation
 *                    and also to initialize the output array elements with the
 *                    appropriate initial value needed for reduction.
 * @tparam KVPReduceOpT A struct providing functions for key-value pair comparison.
 *
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(int)*m. (on device)
 * @param[in]  redOp         reduction operator in the epilogue
 * @param[in]  pairRedOp     reduction operation on key value pairs
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  isRowMajor    whether the input/output is row or column major.
 * @param[in]  metric        Distance metric to be used (supports L2, cosine)
 * @param[in]  metric_arg    power argument for distances like Minkowski (not supported for now)
 * @param[in]  stream        cuda stream
 */


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
__global__ void reduce_min_kernel(OutT* out, const AccT* z, const AccT* x_norm, const AccT* y_norm, IdxT m, IdxT n, bool is_sqrt) {
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT row = blockIdx.x;

  AccT x_norm_row = x_norm[row];

  raft::KeyValuePair<IdxT, AccT> thread_min;

  thread_min.value = max_val<AccT>();
  thread_min.key= max_val<IdxT>();

  for (IdxT col = threadIdx.x; col < n; col+=TPB) {
      auto dist = x_norm_row + y_norm[col] - AccT(2)*z[row*n + col];
      if (dist < thread_min.value) {
        thread_min.value = dist;
        thread_min.key = col;
      }
  }
  typedef cub::BlockReduce<OutT, TPB> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  auto block_result = BlockReduceT(temp_storage).Reduce(thread_min, Reducer<AccT, IdxT>{});

  if (threadIdx.x == 0) {
    if (is_sqrt) {
      out[row] = block_result;
    } else {
      out[row] = block_result;
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void reduce_min(OutT* out, const AccT* z, const AccT* x_norm, const AccT* y_norm, IdxT m, IdxT n, cudaStream_t stream, bool is_sqrt) {
  const int TPB = 128;

  int blocks = m;
  reduce_min_kernel<DataT, AccT, OutT, IdxT, TPB><<<blocks, TPB, 0, stream>>>(out, z, x_norm, y_norm, m, n, is_sqrt);
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void cublas_l2nn(OutT* out, const DataT* x, const DataT* y, IdxT M, IdxT N, IdxT K,
                 const AccT* x_norm, const AccT* y_norm, AccT* workspace, bool is_sqrt, cublasHandle_t& cublas_h, cudaStream_t stream) {

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
      cublas_h,
      CUBLAS_OP_T, CUBLAS_OP_N,
      N, M, K,                   // Dimensions (swapped due to row/col-major difference)
      alpha,                    // alpha
      y, xyType, K,            // B, its data type, and leading dimension
      x, xyType, K,            // A, its data type, and leading dimension
      beta,                     // beta
      workspace, zType, N,            // C, its data type, and leading dimension
      computeType,               // Computation type
      CUBLAS_GEMM_DEFAULT        // Algorithm selection
    ));

    reduce_min<DataT, AccT, OutT, IdxT>(out, workspace, x_norm, y_norm, M, N, stream, is_sqrt);
}
//

/**
 * @brief Wrapper around fusedDistanceNN with minimum reduction operators.
 *
 * fusedDistanceNN cannot be compiled in the distance library due to the lambda
 * operators, so this wrapper covers the most common case (minimum).
 *
 * @tparam DataT     data type
 * @tparam OutT      output type to either store 1-NN indices and their minimum
 *                   distances (e.g. raft::KeyValuePair<int, float>) or store only the min
 * distances.
 * @tparam IdxT      indexing arithmetic type
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(int)*m. (on device)
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  isRowMajor    whether the input/output is row or column major.
 * @param[in]  metric        Distance metric to be used (supports L2, cosine)
 * @param[in]  metric_arg    power argument for distances like Minkowski (not supported for now)
 * @param[in]  stream        cuda stream
 */
template <typename DataT, typename OutT, typename IdxT>
void unfusedDistanceNNMinReduce(OutT* min,
                              const DataT* x,
                              const DataT* y,
                              const DataT* xn,
                              const DataT* yn,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              void* workspace,
                              bool is_sqrt,
                              bool initOutBuffer,
                              bool isRowMajor,
                              cuvs::distance::DistanceType metric,
                              float metric_arg,
                              cudaStream_t stream,
                              cublasHandle_t& cublas_h)
{
  cublas_l2nn<DataT, DataT, OutT, IdxT>(min, x, y, m, n, k, xn, yn, (DataT*)workspace, is_sqrt, cublas_h, stream);
}

/** @} */

}  // namespace distance
}  // namespace cuvs

#endif
