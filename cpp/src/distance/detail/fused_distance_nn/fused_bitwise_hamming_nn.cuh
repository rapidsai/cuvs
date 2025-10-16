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

#pragma once

#include "../distance_ops/bitwise_hamming.cuh"  // ops::bitwise_hamming_distance_op
#include "../pairwise_distance_base.cuh"        // PairwiseDistances
#include "helper_structs.cuh"
#include "simt_kernel.cuh"

namespace cuvs {
namespace distance {
namespace detail {

/**
 * @brief Fused BitwiseHamming distance and 1-nearest-neighbor
 *
 * This implementation is only meaningful for uint8_t data type.
 * The if constexpr in fusedDistanceNNImpl ensures it's only called for uint8_t.
 */
template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedBitwiseHammingNN(OutT* min,
                           const DataT* x,
                           const DataT* y,
                           const DataT* xn,
                           const DataT* yn,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           int* workspace,
                           ReduceOpT redOp,
                           KVPReduceOpT pairRedOp,
                           bool sqrt,
                           cudaStream_t stream)
{
  typedef Policy P;

  dim3 blk(P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  typedef ::raft::KeyValuePair<IdxT, OutT> KVPair;

  ops::bitwise_hamming_distance_op<DataT, uint32_t, IdxT> distance_op{k};

  ::raft::identity_op fin_op{};

  auto kernel = fusedDistanceNNkernel<DataT,
                                      OutT,
                                      IdxT,
                                      P,
                                      ReduceOpT,
                                      KVPReduceOpT,
                                      decltype(distance_op),
                                      decltype(fin_op)>;

  constexpr size_t shmemSize = P::SmemSize;

  cudaError_t prior_error = cudaGetLastError();
  if (prior_error != cudaSuccess) {
    RAFT_LOG_INFO("Prior CUDA error before fusedDistanceNN: %s", cudaGetErrorString(prior_error));
    RAFT_CUDA_TRY(prior_error);
  }

  dim3 grid = launchConfigGenerator<P>(m, n, shmemSize, kernel);

  kernel<<<grid, blk, shmemSize, stream>>>(
    min, x, y, nullptr, nullptr, m, n, k, maxVal, workspace, redOp, pairRedOp, distance_op, fin_op);

  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
