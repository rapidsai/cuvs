/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

  dim3 grid = launchConfigGenerator<P>(m, n, shmemSize, kernel);

  kernel<<<grid, blk, shmemSize, stream>>>(
    min, x, y, nullptr, nullptr, m, n, k, maxVal, workspace, redOp, pairRedOp, distance_op, fin_op);

  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
