/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../distance_ops/l2_exp.cuh"     // ops::l2_exp_distance_op
#include "../pairwise_distance_base.cuh"  // PairwiseDistances
#include <raft/core/kvp.hpp>              // raft::KeyValuePair
#include <raft/linalg/contractions.cuh>   // policy

#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits

namespace cuvs::distance::detail {

// TODO(snanditale): specialize this function for min_and_distance_reduce_op<int, float>
// with atomicCAS of 64 bit which will eliminate mutex and shfls
template <typename P, typename OutT, typename IdxT, typename KVPair, typename ReduceOpT>
DI auto update_reduced_val(
  int* mutex, OutT* min, KVPair* val, ReduceOpT red_op, IdxT m, IdxT gridStrideY) -> void
{
  const auto lid      = threadIdx.x % raft::WarpSize;
  const auto accrowid = threadIdx.x / P::AccThCols;

  // Update each output row in order within a warp. This will resolve hang
  // issues with pre-Volta architectures
#pragma unroll
  for (int j = 0; j < (raft::WarpSize / P::AccThCols); j++) {
    if (lid == j * P::AccThCols) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        auto rid = gridStrideY + accrowid + i * P::AccThRows;
        if (rid < m) {
          auto value = val[i];
          while (atomicCAS(mutex + rid, 0, 1) == 1) {
            ;
          }
          __threadfence();
          red_op(rid, min + rid, value);
          __threadfence();
          atomicCAS(mutex + rid, 1, 0);
        }
      }
    }
  }
}

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename P,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename OpT,
          typename FinalLambda>
__launch_bounds__(P::Nthreads, 2) RAFT_KERNEL  // NOLINT(readability-identifier-naming)
  fused_distance_nn_kernel(OutT* min,
                           const DataT* x,
                           const DataT* y,
                           const DataT* xn,
                           const DataT* yn,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           DataT maxVal,
                           int* mutex,
                           ReduceOpT redOp,
                           KVPReduceOpT pairRedOp,
                           OpT distance_op,
                           FinalLambda fin_op)
{
// compile only if below non-ampere arch.
#if __CUDA_ARCH__ < 800
  extern __shared__ char smem[];

  using KVPair = raft::KeyValuePair<IdxT, DataT>;
  KVPair val[P::AccRowsPerTh];
#pragma unroll
  for (int i = 0; i < P::AccRowsPerTh; ++i) {
    val[i] = {0, maxVal};
  }

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [n, pairRedOp, &val, maxVal] __device__(
                         DataT acc[P::AccRowsPerTh][P::AccColsPerTh],
                         DataT * regxn,
                         DataT * regyn,
                         IdxT gridStrideX,
                         IdxT gridStrideY) -> void {
    KVPReduceOpT pair_red_op(pairRedOp);

    // intra thread reduce
    const auto acccolid = threadIdx.x % P::AccThCols;
    const auto accrowid = threadIdx.x / P::AccThCols;
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto tmpkey = acccolid + j * P::AccThCols + gridStrideX;
        KVPair tmp  = {tmpkey, acc[i][j]};
        if (tmpkey < n) {
          val[i] = pair_red_op(accrowid + i * P::AccThRows + gridStrideY, tmp, val[i]);
        }
      }
    }
  };

  auto row_epilog_lambda =
    [m, mutex, min, pairRedOp, redOp, &val, maxVal] __device__(IdxT gridStrideY) -> void {
    KVPReduceOpT pair_red_op(pairRedOp);
    ReduceOpT red_op(redOp);

    const auto accrowid = threadIdx.x / P::AccThCols;
    const auto lid      = raft::laneId();

    // reduce
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = P::AccThCols / 2; j > 0; j >>= 1) {
        // Actually, the srcLane (lid +j) should be (lid +j) % P:AccThCols,
        // but the shfl op applies the modulo internally.
        auto tmpkey   = raft::shfl(val[i].key, lid + j, P::AccThCols);
        auto tmpvalue = raft::shfl(val[i].value, lid + j, P::AccThCols);
        KVPair tmp    = {tmpkey, tmpvalue};
        val[i]        = pair_red_op(accrowid + i * P::AccThRows + gridStrideY, tmp, val[i]);
      }
    }

    update_reduced_val<P, OutT, IdxT, KVPair, ReduceOpT>(mutex, min, val, red_op, m, gridStrideY);

    // reset the val array.
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      val[i] = {0, maxVal};
    }
  };

  IdxT lda = k, ldb = k, ldd = n;
  constexpr bool kRowMajor = true;
  constexpr bool kWriteOut = false;
  pairwise_distances<DataT,
                     DataT,  // OutT (unused in pairwise_distances)
                     IdxT,
                     P,
                     decltype(distance_op),
                     decltype(epilog_lambda),
                     FinalLambda,
                     decltype(row_epilog_lambda),
                     kRowMajor,
                     kWriteOut>
    obj(x,
        y,
        m,
        n,
        k,
        lda,
        ldb,
        ldd,
        xn,
        yn,
        nullptr,  // Output pointer
        smem,
        distance_op,
        epilog_lambda,
        fin_op,
        row_epilog_lambda);
  obj.run();
#endif
}

}  // namespace cuvs::distance::detail
