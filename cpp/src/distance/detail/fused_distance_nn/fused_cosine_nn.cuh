/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../distance_ops/cosine.cuh"     // ops::l2_exp_distance_op
#include "../pairwise_distance_base.cuh"  // PairwiseDistances
#include "cutlass_base.cuh"
#include "helper_structs.cuh"
#include "simt_kernel.cuh"
#include <raft/core/kvp.hpp>             // raft::KeyValuePair
#include <raft/core/operators.hpp>       // raft::identity_op
#include <raft/linalg/contractions.cuh>  // Policy
#include <raft/util/arch.cuh>            // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>      // raft::ceildiv, raft::shfl

#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits

namespace cuvs::distance::detail {

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fused_cosine_nn(OutT* min,
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
  // The kernel policy is determined by fusedL2NN.
  using policy_t = Policy;

  dim3 blk(policy_t::Nthreads);
  constexpr auto kMaxVal = std::numeric_limits<DataT>::max();
  using kv_pair_t        = raft::KeyValuePair<IdxT, DataT>;

  namespace arch = raft::util::arch;
  using acc_t    = DataT;
  ops::cosine_distance_op<DataT, acc_t, IdxT> distance_op{};

  raft::identity_op fin_op{};

  auto kernel =
    fused_distance_nn_kernel<DataT,
                             OutT,
                             IdxT,
                             policy_t,
                             ReduceOpT,
                             KVPReduceOpT,
                             decltype(distance_op),
                             decltype(fin_op)>;  // NOLINT(readability-identifier-naming)

  // Get pointer to fp32 SIMT kernel to determine the runtime architecture of the
  // current system. Other methods to determine the architecture (that do not
  // require a pointer) can be error prone. See:
  // https://github.com/NVIDIA/cub/issues/545
  void* kernel_ptr   = reinterpret_cast<void*>(kernel);
  auto runtime_arch  = arch::kernel_virtual_arch(kernel_ptr);
  auto cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());

  if (cutlass_range.contains(runtime_arch)) {
    // If device is SM_80 or later, use CUTLASS-based kernel.
    using cosine_op_t            = cuvs::distance::detail::ops::cosine_cutlass_op<DataT, DataT>;
    using kvp_cg_min_reduce_op_t = kvp_cg_min_reduce_op<DataT, IdxT, OutT>;
    kvp_cg_min_reduce_op_t cg_reduce_op;
    cosine_op_t cosine_dist_op;

    IdxT lda, ldb, ldd;
    lda = k, ldb = k, ldd = n;

    cutlass_fused_distance_nn<DataT,
                              DataT,
                              OutT,
                              IdxT,
                              policy_t::Veclen,
                              decltype(cg_reduce_op),
                              decltype(cosine_dist_op),
                              ReduceOpT,
                              KVPReduceOpT>(x,
                                            y,
                                            xn,
                                            yn,
                                            m,
                                            n,
                                            k,
                                            lda,
                                            ldb,
                                            ldd,
                                            min,
                                            workspace,
                                            cg_reduce_op,
                                            cosine_dist_op,
                                            redOp,
                                            pairRedOp,
                                            stream);
  } else {
    // If device less than SM_80, use fp32 SIMT kernel.
    constexpr size_t kShmemSize =
      policy_t::SmemSize + ((policy_t::Mblk + policy_t::Nblk) * sizeof(DataT));
    dim3 grid = launch_config_generator<policy_t>(m, n, kShmemSize, kernel);

    kernel<<<grid, blk, kShmemSize, stream>>>(
      min, x, y, xn, yn, m, n, k, kMaxVal, workspace, redOp, pairRedOp, distance_op, fin_op);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace cuvs::distance::detail
