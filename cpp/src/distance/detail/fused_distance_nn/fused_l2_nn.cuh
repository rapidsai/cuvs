/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../distance_ops/l2_exp.cuh"     // ops::l2_exp_distance_op
#include "../pairwise_distance_base.cuh"  // PairwiseDistances
#include "cutlass_base.cuh"
#include "helper_structs.cuh"
#include "simt_kernel.cuh"
#include <raft/core/kvp.hpp>             // raft::KeyValuePair
#include <raft/core/operators.hpp>       // raft::identity_op
#include <raft/linalg/contractions.cuh>  // policy
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
void fused_l2_nn_impl(OutT* min,
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
                      bool initOutBuffer,
                      cudaStream_t stream)
{
  // The kernel policy is determined by fusedL2NN.
  using policy_t = Policy;

  dim3 blk(policy_t::Nthreads);
  auto nblks             = raft::ceildiv<int>(m, policy_t::Nthreads);
  constexpr auto kMaxVal = std::numeric_limits<DataT>::max();
  using kv_pair_t        = raft::KeyValuePair<IdxT, DataT>;

  if (initOutBuffer) {
    init_kernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, policy_t::Nthreads, 0, stream>>>(min, m, kMaxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  namespace arch = raft::util::arch;
  using acc_t    = DataT;
  ops::l2_exp_distance_op<DataT, acc_t, IdxT> distance_op{sqrt};

  raft::identity_op fin_op{};

  auto kernel = fused_distance_nn_kernel<DataT,
                                         OutT,
                                         IdxT,
                                         policy_t,
                                         ReduceOpT,
                                         KVPReduceOpT,
                                         decltype(distance_op),
                                         decltype(fin_op)>;

  // Get pointer to fp32 SIMT kernel to determine the best compute architecture
  // out of all for which the kernel was compiled for that matches closely
  // to the current device. Other methods to determine the architecture (that do not
  // require a pointer) can be error prone. See:
  // https://github.com/NVIDIA/cub/issues/545
  void* kernel_ptr   = reinterpret_cast<void*>(kernel);
  auto runtime_arch  = arch::kernel_virtual_arch(kernel_ptr);
  auto cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());

  if (cutlass_range.contains(runtime_arch)) {
    // If device is SM_80 or later, use CUTLASS-based kernel.
    using l2_op_t                = cuvs::distance::detail::ops::l2_exp_cutlass_op<DataT, DataT>;
    using kvp_cg_min_reduce_op_t = kvp_cg_min_reduce_op<DataT, IdxT, OutT>;
    kvp_cg_min_reduce_op_t cg_reduce_op;
    l2_op_t l2_dist_op(sqrt);

    IdxT lda, ldb, ldd;
    lda = k, ldb = k, ldd = n;

    cutlass_fused_distance_nn<DataT,
                              DataT,
                              OutT,
                              IdxT,
                              policy_t::Veclen,
                              kvp_cg_min_reduce_op_t,
                              l2_op_t,
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
                                            l2_dist_op,
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
