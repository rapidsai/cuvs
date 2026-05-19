/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

/* This file has two responsibilities:
 *
 * 1. Dispatch to the correct implementation of a kernel based on the
 *    architecture of the device on which the kernel will be launched. For
 *    instance, the cosine distance has a CUTLASS-based implementation that can
 *    be used on SM80+ and the JIT implementation that is used on older
 *    architectures.
 *
 * 2. Provide concise function templates that can be instantiated in
 *    src/distance/detail/pairwise_matrix/. Previously,
 *    cuvs::distance::detail::distance was instantiated. The function
 *    necessarily required a large set of include files, which slowed down the
 *    build.
 */

#include "../distance_ops/cutlass.cuh"                               // ops::has_cutlass_op
#include "../pairwise_matrix/jit_lto_kernels/pairwise_matrix_jit.cuh"  // pairwise_matrix_jit_dispatch
#include <raft/core/error.hpp>                                      // RAFT_CUDA_TRY
#include <raft/util/arch.cuh>                                       // raft::util::arch::SM_*

// NOTE: to minimize compile times, we do not include dispatch_sm80.cuh.
// Including dispatch_sm80.cuh can slow down compile times (due to CUTLASS).
// Therefore, it is the including file's responsibility to include
// dispatch_sm80.cuh for CUTLASS-backed distance ops, as is done in
// src/distance/detail/pairwise_matrix/dispatch_*.cu.

namespace cuvs::distance::detail {

// This forward-declaration ensures that we do not need to include
// dispatch_sm80.cuh if we are not calling it in practice. This makes compiling
// all the non-CUTLASS based distance instantiations faster. For CUTLASS-based
// distances, dispatch_sm80.cuh has to be included by the file including this
// file.
template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
void pairwise_matrix_sm80_dispatch(OpT,
                                   pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>,
                                   SM_compat_t,
                                   cudaStream_t);

inline auto current_device_arch()
{
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));

  int major = 0;
  int minor = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  return raft::util::arch::SM(major * 10 + minor);
}

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void pairwise_matrix_dispatch(OpT distance_op,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              const DataT* x,
                              const DataT* y,
                              const OutT* x_norm,
                              const OutT* y_norm,
                              OutT* out,
                              FinOpT fin_op,
                              cudaStream_t stream,
                              bool is_row_major)
{
  // Create kernel parameter struct. Flip x and y if column major.
  IdxT ldx    = is_row_major ? k : m;
  IdxT ldy    = is_row_major ? k : n;
  IdxT ld_out = is_row_major ? n : m;

  pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params{
    m, n, k, ldx, ldy, ld_out, x, y, x_norm, y_norm, out, fin_op, is_row_major};

  if (!params.is_row_major) { params.flip_x_and_y(); }

  // Dispatch rule:
  // - execute CUTLASS-based kernel on SM_80 and above when the op supports it
  // - execute JIT kernel otherwise
  namespace arch = raft::util::arch;

  constexpr bool cutlass_op_unavailable = !ops::has_cutlass_op<OpT>::value;

  if constexpr (cutlass_op_unavailable) {
    pairwise_matrix_jit_dispatch(distance_op, params, stream);
  } else {
    auto cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());
    auto runtime_arch  = current_device_arch();

    // TODO: CUTLASS does not support odd `k` with half DataT.
    bool unsupported_half = (sizeof(DataT) == 2) && ((k % 2) != 0);

    if (!unsupported_half && cutlass_range.contains(runtime_arch)) {
      // If device is SM_80 or later, use CUTLASS-based kernel.
      pairwise_matrix_sm80_dispatch(distance_op, params, cutlass_range, stream);
      return;
    }

    pairwise_matrix_jit_dispatch(distance_op, params, stream);
  }
}

};  // namespace cuvs::distance::detail
