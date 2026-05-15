/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

/* This file provides concise function templates that can be instantiated in
 * src/distance/detail/pairwise_matrix/. Previously,
 * cuvs::distance::detail::distance was instantiated. The function necessarily
 * required a large set of include files, which slowed down the build.
 */

#include "../pairwise_matrix/jit_lto_kernels/pairwise_matrix_jit.cuh"  // pairwise_matrix_jit_dispatch

namespace cuvs::distance::detail {

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

  pairwise_matrix_jit_dispatch(distance_op, params, stream);
}

};  // namespace cuvs::distance::detail
