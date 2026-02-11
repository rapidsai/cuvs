/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "dispatch_layout.cuh"           // dispatch_layout
#include "kernel_sm60.cuh"               // pairwise_matrix_sm60_wrapper
#include <raft/linalg/contractions.cuh>  // raft::linalg::Policy4x4

#include <algorithm>  // std::min

namespace cuvs::distance::detail {

template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SmCompatT>
auto pairwise_matrix_sm60_get_wrapper(OpT distance_op,
                                      pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                      SmCompatT sm_compat_range)
  -> pairwise_matrix_sm60_wrapper<OpT, IdxT, DataT, OutT, FinOpT>
{
  int vec_len = determine_vec_len(params);

  // f takes compile-time constants row_major and vec_len aligned and returns
  // the corresponding kernel wrapper. The wrapper contains the launch
  // parameters of the kernel: a pointer to the kernel function, grid size,
  // block size, and shared memory size.
  auto f = [&](auto row_major, auto vec_len_aligned) -> auto {
    // row_major and vec_len are std::integral_constants of type bool and int
    // respectively.

    // To keep compile times in check, we only specialize on veclen > 1 when
    // the inner loop is relatively cheap (< 5 flops).
    constexpr int kVecLenOp = distance_op.kExpensiveInnerLoop ? 1 : vec_len_aligned();

    // Prevent double, vec_len=4 combination (this is not supported)
    constexpr int kVecLen = std::min(kVecLenOp, static_cast<int>(16 / sizeof(DataT)));

    using RowPolicy = typename raft::linalg::Policy4x4<DataT, kVecLen>::Policy;
    using ColPolicy = typename raft::linalg::Policy4x4<DataT, kVecLen>::ColPolicy;
    using policy    = typename std::conditional<row_major(), RowPolicy, ColPolicy>::type;

    auto wrapper =
      make_pairwise_matrix_sm60_wrapper<policy, row_major()>(distance_op, params, sm_compat_range);

    return wrapper;
  };

  // Dispatch_layout calls f with appropriate compile time constants based on
  // the runtime values of params.is_row_major and vec_len.
  return dispatch_layout(params.is_row_major, vec_len, f);
}

template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SmCompatT>
void pairwise_matrix_sm60_dispatch(OpT distance_op,
                                   pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                   SmCompatT sm_compat_range,
                                   cudaStream_t stream)
{
  auto wrapper = pairwise_matrix_sm60_get_wrapper(distance_op, params, sm_compat_range);

  wrapper.launch(distance_op, params, stream);
}

}  // namespace cuvs::distance::detail
