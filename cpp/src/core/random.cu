/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "random.hpp"

namespace cuvs::detail {

template <typename T, typename Len>
void uniform_impl(RngState& state, T* out, Len count, T a, T b, cudaStream_t stream)
{
  raft::random::detail::uniform(state, out, count, a, b, stream);
}

template <typename T, typename Len>
void uniformInt_impl(RngState& state, T* out, Len count, T a, T b, cudaStream_t stream)
{
  raft::random::detail::uniformInt(state, out, count, a, b, stream);
}

template <typename T, typename Len>
void normal_impl(RngState& state, T* out, Len count, T a, T b, cudaStream_t stream)
{
  raft::random::detail::normal(state, out, count, a, b, stream);
}

template <typename T, typename IdxT>
void sample_rows_impl(const raft::resources& res,
                      RngState state,
                      const T* input,
                      IdxT stride,
                      IdxT n_rows,
                      raft::device_matrix_view<T, IdxT> output)
{
  raft::matrix::detail::sample_rows(res, state, input, stride, n_rows, output);
}

raft::device_vector<int, int> sample_indices(const raft::resources& res,
                                             RngState& state,
                                             int n,
                                             int count)
{
  return raft::random::excess_subsample<int, int>(res, state, n, count);
}

raft::device_vector<int64_t, int64_t> sample_indices(const raft::resources& res,
                                                     RngState& state,
                                                     int64_t n,
                                                     int64_t count)
{
  return raft::random::excess_subsample<int64_t, int64_t>(res, state, n, count);
}

// EXPLICIT_HOST_INSTANTIATIONS

}  // namespace cuvs::detail
