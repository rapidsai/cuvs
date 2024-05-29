/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuvs/selection/select_k.hpp>
#include <raft/matrix/detail/select_k.cuh>

namespace cuvs::selection::detail {

template <typename T, typename IdxT>
void select_k(raft::resources const& handle,
              raft::device_matrix_view<const T, int64_t, raft::row_major> in_val,
              std::optional<raft::device_matrix_view<const IdxT, int64_t, raft::row_major>> in_idx,
              raft::device_matrix_view<T, int64_t, raft::row_major> out_val,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> out_idx,
              bool select_min,
              bool sorted,
              SelectAlgo algo,
              std::optional<raft::device_vector_view<const IdxT, int64_t>> len_i)
{
  RAFT_EXPECTS(out_val.extent(1) <= int64_t(std::numeric_limits<int>::max()),
               "output k must fit the int type.");
  auto batch_size = in_val.extent(0);
  auto len        = in_val.extent(1);
  auto k          = int(out_val.extent(1));
  RAFT_EXPECTS(batch_size == out_val.extent(0), "batch sizes must be equal");
  RAFT_EXPECTS(batch_size == out_idx.extent(0), "batch sizes must be equal");
  if (in_idx.has_value()) {
    RAFT_EXPECTS(batch_size == in_idx->extent(0), "batch sizes must be equal");
    RAFT_EXPECTS(len == in_idx->extent(1), "value and index input lengths must be equal");
  }
  RAFT_EXPECTS(int64_t(k) == out_idx.extent(1), "value and index output lengths must be equal");

  // just delegate implementation to raft - the primary benefit here is to have
  // instantiations only compiled once in cuvs
  return raft::matrix::detail::select_k<T, IdxT>(
    handle,
    in_val.data_handle(),
    in_idx.has_value() ? in_idx->data_handle() : nullptr,
    batch_size,
    len,
    k,
    out_val.data_handle(),
    out_idx.data_handle(),
    select_min,
    sorted,
    algo,
    len_i.has_value() ? len_i->data_handle() : nullptr);
}
}  // namespace cuvs::selection::detail

#define instantiate_cuvs_selection_select_k(T, IdxT)                                      \
  void cuvs::selection::select_k(                                                         \
    raft::resources const& handle,                                                        \
    raft::device_matrix_view<const T, int64_t, raft::row_major> in_val,                   \
    std::optional<raft::device_matrix_view<const IdxT, int64_t, raft::row_major>> in_idx, \
    raft::device_matrix_view<T, int64_t, raft::row_major> out_val,                        \
    raft::device_matrix_view<IdxT, int64_t, raft::row_major> out_idx,                     \
    bool select_min,                                                                      \
    bool sorted,                                                                          \
    SelectAlgo algo,                                                                      \
    std::optional<raft::device_vector_view<const IdxT, int64_t>> len_i)                   \
  {                                                                                       \
    detail::select_k<T, IdxT>(                                                            \
      handle, in_val, in_idx, out_val, out_idx, select_min, sorted, algo, len_i);         \
  }
