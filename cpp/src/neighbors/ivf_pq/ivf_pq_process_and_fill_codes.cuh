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

#include <cstdint>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {
/**
 * A helper function: given the dataset in the rotated space
 *  [n_rows, rot_dim] = [n_rows, pq_dim * pq_len],
 * reinterpret the last dimension as two: [n_rows, pq_dim, pq_len]
 *
 * @tparam T
 * @tparam IdxT
 *
 * @param vectors input data [n_rows, rot_dim]
 * @param pq_centers codebook (used to infer the structure - pq_len)
 * @return reinterpreted vectors [n_rows, pq_dim, pq_len]
 */
template <typename T, typename IdxT>
static __device__ auto reinterpret_vectors(
  raft::device_matrix_view<T, IdxT, raft::row_major> vectors,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers)
  -> raft::device_mdspan<T, raft::extent_3d<IdxT>, raft::row_major>
{
  const uint32_t pq_len = pq_centers.extent(1);
  const uint32_t pq_dim = vectors.extent(1) / pq_len;
  using layout_t        = typename decltype(vectors)::layout_type;
  using accessor_t      = typename decltype(vectors)::accessor_type;
  return raft::mdspan<T, raft::extent_3d<IdxT>, layout_t, accessor_t>(
    vectors.data_handle(), raft::extent_3d<IdxT>{vectors.extent(0), pq_dim, pq_len});
}

template <typename IdxT>
void launch_process_and_fill_codes_kernel(raft::resources const& handle,
                                          index<IdxT>& index,
                                          raft::device_matrix_view<float> new_vectors_residual,
                                          std::variant<IdxT, const IdxT*> src_offset_or_indices,
                                          const uint32_t* new_labels,
                                          IdxT n_rows);

template <typename IdxT>
void launch_encode_list_data_kernel(raft::resources const& handle,
                                    index<IdxT>* index,
                                    raft::device_matrix_view<float> new_vectors_residual,
                                    uint32_t label,
                                    std::variant<uint32_t, const uint32_t*> offset_or_indices,
                                    IdxT n_rows);

}  // namespace cuvs::neighbors::ivf_pq::detail
