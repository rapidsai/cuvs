/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "detail/ann_utils.cuh"
#include <cuvs/neighbors/cagra.hpp>
#include <raft/linalg/map.cuh>

namespace cuvs::neighbors::cagra {
void search(raft::resources const& handle,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  uint8_t* queries_ptr_uint8 = const_cast<uint8_t*>(queries.data_handle());
  auto queries_int8          = raft::make_device_matrix_view<int8_t, int64_t, raft::row_major>(
    reinterpret_cast<int8_t*>(queries_ptr_uint8), queries.extent(0), queries.extent(1));
  cuvs::spatial::knn::detail::utils::mapping<int8_t> map_i8;
  cuvs::spatial::knn::detail::utils::mapping<uint8_t> map_u8;
  if (index.metric() != cuvs::distance::DistanceType::BitwiseHamming) {
    raft::linalg::map(handle, queries_int8, map_i8, queries);
  }
  auto index_int8 = cuvs::neighbors::cagra::index<int8_t, uint32_t>(handle, index.metric());
  index_int8.update_graph(handle, index.graph());
  const uint32_t src_stride =
    index.dataset().stride(0) > 0 ? index.dataset().stride(0) : index.dataset().extent(1);
  index_int8.update_dataset(handle,
                            raft::make_device_strided_matrix_view<const int8_t, int64_t>(
                              (const int8_t*)index.dataset().data_handle(),
                              index.dataset().extent(0),
                              index.dataset().extent(1),
                              src_stride));

  cuvs::neighbors::cagra::search(handle,
                                 params,
                                 index_int8,
                                 raft::make_const_mdspan(queries_int8),
                                 neighbors,
                                 distances,
                                 sample_filter);
  if (index.metric() != cuvs::distance::DistanceType::BitwiseHamming) {
    auto queries_uint8 = raft::make_device_matrix_view<uint8_t, int64_t, raft::row_major>(
      queries_ptr_uint8, queries.extent(0), queries.extent(1));
    raft::linalg::map(handle, queries_uint8, map_u8, raft::make_const_mdspan(queries_int8));
  }
}
void search(raft::resources const& handle,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  uint8_t* queries_ptr_uint8 = const_cast<uint8_t*>(queries.data_handle());
  auto queries_int8          = raft::make_device_matrix_view<int8_t, int64_t, raft::row_major>(
    reinterpret_cast<int8_t*>(queries_ptr_uint8), queries.extent(0), queries.extent(1));
  cuvs::spatial::knn::detail::utils::mapping<int8_t> map_i8;
  cuvs::spatial::knn::detail::utils::mapping<uint8_t> map_u8;
  if (index.metric() != cuvs::distance::DistanceType::BitwiseHamming) {
    raft::linalg::map(handle, queries_int8, map_i8, queries);
  }
  auto index_int8 = cuvs::neighbors::cagra::index<int8_t, uint32_t>(handle, index.metric());
  index_int8.update_graph(handle, index.graph());
  const uint32_t src_stride =
    index.dataset().stride(0) > 0 ? index.dataset().stride(0) : index.dataset().extent(1);
  index_int8.update_dataset(handle,
                            raft::make_device_strided_matrix_view<const int8_t, int64_t>(
                              (const int8_t*)index.dataset().data_handle(),
                              index.dataset().extent(0),
                              index.dataset().extent(1),
                              src_stride));

  cuvs::neighbors::cagra::search(handle,
                                 params,
                                 index_int8,
                                 raft::make_const_mdspan(queries_int8),
                                 neighbors,
                                 distances,
                                 sample_filter);
  if (index.metric() != cuvs::distance::DistanceType::BitwiseHamming) {
    auto queries_uint8 = raft::make_device_matrix_view<uint8_t, int64_t, raft::row_major>(
      queries_ptr_uint8, queries.extent(0), queries.extent(1));
    raft::linalg::map(handle, queries_uint8, map_u8, raft::make_const_mdspan(queries_int8));
  }
}

}  // namespace cuvs::neighbors::cagra
