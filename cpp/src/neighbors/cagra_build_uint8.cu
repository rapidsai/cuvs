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
#include <omp.h>
#include <raft/linalg/map.cuh>

namespace cuvs::neighbors::cagra {

auto build(raft::resources const& handle,
           const cuvs::neighbors::cagra::index_params& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<uint8_t, uint32_t>
{
  uint8_t* dataset_ptr_uint8 = const_cast<uint8_t*>(dataset.data_handle());
  auto dataset_int8          = raft::make_device_matrix_view<int8_t, int64_t, raft::row_major>(
    reinterpret_cast<int8_t*>(dataset_ptr_uint8), dataset.extent(0), dataset.extent(1));
  cuvs::spatial::knn::detail::utils::offset_mapping map;
  if (params.metric != cuvs::distance::DistanceType::BitwiseHamming) {
    raft::linalg::map(handle, dataset_int8, map, dataset);
  }
  auto index_int8 =
    cuvs::neighbors::cagra::build(handle, params, raft::make_const_mdspan(dataset_int8));

  auto index_uint8 = cuvs::neighbors::cagra::index<uint8_t, uint32_t>(handle, params.metric);
  index_uint8.update_graph(handle, raft::make_const_mdspan(index_int8.graph()), true);
  index_uint8.update_dataset(handle, dataset);

  if (params.metric != cuvs::distance::DistanceType::BitwiseHamming) {
    auto dataset_uint8 = raft::make_device_matrix_view<uint8_t, int64_t, raft::row_major>(
      dataset_ptr_uint8, dataset.extent(0), dataset.extent(1));
    raft::linalg::map(handle, dataset_uint8, map, raft::make_const_mdspan(dataset_int8));
  }
  return index_uint8;
}

auto build(raft::resources const& handle,
           const cuvs::neighbors::cagra::index_params& params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<uint8_t, uint32_t>
{
  uint8_t* dataset_ptr_uint8 = const_cast<uint8_t*>(dataset.data_handle());
  auto dataset_int8          = raft::make_host_matrix_view<int8_t, int64_t, raft::row_major>(
    reinterpret_cast<int8_t*>(dataset_ptr_uint8), dataset.extent(0), dataset.extent(1));
  auto num_threads = omp_get_max_threads();
  cuvs::spatial::knn::detail::utils::mapping<int8_t> map_i8;
  cuvs::spatial::knn::detail::utils::mapping<uint8_t> map_u8;
  if (params.metric != cuvs::distance::DistanceType::BitwiseHamming) {
#pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < dataset_int8.extent(0); i++) {
      for (int64_t j = 0; j < dataset_int8.extent(1); j++) {
        dataset_int8(i, j) = map_i8(dataset(i, j));
      }
    }
  }
  auto index_int8 =
    cuvs::neighbors::cagra::build(handle, params, raft::make_const_mdspan(dataset_int8));

  auto index_uint8 = cuvs::neighbors::cagra::index<uint8_t, uint32_t>(handle, params.metric);
  index_uint8.update_graph(handle, raft::make_const_mdspan(index_int8.graph()), true);
  index_uint8.update_dataset(handle, dataset);

  if (params.metric != cuvs::distance::DistanceType::BitwiseHamming) {
    auto dataset_uint8 = raft::make_host_matrix_view<uint8_t, int64_t, raft::row_major>(
      dataset_ptr_uint8, dataset.extent(0), dataset.extent(1));
#pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < dataset_uint8.extent(0); i++) {
      for (int64_t j = 0; j < dataset_uint8.extent(1); j++) {
        dataset_uint8(i, j) = map_u8(dataset_int8(i, j));
      }
    }
  }
  return index_uint8;
}

}  // namespace cuvs::neighbors::cagra
