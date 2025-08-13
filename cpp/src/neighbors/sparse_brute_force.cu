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

#include <cuvs/neighbors/brute_force.hpp>

#include "detail/sparse_knn.cuh"

namespace cuvs::neighbors::brute_force {
template <typename T, typename IdxT>
sparse_index<T, IdxT>::sparse_index(raft::resources const& res,
                                    raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> dataset,
                                    cuvs::distance::DistanceType metric,
                                    T metric_arg)
  : dataset_(dataset), metric_(metric), metric_arg_(metric_arg)
{
}

auto build(raft::resources const& handle,
           raft::device_csr_matrix_view<const float, int, int, int> dataset,
           cuvs::distance::DistanceType metric,
           float metric_arg) -> cuvs::neighbors::brute_force::sparse_index<float, int>
{
  return sparse_index<float, int>(handle, dataset, metric, metric_arg);
}

void search(raft::resources const& handle,
            const sparse_search_params& params,
            const sparse_index<float, int>& index,
            raft::device_csr_matrix_view<const float, int, int, int> query,
            raft::device_matrix_view<int, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  auto idx_structure   = index.dataset().structure_view();
  auto query_structure = query.structure_view();
  int k                = neighbors.extent(1);

  detail::sparse_knn_t<int, float>(idx_structure.get_indptr().data(),
                                   idx_structure.get_indices().data(),
                                   index.dataset().get_elements().data(),
                                   idx_structure.get_nnz(),
                                   idx_structure.get_n_rows(),
                                   idx_structure.get_n_cols(),
                                   query_structure.get_indptr().data(),
                                   query_structure.get_indices().data(),
                                   query.get_elements().data(),
                                   query_structure.get_nnz(),
                                   query_structure.get_n_rows(),
                                   query_structure.get_n_cols(),
                                   neighbors.data_handle(),
                                   distances.data_handle(),
                                   k,
                                   handle,
                                   params.batch_size_index,
                                   params.batch_size_query,
                                   index.metric(),
                                   index.metric_arg())
    .run();
}
}  // namespace cuvs::neighbors::brute_force
