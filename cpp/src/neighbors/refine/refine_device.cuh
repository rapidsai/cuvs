/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../../core/nvtx.hpp"
#include "../detail/ann_utils.cuh"
#include "../ivf_flat/ivf_flat_build.cuh"
#include "../ivf_flat/ivf_flat_interleaved_scan.cuh"
#include "refine_common.hpp"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>

#include <thrust/sequence.h>

namespace cuvs::neighbors {

namespace detail {
/**
 * See cuvs::neighbors::refine for docs.
 */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_device(
  raft::resources const& handle,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> queries,
  raft::device_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates,
  raft::device_matrix_view<idx_t, matrix_idx, raft::row_major> indices,
  raft::device_matrix_view<distance_t, matrix_idx, raft::row_major> distances,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  matrix_idx n_candidates = neighbor_candidates.extent(1);
  matrix_idx n_queries    = queries.extent(0);
  matrix_idx dim          = dataset.extent(1);
  uint32_t k              = static_cast<uint32_t>(indices.extent(1));

  // TODO: this restriction could be lifted with some effort
  RAFT_EXPECTS(k <= raft::matrix::detail::select::warpsort::kMaxCapacity,
               "k must be less than topk::kMaxCapacity (%d).",
               raft::matrix::detail::select::warpsort::kMaxCapacity);

  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "neighbors::refine(%zu, %u)", size_t(n_queries), uint32_t(n_candidates));

  refine_check_input(dataset.extents(),
                     queries.extents(),
                     neighbor_candidates.extents(),
                     indices.extents(),
                     distances.extents(),
                     metric);

  // The refinement search can be mapped to an IVF flat search:
  // - We consider that the candidate vectors form a cluster, separately for each query.
  // - In other words, the n_queries * n_candidates vectors form n_queries clusters, each with
  //   n_candidates elements.
  // - We consider that the coarse level search is already performed and assigned a single cluster
  //   to search for each query (the cluster formed from the corresponding candidates).
  // - We run IVF flat search with n_probes=1 to select the best k elements of the candidates.
  rmm::device_uvector<uint32_t> fake_coarse_idx(n_queries, raft::resource::get_cuda_stream(handle));

  thrust::sequence(raft::resource::get_thrust_policy(handle),
                   fake_coarse_idx.data(),
                   fake_coarse_idx.data() + n_queries);

  cuvs::neighbors::ivf_flat::index<data_t, idx_t> refinement_index(
    handle, cuvs::distance::DistanceType(metric), n_queries, false, true, dim);

  cuvs::neighbors::ivf_flat::detail::fill_refinement_index(handle,
                                                           &refinement_index,
                                                           dataset.data_handle(),
                                                           neighbor_candidates.data_handle(),
                                                           n_queries,
                                                           n_candidates);
  uint32_t grid_dim_x = 1;

  // the neighbor ids will be computed in uint32_t as offset
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, raft::resource::get_cuda_stream(handle));
  // Offsets per probe for each query [n_queries] as n_probes = 1
  rmm::device_uvector<uint32_t> chunk_index(n_queries, raft::resource::get_cuda_stream(handle));

  // we know that each cluster has exactly n_candidates entries
  thrust::fill(raft::resource::get_thrust_policy(handle),
               chunk_index.data(),
               chunk_index.data() + n_queries,
               uint32_t(n_candidates));

  uint32_t* neighbors_uint32 = nullptr;
  if constexpr (sizeof(idx_t) == sizeof(uint32_t)) {
    neighbors_uint32 = reinterpret_cast<uint32_t*>(indices.data_handle());
  } else {
    neighbors_uint32_buf.resize(std::size_t(n_queries) * std::size_t(k),
                                raft::resource::get_cuda_stream(handle));
    neighbors_uint32 = neighbors_uint32_buf.data();
  }

  cuvs::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<
    data_t,
    typename cuvs::spatial::knn::detail::utils::config<data_t>::value_t,
    idx_t>(refinement_index,
           queries.data_handle(),
           fake_coarse_idx.data(),
           static_cast<uint32_t>(n_queries),
           0,
           cuvs::distance::DistanceType(refinement_index.metric()),
           1,
           k,
           0,
           chunk_index.data(),
           cuvs::distance::is_min_close(cuvs::distance::DistanceType(metric)),
           cuvs::neighbors::filtering::none_ivf_sample_filter(),
           neighbors_uint32,
           distances.data_handle(),
           grid_dim_x,
           raft::resource::get_cuda_stream(handle));

  // postprocessing -- neighbors from position to actual id
  cuvs::neighbors::ivf::detail::postprocess_neighbors(indices.data_handle(),
                                                      neighbors_uint32,
                                                      refinement_index.inds_ptrs().data_handle(),
                                                      fake_coarse_idx.data(),
                                                      chunk_index.data(),
                                                      n_queries,
                                                      1,
                                                      k,
                                                      raft::resource::get_cuda_stream(handle));
}

}  // namespace detail

template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_impl(
  raft::resources const& handle,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> queries,
  raft::device_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates,
  raft::device_matrix_view<idx_t, matrix_idx, raft::row_major> indices,
  raft::device_matrix_view<distance_t, matrix_idx, raft::row_major> distances,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  detail::refine_device(handle, dataset, queries, neighbor_candidates, indices, distances, metric);
}
}  // namespace cuvs::neighbors