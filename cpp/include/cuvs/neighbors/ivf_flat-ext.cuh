/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cstdint>  // int64_t

#include <cuvs/neighbors/ivf_flat_serialize.cuh>
#include <cuvs/neighbors/ivf_flat_types.hpp>      // cuvs::neighbors::ivf_flat::index
#include <raft/core/device_mdspan.hpp>            // raft::device_matrix_view
#include <raft/core/resources.hpp>                // raft::resources
#include <raft/util/raft_explicit.hpp>            // RAFT_EXPLICIT
#include <rmm/mr/device/per_device_resource.hpp>  // rmm::mr::device_memory_resource

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace cuvs::neighbors::ivf_flat {

template <typename T, typename IdxT>
auto build(raft::resources const& handle,
           const index_params& params,
           const T* dataset,
           IdxT n_rows,
           uint32_t dim) -> index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
auto build(raft::resources const& handle,
           const index_params& params,
           raft::device_matrix_view<const T, IdxT, raft::row_major> dataset)
  -> index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
void build(raft::resources const& handle,
           const index_params& params,
           raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<T, IdxT>& idx) RAFT_EXPLICIT;

template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            const index<T, IdxT>& orig_index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,
            const index<T, IdxT>& orig_index) -> index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            index<T, IdxT>* index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,
            index<T, IdxT>* index) RAFT_EXPLICIT;

template <typename T, typename IdxT, typename IvfSampleFilterT>
void search_with_filtering(raft::resources const& handle,
                           const search_params& params,
                           const index<T, IdxT>& index,
                           const T* queries,
                           uint32_t n_queries,
                           uint32_t k,
                           IdxT* neighbors,
                           float* distances,
                           rmm::mr::device_memory_resource* mr = nullptr,
                           IvfSampleFilterT sample_filter      = IvfSampleFilterT()) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const search_params& params,
            const index<T, IdxT>& index,
            const T* queries,
            uint32_t n_queries,
            uint32_t k,
            IdxT* neighbors,
            float* distances,
            rmm::mr::device_memory_resource* mr = nullptr) RAFT_EXPLICIT;

template <typename T, typename IdxT, typename IvfSampleFilterT>
void search_with_filtering(raft::resources const& handle,
                           const search_params& params,
                           const index<T, IdxT>& index,
                           raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
                           raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
                           raft::device_matrix_view<float, IdxT, raft::row_major> distances,
                           IvfSampleFilterT sample_filter = IvfSampleFilterT()) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const search_params& params,
            const index<T, IdxT>& index,
            raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
            raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
            raft::device_matrix_view<float, IdxT, raft::row_major> distances) RAFT_EXPLICIT;

}  // namespace cuvs::neighbors::ivf_flat

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_neighbors_ivf_flat_build(T, IdxT)            \
  extern template auto cuvs::neighbors::ivf_flat::build<T, IdxT>(     \
    raft::resources const& handle,                                    \
    const cuvs::neighbors::ivf_flat::index_params& params,            \
    const T* dataset,                                                 \
    IdxT n_rows,                                                      \
    uint32_t dim)                                                     \
    ->cuvs::neighbors::ivf_flat::index<T, IdxT>;                      \
                                                                      \
  extern template auto cuvs::neighbors::ivf_flat::build<T, IdxT>(     \
    raft::resources const& handle,                                    \
    const cuvs::neighbors::ivf_flat::index_params& params,            \
    raft::device_matrix_view<const T, IdxT, raft::row_major> dataset) \
    ->cuvs::neighbors::ivf_flat::index<T, IdxT>;                      \
                                                                      \
  extern template void cuvs::neighbors::ivf_flat::build<T, IdxT>(     \
    raft::resources const& handle,                                    \
    const cuvs::neighbors::ivf_flat::index_params& params,            \
    raft::device_matrix_view<const T, IdxT, raft::row_major> dataset, \
    cuvs::neighbors::ivf_flat::index<T, IdxT>& idx);

instantiate_raft_neighbors_ivf_flat_build(float, int64_t);
instantiate_raft_neighbors_ivf_flat_build(int8_t, int64_t);
instantiate_raft_neighbors_ivf_flat_build(uint8_t, int64_t);
#undef instantiate_raft_neighbors_ivf_flat_build

#define instantiate_raft_neighbors_ivf_flat_extend(T, IdxT)                \
  extern template auto cuvs::neighbors::ivf_flat::extend<T, IdxT>(         \
    raft::resources const& handle,                                         \
    const cuvs::neighbors::ivf_flat::index<T, IdxT>& orig_index,           \
    const T* new_vectors,                                                  \
    const IdxT* new_indices,                                               \
    IdxT n_rows)                                                           \
    ->cuvs::neighbors::ivf_flat::index<T, IdxT>;                           \
                                                                           \
  extern template auto cuvs::neighbors::ivf_flat::extend<T, IdxT>(         \
    raft::resources const& handle,                                         \
    raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,  \
    std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
    const cuvs::neighbors::ivf_flat::index<T, IdxT>& orig_index)           \
    ->cuvs::neighbors::ivf_flat::index<T, IdxT>;                           \
                                                                           \
  extern template void cuvs::neighbors::ivf_flat::extend<T, IdxT>(         \
    raft::resources const& handle,                                         \
    cuvs::neighbors::ivf_flat::index<T, IdxT>* index,                      \
    const T* new_vectors,                                                  \
    const IdxT* new_indices,                                               \
    IdxT n_rows);                                                          \
                                                                           \
  extern template void cuvs::neighbors::ivf_flat::extend<T, IdxT>(         \
    raft::resources const& handle,                                         \
    raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,  \
    std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
    cuvs::neighbors::ivf_flat::index<T, IdxT>* index);

instantiate_raft_neighbors_ivf_flat_extend(float, int64_t);
instantiate_raft_neighbors_ivf_flat_extend(int8_t, int64_t);
instantiate_raft_neighbors_ivf_flat_extend(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_flat_extend

#define instantiate_raft_neighbors_ivf_flat_search(T, IdxT)           \
  extern template void cuvs::neighbors::ivf_flat::search<T, IdxT>(    \
    raft::resources const& handle,                                    \
    const cuvs::neighbors::ivf_flat::search_params& params,           \
    const cuvs::neighbors::ivf_flat::index<T, IdxT>& index,           \
    const T* queries,                                                 \
    uint32_t n_queries,                                               \
    uint32_t k,                                                       \
    IdxT* neighbors,                                                  \
    float* distances,                                                 \
    rmm::mr::device_memory_resource* mr);                             \
                                                                      \
  extern template void cuvs::neighbors::ivf_flat::search<T, IdxT>(    \
    raft::resources const& handle,                                    \
    const cuvs::neighbors::ivf_flat::search_params& params,           \
    const cuvs::neighbors::ivf_flat::index<T, IdxT>& index,           \
    raft::device_matrix_view<const T, IdxT, raft::row_major> queries, \
    raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,  \
    raft::device_matrix_view<float, IdxT, raft::row_major> distances);

instantiate_raft_neighbors_ivf_flat_search(float, int64_t);
instantiate_raft_neighbors_ivf_flat_search(int8_t, int64_t);
instantiate_raft_neighbors_ivf_flat_search(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_flat_search
