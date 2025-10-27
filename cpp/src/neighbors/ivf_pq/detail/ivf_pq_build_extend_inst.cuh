/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * NOTE: this file is used by generate_ivf_pq.py
 *
 */

#include <cuvs/neighbors/ivf_pq.hpp>

#include "../ivf_pq_build.cuh"

namespace cuvs::neighbors::ivf_pq {

#define CUVS_INST_IVF_PQ_BUILD_EXTEND(T, IdxT)                                                    \
  auto build(raft::resources const& handle,                                                       \
             const cuvs::neighbors::ivf_pq::index_params& params,                                 \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)                 \
    -> cuvs::neighbors::ivf_pq::index<IdxT>                                                       \
  {                                                                                               \
    return cuvs::neighbors::ivf_pq::detail::build(handle, params, dataset);                       \
  }                                                                                               \
                                                                                                  \
  void build(raft::resources const& handle,                                                       \
             const cuvs::neighbors::ivf_pq::index_params& params,                                 \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,                 \
             cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                           \
  {                                                                                               \
    cuvs::neighbors::ivf_pq::detail::build(handle, params, dataset, idx);                         \
  }                                                                                               \
                                                                                                  \
  auto build(raft::resources const& handle,                                                       \
             const cuvs::neighbors::ivf_pq::index_params& params,                                 \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)                   \
    -> cuvs::neighbors::ivf_pq::index<IdxT>                                                       \
  {                                                                                               \
    return cuvs::neighbors::ivf_pq::detail::build(handle, params, dataset);                       \
  }                                                                                               \
                                                                                                  \
  void build(raft::resources const& handle,                                                       \
             const cuvs::neighbors::ivf_pq::index_params& params,                                 \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,                   \
             cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                           \
  {                                                                                               \
    cuvs::neighbors::ivf_pq::detail::build(handle, params, dataset, idx);                         \
  }                                                                                               \
  auto extend(                                                                                    \
    raft::resources const& handle,                                                                \
    raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,                      \
    std::optional<raft::device_vector_view<const IdxT, int64_t, raft::row_major>> new_indices,    \
    const cuvs::neighbors::ivf_pq::index<IdxT>& orig_index)                                       \
    -> cuvs::neighbors::ivf_pq::index<IdxT>                                                       \
  {                                                                                               \
    return cuvs::neighbors::ivf_pq::detail::extend(handle, new_vectors, new_indices, orig_index); \
  }                                                                                               \
  void extend(raft::resources const& handle,                                                      \
              raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,            \
              std::optional<raft::device_vector_view<const IdxT, int64_t>> new_indices,           \
              cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                          \
  {                                                                                               \
    cuvs::neighbors::ivf_pq::detail::extend(handle, new_vectors, new_indices, idx);               \
  }                                                                                               \
  auto extend(raft::resources const& handle,                                                      \
              raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,              \
              std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices,             \
              const cuvs::neighbors::ivf_pq::index<IdxT>& orig_index)                             \
    -> cuvs::neighbors::ivf_pq::index<IdxT>                                                       \
  {                                                                                               \
    return cuvs::neighbors::ivf_pq::detail::extend(handle, new_vectors, new_indices, orig_index); \
  }                                                                                               \
                                                                                                  \
  void extend(raft::resources const& handle,                                                      \
              raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,              \
              std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices,             \
              cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                          \
  {                                                                                               \
    cuvs::neighbors::ivf_pq::detail::extend(handle, new_vectors, new_indices, idx);               \
  }

}  // namespace cuvs::neighbors::ivf_pq
