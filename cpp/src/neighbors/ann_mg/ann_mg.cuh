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

#include "../detail/knn_merge_parts.cuh"
#include <raft/core/serialize.hpp>
#include <raft/linalg/add.cuh>
#include <raft/util/cuda_dev_essentials.cuh>

#include "../../utils/nccl_helpers.cuh"
#define NO_NCCL_FORWARD_DECLARATION
#include <cuvs/neighbors/ann_mg.hpp>
#undef NO_NCCL_FORWARD_DECLARATION
#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::mg {
using namespace cuvs::neighbors;
using namespace raft;

void check_omp_threads(const int requirements);

template <typename AnnIndexType, typename T, typename IdxT>
template <typename Accessor>
void ann_interface<AnnIndexType, T, IdxT>::build(
  raft::resources const& handle,
  const cuvs::neighbors::index_params* index_params,
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> index_dataset)
{
  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_flat::build(
      handle, *static_cast<const ivf_flat::index_params*>(index_params), index_dataset);
    index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_pq::build(
      handle, *static_cast<const ivf_pq::index_params*>(index_params), index_dataset);
    index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::index<T, IdxT> idx(handle);
    idx = cuvs::neighbors::cagra::build(
      handle, *static_cast<const cagra::index_params*>(index_params), index_dataset);
    index_.emplace(std::move(idx));
  }
  resource::sync_stream(handle);
}

template <typename AnnIndexType, typename T, typename IdxT>
template <typename Accessor1, typename Accessor2>
void ann_interface<AnnIndexType, T, IdxT>::extend(
  raft::resources const& handle,
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor1> new_vectors,
  std::optional<raft::mdspan<const IdxT, vector_extent<int64_t>, layout_c_contiguous, Accessor2>>
    new_indices)
{
  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_flat::extend(handle, new_vectors, new_indices, index_.value());
    index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_pq::extend(handle, new_vectors, new_indices, index_.value());
    index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    RAFT_FAIL("CAGRA does not implement the extend method");
  }
  resource::sync_stream(handle);
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_interface<AnnIndexType, T, IdxT>::search(
  raft::resources const& handle,
  const cuvs::neighbors::search_params* search_params,
  raft::host_matrix_view<const T, int64_t, row_major> h_queries,
  raft::device_matrix_view<IdxT, int64_t, row_major> d_neighbors,
  raft::device_matrix_view<float, int64_t, row_major> d_distances) const
{
  int64_t n_rows = h_queries.extent(0);
  int64_t n_dims = h_queries.extent(1);
  auto d_queries = raft::make_device_matrix<T, int64_t, row_major>(handle, n_rows, n_dims);
  raft::copy(d_queries.data_handle(),
             h_queries.data_handle(),
             n_rows * n_dims,
             resource::get_cuda_stream(handle));
  auto d_query_view = raft::make_const_mdspan(d_queries.view());

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, int64_t>>::value) {
    cuvs::neighbors::ivf_flat::search(
      handle,
      *reinterpret_cast<const ivf_flat::search_params*>(search_params),
      index_.value(),
      d_query_view,
      d_neighbors,
      d_distances);
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<int64_t>>::value) {
    cuvs::neighbors::ivf_pq::search(handle,
                                    *reinterpret_cast<const ivf_pq::search_params*>(search_params),
                                    index_.value(),
                                    d_query_view,
                                    d_neighbors,
                                    d_distances);
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, uint32_t>>::value) {
    cuvs::neighbors::cagra::search(handle,
                                   *reinterpret_cast<const cagra::search_params*>(search_params),
                                   index_.value(),
                                   d_query_view,
                                   d_neighbors,
                                   d_distances);
  }
  resource::sync_stream(handle);
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_interface<AnnIndexType, T, IdxT>::serialize(raft::resources const& handle,
                                                     std::ostream& os) const
{
  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::serialize(handle, os, index_.value());
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::serialize(handle, os, index_.value());
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::serialize(handle, os, index_.value(), true);
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_interface<AnnIndexType, T, IdxT>::deserialize(raft::resources const& handle,
                                                       std::istream& is)
{
  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::index<T, IdxT> idx(handle);
    ivf_flat::deserialize(handle, is, &idx);
    index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::index<IdxT> idx(handle);
    ivf_pq::deserialize(handle, is, &idx);
    index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::index<T, IdxT> idx(handle);
    cagra::deserialize(handle, is, &idx);
    index_.emplace(std::move(idx));
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_interface<AnnIndexType, T, IdxT>::deserialize(raft::resources const& handle,
                                                       const std::string& filename)
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::index<T, IdxT> idx(handle);
    ivf_flat::deserialize(handle, is, &idx);
    index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::index<IdxT> idx(handle);
    ivf_pq::deserialize(handle, is, &idx);
    index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::index<T, IdxT> idx(handle);
    cagra::deserialize(handle, is, &idx);
    index_.emplace(std::move(idx));
  }

  is.close();
}

template <typename AnnIndexType, typename T, typename IdxT>
const IdxT ann_interface<AnnIndexType, T, IdxT>::size() const
{
  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    return index_.value().size();
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    return index_.value().size();
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    return index_.value().size();
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
ann_mg_index<AnnIndexType, T, IdxT>::ann_mg_index(distribution_mode mode, int num_ranks_)
  : mode_(mode), num_ranks_(num_ranks_)
{
}

template <typename AnnIndexType, typename T, typename IdxT>
ann_mg_index<AnnIndexType, T, IdxT>::ann_mg_index(const raft::resources& handle,
                                                  const cuvs::neighbors::mg::nccl_clique& clique,
                                                  const std::string& filename)
{
  deserialize_mg_index(handle, clique, filename);
}

// local index deserialization and distribution
template <typename AnnIndexType, typename T, typename IdxT>
void ann_mg_index<AnnIndexType, T, IdxT>::deserialize_and_distribute(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const std::string& filename)
{
  for (int rank = 0; rank < num_ranks_; rank++) {
    int dev_id                            = clique.device_ids_[rank];
    const raft::device_resources& dev_res = clique.device_resources_[rank];
    RAFT_CUDA_TRY(cudaSetDevice(dev_id));
    auto& ann_if = ann_interfaces_.emplace_back();
    ann_if.deserialize(dev_res, filename);
  }
}

// MG index deserialization
template <typename AnnIndexType, typename T, typename IdxT>
void ann_mg_index<AnnIndexType, T, IdxT>::deserialize_mg_index(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const std::string& filename)
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  mode_      = (cuvs::neighbors::mg::distribution_mode)deserialize_scalar<int>(handle, is);
  num_ranks_ = deserialize_scalar<int>(handle, is);

  for (int rank = 0; rank < num_ranks_; rank++) {
    int dev_id                            = clique.device_ids_[rank];
    const raft::device_resources& dev_res = clique.device_resources_[rank];
    RAFT_CUDA_TRY(cudaSetDevice(dev_id));
    auto& ann_if = ann_interfaces_.emplace_back();
    ann_if.deserialize(dev_res, is);
  }

  is.close();
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_mg_index<AnnIndexType, T, IdxT>::build(
  const cuvs::neighbors::mg::nccl_clique& clique,
  const cuvs::neighbors::index_params* index_params,
  raft::host_matrix_view<const T, int64_t, row_major> index_dataset)
{
  if (mode_ == REPLICATED) {
    int64_t n_rows = index_dataset.extent(0);
    RAFT_LOG_INFO("REPLICATED BUILD: %d*%drows", num_ranks_, n_rows);

    ann_interfaces_.resize(num_ranks_);
#pragma omp parallel for
    for (int rank = 0; rank < num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));
      auto& ann_if = ann_interfaces_[rank];
      ann_if.build(dev_res, index_params, index_dataset);
      resource::sync_stream(dev_res);
    }
  } else if (mode_ == SHARDED) {
    int64_t n_rows           = index_dataset.extent(0);
    int64_t n_cols           = index_dataset.extent(1);
    int64_t n_rows_per_shard = raft::ceildiv(n_rows, (int64_t)num_ranks_);

    RAFT_LOG_INFO("SHARDED BUILD: %d*%drows", num_ranks_, n_rows_per_shard);

    ann_interfaces_.resize(num_ranks_);
#pragma omp parallel for
    for (int rank = 0; rank < num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));
      int64_t offset                  = rank * n_rows_per_shard;
      int64_t n_rows_of_current_shard = std::min(n_rows_per_shard, n_rows - offset);
      const T* partition_ptr          = index_dataset.data_handle() + (offset * n_cols);
      auto partition                  = raft::make_host_matrix_view<const T, int64_t, row_major>(
        partition_ptr, n_rows_of_current_shard, n_cols);
      auto& ann_if = ann_interfaces_[rank];
      ann_if.build(dev_res, index_params, partition);
      resource::sync_stream(dev_res);
    }
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_mg_index<AnnIndexType, T, IdxT>::extend(
  const cuvs::neighbors::mg::nccl_clique& clique,
  raft::host_matrix_view<const T, int64_t, row_major> new_vectors,
  std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices)
{
  int64_t n_rows = new_vectors.extent(0);
  if (mode_ == REPLICATED) {
    RAFT_LOG_INFO("REPLICATED EXTEND: %d*%drows", num_ranks_, n_rows);

#pragma omp parallel for
    for (int rank = 0; rank < num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));
      auto& ann_if = ann_interfaces_[rank];
      ann_if.extend(dev_res, new_vectors, new_indices);
      resource::sync_stream(dev_res);
    }
  } else if (mode_ == SHARDED) {
    int64_t n_cols           = new_vectors.extent(1);
    int64_t n_rows_per_shard = raft::ceildiv(n_rows, (int64_t)num_ranks_);

    RAFT_LOG_INFO("SHARDED EXTEND: %d*%drows", num_ranks_, n_rows_per_shard);

#pragma omp parallel for
    for (int rank = 0; rank < num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));
      int64_t offset                  = rank * n_rows_per_shard;
      int64_t n_rows_of_current_shard = std::min(n_rows_per_shard, n_rows - offset);
      const T* new_vectors_ptr        = new_vectors.data_handle() + (offset * n_cols);
      auto new_vectors_part           = raft::make_host_matrix_view<const T, int64_t, row_major>(
        new_vectors_ptr, n_rows_of_current_shard, n_cols);

      std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices_part = std::nullopt;
      if (new_indices.has_value()) {
        const IdxT* new_indices_ptr = new_indices.value().data_handle() + offset;
        new_indices_part            = raft::make_host_vector_view<const IdxT, int64_t>(
          new_indices_ptr, n_rows_of_current_shard);
      }
      auto& ann_if = ann_interfaces_[rank];
      ann_if.extend(dev_res, new_vectors_part, new_indices_part);
      resource::sync_stream(dev_res);
    }
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_mg_index<AnnIndexType, T, IdxT>::search(
  const cuvs::neighbors::mg::nccl_clique& clique,
  const cuvs::neighbors::search_params* search_params,
  raft::host_matrix_view<const T, int64_t, row_major> queries,
  raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances,
  cuvs::neighbors::mg::sharded_merge_mode merge_mode,
  int64_t n_rows_per_batch) const
{
  int64_t n_rows      = queries.extent(0);
  int64_t n_cols      = queries.extent(1);
  int64_t n_neighbors = neighbors.extent(1);

  int64_t n_rows_per_rank = raft::ceildiv(n_rows, (int64_t)num_ranks_);
  n_rows_per_batch =
    std::min(n_rows_per_batch, n_rows_per_rank);  // get at least num_ranks_ batches

  int64_t n_batches = raft::ceildiv(n_rows, (int64_t)n_rows_per_batch);
  if (n_batches == 1) n_rows_per_batch = n_rows;

  if (mode_ == REPLICATED) {
    RAFT_LOG_INFO("REPLICATED SEARCH: %d*%drows", n_batches, n_rows_per_batch);

#pragma omp parallel for
    for (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
      int rank                              = batch_idx % num_ranks_;  // alternate GPUs
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));

      int64_t offset                  = batch_idx * n_rows_per_batch;
      int64_t query_offset            = offset * n_cols;
      int64_t output_offset           = offset * n_neighbors;
      int64_t n_rows_of_current_batch = std::min(n_rows_per_batch, n_rows - offset);

      auto query_partition = raft::make_host_matrix_view<const T, int64_t, row_major>(
        queries.data_handle() + query_offset, n_rows_of_current_batch, n_cols);
      auto d_neighbors = raft::make_device_matrix<IdxT, int64_t, row_major>(
        dev_res, n_rows_of_current_batch, n_neighbors);
      auto d_distances = raft::make_device_matrix<float, int64_t, row_major>(
        dev_res, n_rows_of_current_batch, n_neighbors);

      auto& ann_if = ann_interfaces_[rank];
      ann_if.search(
        dev_res, search_params, query_partition, d_neighbors.view(), d_distances.view());

      raft::copy(neighbors.data_handle() + output_offset,
                 d_neighbors.data_handle(),
                 n_rows_of_current_batch * n_neighbors,
                 resource::get_cuda_stream(dev_res));
      raft::copy(distances.data_handle() + output_offset,
                 d_distances.data_handle(),
                 n_rows_of_current_batch * n_neighbors,
                 resource::get_cuda_stream(dev_res));

      resource::sync_stream(dev_res);
    }
  } else if (mode_ == SHARDED) {
    if (merge_mode == MERGE_ON_ROOT_RANK) {
      RAFT_LOG_INFO("SHARDED SEARCH WITH MERGE_ON_ROOT_RANK MERGE MODE: %d*%drows",
                    n_batches,
                    n_rows_per_batch);
      sharded_search_with_direct_merge(clique,
                                       search_params,
                                       queries,
                                       neighbors,
                                       distances,
                                       n_rows_per_batch,
                                       n_rows,
                                       n_cols,
                                       n_neighbors,
                                       n_batches);
    } else if (merge_mode == TREE_MERGE) {
      RAFT_LOG_INFO(
        "SHARDED SEARCH WITH TREE_MERGE MERGE MODE %d*%drows", n_batches, n_rows_per_batch);
      sharded_search_with_tree_merge(clique,
                                     search_params,
                                     queries,
                                     neighbors,
                                     distances,
                                     n_rows_per_batch,
                                     n_rows,
                                     n_cols,
                                     n_neighbors,
                                     n_batches);
    }
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_mg_index<AnnIndexType, T, IdxT>::sharded_search_with_direct_merge(
  const cuvs::neighbors::mg::nccl_clique& clique,
  const cuvs::neighbors::search_params* search_params,
  raft::host_matrix_view<const T, int64_t, row_major> queries,
  raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances,
  int64_t n_rows_per_batch,
  int64_t n_rows,
  int64_t n_cols,
  int64_t n_neighbors,
  int64_t n_batches) const
{
  const auto& root_handle = clique.set_current_device_to_root_rank();
  auto in_neighbors       = raft::make_device_matrix<IdxT, int64_t, row_major>(
    root_handle, num_ranks_ * n_rows_per_batch, n_neighbors);
  auto in_distances = raft::make_device_matrix<float, int64_t, row_major>(
    root_handle, num_ranks_ * n_rows_per_batch, n_neighbors);
  auto out_neighbors =
    raft::make_device_matrix<IdxT, int64_t, row_major>(root_handle, n_rows_per_batch, n_neighbors);
  auto out_distances =
    raft::make_device_matrix<float, int64_t, row_major>(root_handle, n_rows_per_batch, n_neighbors);

  for (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
    int64_t offset                  = batch_idx * n_rows_per_batch;
    int64_t query_offset            = offset * n_cols;
    int64_t output_offset           = offset * n_neighbors;
    int64_t n_rows_of_current_batch = std::min((int64_t)n_rows_per_batch, n_rows - offset);
    int64_t part_size               = n_rows_of_current_batch * n_neighbors;
    auto query_partition            = raft::make_host_matrix_view<const T, int64_t, row_major>(
      queries.data_handle() + query_offset, n_rows_of_current_batch, n_cols);

    const int& requirements = num_ranks_;
    check_omp_threads(requirements);  // should use at least num_ranks_ threads to avoid NCCL hang
#pragma omp parallel for num_threads(num_ranks_)
    for (int rank = 0; rank < num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      auto& ann_if                          = ann_interfaces_[rank];
      const auto& comms                     = resource::get_comms(dev_res);
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));

      if (rank == clique.root_rank_) {  // root rank
        uint64_t batch_offset = clique.root_rank_ * part_size;
        auto d_neighbors      = raft::make_device_matrix_view<IdxT, int64_t, row_major>(
          in_neighbors.data_handle() + batch_offset, n_rows_of_current_batch, n_neighbors);
        auto d_distances = raft::make_device_matrix_view<float, int64_t, row_major>(
          in_distances.data_handle() + batch_offset, n_rows_of_current_batch, n_neighbors);
        ann_if.search(dev_res, search_params, query_partition, d_neighbors, d_distances);

        // wait for other ranks
        comms.group_start();
        for (int from_rank = 0; from_rank < num_ranks_; from_rank++) {
          if (from_rank == clique.root_rank_) continue;

          batch_offset = from_rank * part_size;
          comms.device_recv(in_neighbors.data_handle() + batch_offset,
                            part_size,
                            from_rank,
                            resource::get_cuda_stream(dev_res));
          comms.device_recv(in_distances.data_handle() + batch_offset,
                            part_size,
                            from_rank,
                            resource::get_cuda_stream(dev_res));
        }
        comms.group_end();
        resource::sync_stream(dev_res);
      } else {  // non-root ranks
        auto d_neighbors = raft::make_device_matrix<IdxT, int64_t, row_major>(
          dev_res, n_rows_of_current_batch, n_neighbors);
        auto d_distances = raft::make_device_matrix<float, int64_t, row_major>(
          dev_res, n_rows_of_current_batch, n_neighbors);
        ann_if.search(
          dev_res, search_params, query_partition, d_neighbors.view(), d_distances.view());

        // send results to root rank
        comms.group_start();
        comms.device_send(d_neighbors.data_handle(),
                          part_size,
                          clique.root_rank_,
                          resource::get_cuda_stream(dev_res));
        comms.device_send(d_distances.data_handle(),
                          part_size,
                          clique.root_rank_,
                          resource::get_cuda_stream(dev_res));
        comms.group_end();
        resource::sync_stream(dev_res);
      }
    }

    const auto& root_handle_   = clique.set_current_device_to_root_rank();
    auto h_trans               = std::vector<IdxT>(num_ranks_);
    int64_t translation_offset = 0;
    for (int rank = 0; rank < num_ranks_; rank++) {
      h_trans[rank] = translation_offset;
      translation_offset += ann_interfaces_[rank].size();
    }
    auto d_trans = raft::make_device_vector<IdxT, IdxT>(root_handle_, num_ranks_);
    raft::copy(
      d_trans.data_handle(), h_trans.data(), num_ranks_, resource::get_cuda_stream(root_handle_));

    cuvs::neighbors::detail::knn_merge_parts(in_distances.data_handle(),
                                             in_neighbors.data_handle(),
                                             out_distances.data_handle(),
                                             out_neighbors.data_handle(),
                                             n_rows_of_current_batch,
                                             num_ranks_,
                                             n_neighbors,
                                             resource::get_cuda_stream(root_handle_),
                                             d_trans.data_handle());

    raft::copy(neighbors.data_handle() + output_offset,
               out_neighbors.data_handle(),
               part_size,
               resource::get_cuda_stream(root_handle_));
    raft::copy(distances.data_handle() + output_offset,
               out_distances.data_handle(),
               part_size,
               resource::get_cuda_stream(root_handle_));

    resource::sync_stream(root_handle_);
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_mg_index<AnnIndexType, T, IdxT>::sharded_search_with_tree_merge(
  const cuvs::neighbors::mg::nccl_clique& clique,
  const cuvs::neighbors::search_params* search_params,
  raft::host_matrix_view<const T, int64_t, row_major> queries,
  raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances,
  int64_t n_rows_per_batch,
  int64_t n_rows,
  int64_t n_cols,
  int64_t n_neighbors,
  int64_t n_batches) const
{
  for (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
    int64_t offset                  = batch_idx * n_rows_per_batch;
    int64_t query_offset            = offset * n_cols;
    int64_t output_offset           = offset * n_neighbors;
    int64_t n_rows_of_current_batch = std::min((int64_t)n_rows_per_batch, n_rows - offset);
    auto query_partition            = raft::make_host_matrix_view<const T, int64_t, row_major>(
      queries.data_handle() + query_offset, n_rows_of_current_batch, n_cols);

    const int& requirements = num_ranks_;
    check_omp_threads(requirements);  // should use at least num_ranks_ threads to avoid NCCL hang
#pragma omp parallel for num_threads(num_ranks_)
    for (int rank = 0; rank < num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      auto& ann_if                          = ann_interfaces_[rank];
      const auto& comms                     = resource::get_comms(dev_res);
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));

      int64_t part_size = n_rows_of_current_batch * n_neighbors;

      auto tmp_neighbors = raft::make_device_matrix<IdxT, int64_t, row_major>(
        dev_res, 2 * n_rows_of_current_batch, n_neighbors);
      auto tmp_distances = raft::make_device_matrix<float, int64_t, row_major>(
        dev_res, 2 * n_rows_of_current_batch, n_neighbors);
      auto neighbors_view = raft::make_device_matrix_view<IdxT, int64_t, row_major>(
        tmp_neighbors.data_handle(), n_rows_of_current_batch, n_neighbors);
      auto distances_view = raft::make_device_matrix_view<float, int64_t, row_major>(
        tmp_distances.data_handle(), n_rows_of_current_batch, n_neighbors);
      ann_if.search(dev_res, search_params, query_partition, neighbors_view, distances_view);

      int64_t translation_offset = 0;
      for (int r = 0; r < rank; r++) {
        translation_offset += ann_interfaces_[r].size();
      }
      raft::linalg::addScalar(neighbors_view.data_handle(),
                              neighbors_view.data_handle(),
                              (IdxT)translation_offset,
                              part_size,
                              resource::get_cuda_stream(dev_res));

      auto d_trans = raft::make_device_vector<IdxT, IdxT>(dev_res, 2);
      cudaMemsetAsync(
        d_trans.data_handle(), 0, 2 * sizeof(IdxT), resource::get_cuda_stream(dev_res));

      int64_t remaining = num_ranks_;
      int64_t radix     = 2;

      while (remaining > 1) {
        bool received_something = false;
        int64_t offset          = radix / 2;
        comms.group_start();
        if (rank % radix == 0)  // This is one of the receivers
        {
          int other_id = rank + offset;
          if (other_id < num_ranks_)  // Make sure someone's sending anything
          {
            comms.device_recv(tmp_neighbors.data_handle() + part_size,
                              part_size,
                              other_id,
                              resource::get_cuda_stream(dev_res));
            comms.device_recv(tmp_distances.data_handle() + part_size,
                              part_size,
                              other_id,
                              resource::get_cuda_stream(dev_res));
            received_something = true;
          }
        } else if (rank % radix == offset)  // This is one of the senders
        {
          int other_id = rank - offset;
          comms.device_send(
            tmp_neighbors.data_handle(), part_size, other_id, resource::get_cuda_stream(dev_res));
          comms.device_send(
            tmp_distances.data_handle(), part_size, other_id, resource::get_cuda_stream(dev_res));
        }
        comms.group_end();

        remaining = (remaining + 1) / 2;
        radix *= 2;

        if (received_something) {
          // merge inplace
          cuvs::neighbors::detail::knn_merge_parts(tmp_distances.data_handle(),
                                                   tmp_neighbors.data_handle(),
                                                   tmp_distances.data_handle(),
                                                   tmp_neighbors.data_handle(),
                                                   n_rows_of_current_batch,
                                                   2,
                                                   n_neighbors,
                                                   resource::get_cuda_stream(dev_res),
                                                   d_trans.data_handle());

          // If done, copy the final result
          if (remaining <= 1) {
            raft::copy(neighbors.data_handle() + output_offset,
                       tmp_neighbors.data_handle(),
                       part_size,
                       resource::get_cuda_stream(dev_res));
            raft::copy(distances.data_handle() + output_offset,
                       tmp_distances.data_handle(),
                       part_size,
                       resource::get_cuda_stream(dev_res));

            resource::sync_stream(dev_res);
          }
        }
      }
    }
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void ann_mg_index<AnnIndexType, T, IdxT>::serialize(raft::resources const& handle,
                                                    const cuvs::neighbors::mg::nccl_clique& clique,
                                                    const std::string& filename) const
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  serialize_scalar(handle, of, (int)mode_);
  serialize_scalar(handle, of, num_ranks_);
  for (int rank = 0; rank < num_ranks_; rank++) {
    int dev_id                            = clique.device_ids_[rank];
    const raft::device_resources& dev_res = clique.device_resources_[rank];
    RAFT_CUDA_TRY(cudaSetDevice(dev_id));
    auto& ann_if = ann_interfaces_[rank];
    ann_if.serialize(dev_res, of);
  }

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}
}  // namespace cuvs::neighbors::mg

namespace cuvs::neighbors::mg::detail {
using namespace cuvs::neighbors;
using namespace raft;

template <typename T, typename IdxT>
ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> build(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const ivf_flat::mg_index_params& index_params,
  raft::host_matrix_view<const T, int64_t, row_major> index_dataset)
{
  ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> index(index_params.mode, clique.num_ranks_);
  index.build(
    clique, static_cast<const cuvs::neighbors::index_params*>(&index_params), index_dataset);
  return index;
}

template <typename T, typename IdxT>
ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> build(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const ivf_pq::mg_index_params& index_params,
  raft::host_matrix_view<const T, int64_t, row_major> index_dataset)
{
  ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> index(index_params.mode, clique.num_ranks_);
  index.build(
    clique, static_cast<const cuvs::neighbors::index_params*>(&index_params), index_dataset);
  return index;
}

template <typename T, typename IdxT>
ann_mg_index<cagra::index<T, IdxT>, T, IdxT> build(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const cagra::mg_index_params& index_params,
  raft::host_matrix_view<const T, int64_t, row_major> index_dataset)
{
  ann_mg_index<cagra::index<T, IdxT>, T, IdxT> index(index_params.mode, clique.num_ranks_);
  index.build(
    clique, static_cast<const cuvs::neighbors::index_params*>(&index_params), index_dataset);
  return index;
}

template <typename T, typename IdxT>
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
            raft::host_matrix_view<const T, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, IdxT>> new_indices)
{
  index.extend(clique, new_vectors, new_indices);
}

template <typename T, typename IdxT>
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,
            raft::host_matrix_view<const T, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices)
{
  index.extend(clique, new_vectors, new_indices);
}

template <typename T, typename IdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
            const ivf_flat::search_params& search_params,
            raft::host_matrix_view<const T, int64_t, row_major> queries,
            raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            cuvs::neighbors::mg::sharded_merge_mode merge_mode,
            int64_t n_rows_per_batch)
{
  index.search(clique,
               static_cast<const cuvs::neighbors::search_params*>(&search_params),
               queries,
               neighbors,
               distances,
               merge_mode,
               n_rows_per_batch);
}

template <typename T, typename IdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,
            const ivf_pq::search_params& search_params,
            raft::host_matrix_view<const T, int64_t, row_major> queries,
            raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            cuvs::neighbors::mg::sharded_merge_mode merge_mode,
            int64_t n_rows_per_batch)
{
  index.search(clique,
               static_cast<const cuvs::neighbors::search_params*>(&search_params),
               queries,
               neighbors,
               distances,
               merge_mode,
               n_rows_per_batch);
}

template <typename T, typename IdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,
            const cagra::search_params& search_params,
            raft::host_matrix_view<const T, int64_t, row_major> queries,
            raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            cuvs::neighbors::mg::sharded_merge_mode merge_mode,
            int64_t n_rows_per_batch)
{
  index.search(clique,
               static_cast<const cuvs::neighbors::search_params*>(&search_params),
               queries,
               neighbors,
               distances,
               merge_mode,
               n_rows_per_batch);
}

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  index.serialize(handle, clique, filename);
}

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  index.serialize(handle, clique, filename);
}

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  index.serialize(handle, clique, filename);
}

template <typename T, typename IdxT>
ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> deserialize_flat(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const std::string& filename)
{
  auto index = ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>(handle, clique, filename);
  return index;
}

template <typename T, typename IdxT>
ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> deserialize_pq(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const std::string& filename)
{
  auto index = ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>(handle, clique, filename);
  return index;
}

template <typename T, typename IdxT>
ann_mg_index<cagra::index<T, IdxT>, T, IdxT> deserialize_cagra(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const std::string& filename)
{
  auto index = ann_mg_index<cagra::index<T, IdxT>, T, IdxT>(handle, clique, filename);
  return index;
}

template <typename T, typename IdxT>
ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> distribute_flat(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const std::string& filename)
{
  auto index = ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>(REPLICATED, clique.num_ranks_);
  index.deserialize_and_distribute(handle, clique, filename);
  return index;
}

template <typename T, typename IdxT>
ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> distribute_pq(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const std::string& filename)
{
  auto index = ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>(REPLICATED, clique.num_ranks_);
  index.deserialize_and_distribute(handle, clique, filename);
  return index;
}

template <typename T, typename IdxT>
ann_mg_index<cagra::index<T, IdxT>, T, IdxT> distribute_cagra(
  const raft::resources& handle,
  const cuvs::neighbors::mg::nccl_clique& clique,
  const std::string& filename)
{
  auto index = ann_mg_index<cagra::index<T, IdxT>, T, IdxT>(REPLICATED, clique.num_ranks_);
  index.deserialize_and_distribute(handle, clique, filename);
  return index;
}

}  // namespace cuvs::neighbors::mg::detail
