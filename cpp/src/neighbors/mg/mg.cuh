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
#include <raft/core/resource/nccl_clique.hpp>
#include <raft/core/serialize.hpp>
#include <raft/linalg/add.cuh>
#include <raft/util/cuda_dev_essentials.cuh>

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/mg.hpp>

namespace cuvs::neighbors {
using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT>
void search(const raft::device_resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::host_matrix_view<const T, int64_t, row_major> h_queries,
            raft::device_matrix_view<IdxT, int64_t, row_major> d_neighbors,
            raft::device_matrix_view<float, int64_t, row_major> d_distances);
}  // namespace cuvs::neighbors

namespace cuvs::neighbors::mg {
void check_omp_threads(const int requirements);
}  // namespace cuvs::neighbors::mg

namespace cuvs::neighbors::mg::detail {
using namespace cuvs::neighbors;
using namespace raft;

// local index deserialization and distribution
template <typename AnnIndexType, typename T, typename IdxT>
void deserialize_and_distribute(const raft::device_resources& handle,
                                index<AnnIndexType, T, IdxT>& index,
                                const std::string& filename)
{
  const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);
  for (int rank = 0; rank < index.num_ranks_; rank++) {
    int dev_id                            = clique.device_ids_[rank];
    const raft::device_resources& dev_res = clique.device_resources_[rank];
    RAFT_CUDA_TRY(cudaSetDevice(dev_id));
    auto& ann_if = index.ann_interfaces_.emplace_back();
    cuvs::neighbors::deserialize(dev_res, ann_if, filename);
  }
}

// MG index deserialization
template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::device_resources& handle,
                 index<AnnIndexType, T, IdxT>& index,
                 const std::string& filename)
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);

  index.mode_      = (cuvs::neighbors::mg::distribution_mode)deserialize_scalar<int>(handle, is);
  index.num_ranks_ = deserialize_scalar<int>(handle, is);

  if (index.num_ranks_ != clique.num_ranks_) {
    RAFT_FAIL("Serialized index has %d ranks whereas NCCL clique has %d ranks",
              index.num_ranks_,
              clique.num_ranks_);
  }

  for (int rank = 0; rank < index.num_ranks_; rank++) {
    int dev_id                            = clique.device_ids_[rank];
    const raft::device_resources& dev_res = clique.device_resources_[rank];
    RAFT_CUDA_TRY(cudaSetDevice(dev_id));
    auto& ann_if = index.ann_interfaces_.emplace_back();
    cuvs::neighbors::deserialize(dev_res, ann_if, is);
  }

  is.close();
}

template <typename AnnIndexType, typename T, typename IdxT>
void build(const raft::device_resources& handle,
           index<AnnIndexType, T, IdxT>& index,
           const cuvs::neighbors::index_params* index_params,
           raft::host_matrix_view<const T, int64_t, row_major> index_dataset)
{
  const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);

  if (index.mode_ == REPLICATED) {
    int64_t n_rows = index_dataset.extent(0);
    RAFT_LOG_INFO("REPLICATED BUILD: %d*%drows", index.num_ranks_, n_rows);

    index.ann_interfaces_.resize(index.num_ranks_);
#pragma omp parallel for
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));
      auto& ann_if = index.ann_interfaces_[rank];
      cuvs::neighbors::build(dev_res, ann_if, index_params, index_dataset);
      resource::sync_stream(dev_res);
    }
  } else if (index.mode_ == SHARDED) {
    int64_t n_rows           = index_dataset.extent(0);
    int64_t n_cols           = index_dataset.extent(1);
    int64_t n_rows_per_shard = raft::ceildiv(n_rows, (int64_t)index.num_ranks_);

    RAFT_LOG_INFO("SHARDED BUILD: %d*%drows", index.num_ranks_, n_rows_per_shard);

    index.ann_interfaces_.resize(index.num_ranks_);
#pragma omp parallel for
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));
      int64_t offset                  = rank * n_rows_per_shard;
      int64_t n_rows_of_current_shard = std::min(n_rows_per_shard, n_rows - offset);
      const T* partition_ptr          = index_dataset.data_handle() + (offset * n_cols);
      auto partition                  = raft::make_host_matrix_view<const T, int64_t, row_major>(
        partition_ptr, n_rows_of_current_shard, n_cols);
      auto& ann_if = index.ann_interfaces_[rank];
      cuvs::neighbors::build(dev_res, ann_if, index_params, partition);
      resource::sync_stream(dev_res);
    }
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void extend(const raft::device_resources& handle,
            index<AnnIndexType, T, IdxT>& index,
            raft::host_matrix_view<const T, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices)
{
  const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);

  int64_t n_rows = new_vectors.extent(0);
  if (index.mode_ == REPLICATED) {
    RAFT_LOG_INFO("REPLICATED EXTEND: %d*%drows", index.num_ranks_, n_rows);

#pragma omp parallel for
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));
      auto& ann_if = index.ann_interfaces_[rank];
      cuvs::neighbors::extend(dev_res, ann_if, new_vectors, new_indices);
      resource::sync_stream(dev_res);
    }
  } else if (index.mode_ == SHARDED) {
    int64_t n_cols           = new_vectors.extent(1);
    int64_t n_rows_per_shard = raft::ceildiv(n_rows, (int64_t)index.num_ranks_);

    RAFT_LOG_INFO("SHARDED EXTEND: %d*%drows", index.num_ranks_, n_rows_per_shard);

#pragma omp parallel for
    for (int rank = 0; rank < index.num_ranks_; rank++) {
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
      auto& ann_if = index.ann_interfaces_[rank];
      cuvs::neighbors::extend(dev_res, ann_if, new_vectors_part, new_indices_part);
      resource::sync_stream(dev_res);
    }
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void sharded_search_with_direct_merge(const raft::comms::nccl_clique& clique,
                                      const index<AnnIndexType, T, IdxT>& index,
                                      const cuvs::neighbors::search_params* search_params,
                                      raft::host_matrix_view<const T, int64_t, row_major> queries,
                                      raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
                                      raft::host_matrix_view<float, int64_t, row_major> distances,
                                      int64_t n_rows_per_batch,
                                      int64_t n_rows,
                                      int64_t n_cols,
                                      int64_t n_neighbors,
                                      int64_t n_batches)
{
  const auto& root_handle = clique.set_current_device_to_root_rank();
  auto in_neighbors       = raft::make_device_matrix<IdxT, int64_t, row_major>(
    root_handle, index.num_ranks_ * n_rows_per_batch, n_neighbors);
  auto in_distances = raft::make_device_matrix<float, int64_t, row_major>(
    root_handle, index.num_ranks_ * n_rows_per_batch, n_neighbors);
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

    const int& requirements = index.num_ranks_;
    check_omp_threads(requirements);  // should use at least num_ranks_ threads to avoid NCCL hang
#pragma omp parallel for num_threads(index.num_ranks_)
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      auto& ann_if                          = index.ann_interfaces_[rank];
      RAFT_CUDA_TRY(cudaSetDevice(dev_id));

      if (rank == clique.root_rank_) {  // root rank
        uint64_t batch_offset = clique.root_rank_ * part_size;
        auto d_neighbors      = raft::make_device_matrix_view<IdxT, int64_t, row_major>(
          in_neighbors.data_handle() + batch_offset, n_rows_of_current_batch, n_neighbors);
        auto d_distances = raft::make_device_matrix_view<float, int64_t, row_major>(
          in_distances.data_handle() + batch_offset, n_rows_of_current_batch, n_neighbors);
        cuvs::neighbors::search(
          dev_res, ann_if, search_params, query_partition, d_neighbors, d_distances);

        // wait for other ranks
        ncclGroupStart();
        for (int from_rank = 0; from_rank < index.num_ranks_; from_rank++) {
          if (from_rank == clique.root_rank_) continue;

          batch_offset = from_rank * part_size;
          ncclRecv(in_neighbors.data_handle() + batch_offset,
                   part_size * sizeof(IdxT),
                   ncclUint8,
                   from_rank,
                   clique.nccl_comms_[rank],
                   resource::get_cuda_stream(dev_res));
          ncclRecv(in_distances.data_handle() + batch_offset,
                   part_size * sizeof(float),
                   ncclUint8,
                   from_rank,
                   clique.nccl_comms_[rank],
                   resource::get_cuda_stream(dev_res));
        }
        ncclGroupEnd();
        resource::sync_stream(dev_res);
      } else {  // non-root ranks
        auto d_neighbors = raft::make_device_matrix<IdxT, int64_t, row_major>(
          dev_res, n_rows_of_current_batch, n_neighbors);
        auto d_distances = raft::make_device_matrix<float, int64_t, row_major>(
          dev_res, n_rows_of_current_batch, n_neighbors);
        cuvs::neighbors::search(
          dev_res, ann_if, search_params, query_partition, d_neighbors.view(), d_distances.view());

        // send results to root rank
        ncclGroupStart();
        ncclSend(d_neighbors.data_handle(),
                 part_size * sizeof(IdxT),
                 ncclUint8,
                 clique.root_rank_,
                 clique.nccl_comms_[rank],
                 resource::get_cuda_stream(dev_res));
        ncclSend(d_distances.data_handle(),
                 part_size * sizeof(float),
                 ncclUint8,
                 clique.root_rank_,
                 clique.nccl_comms_[rank],
                 resource::get_cuda_stream(dev_res));
        ncclGroupEnd();
        resource::sync_stream(dev_res);
      }
    }

    const auto& root_handle_   = clique.set_current_device_to_root_rank();
    auto h_trans               = std::vector<IdxT>(index.num_ranks_);
    int64_t translation_offset = 0;
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      h_trans[rank] = translation_offset;
      translation_offset += index.ann_interfaces_[rank].size();
    }
    auto d_trans = raft::make_device_vector<IdxT, IdxT>(root_handle_, index.num_ranks_);
    raft::copy(d_trans.data_handle(),
               h_trans.data(),
               index.num_ranks_,
               resource::get_cuda_stream(root_handle_));

    cuvs::neighbors::detail::knn_merge_parts(in_distances.data_handle(),
                                             in_neighbors.data_handle(),
                                             out_distances.data_handle(),
                                             out_neighbors.data_handle(),
                                             n_rows_of_current_batch,
                                             index.num_ranks_,
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
void sharded_search_with_tree_merge(const raft::comms::nccl_clique& clique,
                                    const index<AnnIndexType, T, IdxT>& index,
                                    const cuvs::neighbors::search_params* search_params,
                                    raft::host_matrix_view<const T, int64_t, row_major> queries,
                                    raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
                                    raft::host_matrix_view<float, int64_t, row_major> distances,
                                    int64_t n_rows_per_batch,
                                    int64_t n_rows,
                                    int64_t n_cols,
                                    int64_t n_neighbors,
                                    int64_t n_batches)
{
  for (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
    int64_t offset                  = batch_idx * n_rows_per_batch;
    int64_t query_offset            = offset * n_cols;
    int64_t output_offset           = offset * n_neighbors;
    int64_t n_rows_of_current_batch = std::min((int64_t)n_rows_per_batch, n_rows - offset);
    auto query_partition            = raft::make_host_matrix_view<const T, int64_t, row_major>(
      queries.data_handle() + query_offset, n_rows_of_current_batch, n_cols);

    const int& requirements = index.num_ranks_;
    check_omp_threads(requirements);  // should use at least num_ranks_ threads to avoid NCCL hang
#pragma omp parallel for num_threads(index.num_ranks_)
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      int dev_id                            = clique.device_ids_[rank];
      const raft::device_resources& dev_res = clique.device_resources_[rank];
      auto& ann_if                          = index.ann_interfaces_[rank];
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
      cuvs::neighbors::search(
        dev_res, ann_if, search_params, query_partition, neighbors_view, distances_view);

      int64_t translation_offset = 0;
      for (int r = 0; r < rank; r++) {
        translation_offset += index.ann_interfaces_[r].size();
      }
      raft::linalg::addScalar(neighbors_view.data_handle(),
                              neighbors_view.data_handle(),
                              (IdxT)translation_offset,
                              part_size,
                              resource::get_cuda_stream(dev_res));

      auto d_trans = raft::make_device_vector<IdxT, IdxT>(dev_res, 2);
      cudaMemsetAsync(
        d_trans.data_handle(), 0, 2 * sizeof(IdxT), resource::get_cuda_stream(dev_res));

      int64_t remaining = index.num_ranks_;
      int64_t radix     = 2;

      while (remaining > 1) {
        bool received_something = false;
        int64_t offset          = radix / 2;
        ncclGroupStart();
        if (rank % radix == 0)  // This is one of the receivers
        {
          int other_id = rank + offset;
          if (other_id < index.num_ranks_)  // Make sure someone's sending anything
          {
            ncclRecv(tmp_neighbors.data_handle() + part_size,
                     part_size * sizeof(IdxT),
                     ncclUint8,
                     other_id,
                     clique.nccl_comms_[rank],
                     resource::get_cuda_stream(dev_res));
            ncclRecv(tmp_distances.data_handle() + part_size,
                     part_size * sizeof(float),
                     ncclUint8,
                     other_id,
                     clique.nccl_comms_[rank],
                     resource::get_cuda_stream(dev_res));
            received_something = true;
          }
        } else if (rank % radix == offset)  // This is one of the senders
        {
          int other_id = rank - offset;
          ncclSend(tmp_neighbors.data_handle(),
                   part_size * sizeof(IdxT),
                   ncclUint8,
                   other_id,
                   clique.nccl_comms_[rank],
                   resource::get_cuda_stream(dev_res));
          ncclSend(tmp_distances.data_handle(),
                   part_size * sizeof(float),
                   ncclUint8,
                   other_id,
                   clique.nccl_comms_[rank],
                   resource::get_cuda_stream(dev_res));
        }
        ncclGroupEnd();

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
void run_search_batch(const raft::comms::nccl_clique& clique,
                      const index<AnnIndexType, T, IdxT>& index,
                      int rank,
                      const cuvs::neighbors::search_params* search_params,
                      raft::host_matrix_view<const T, int64_t, row_major>& queries,
                      raft::host_matrix_view<IdxT, int64_t, row_major>& neighbors,
                      raft::host_matrix_view<float, int64_t, row_major>& distances,
                      int64_t query_offset,
                      int64_t output_offset,
                      int64_t n_rows_of_current_batch,
                      int64_t n_cols,
                      int64_t n_neighbors)
{
  int dev_id = clique.device_ids_[rank];
  RAFT_CUDA_TRY(cudaSetDevice(dev_id));
  const raft::device_resources& dev_res = clique.device_resources_[rank];
  auto& ann_if                          = index.ann_interfaces_[rank];

  auto query_partition = raft::make_host_matrix_view<const T, int64_t, row_major>(
    queries.data_handle() + query_offset, n_rows_of_current_batch, n_cols);
  auto d_neighbors = raft::make_device_matrix<IdxT, int64_t, row_major>(
    dev_res, n_rows_of_current_batch, n_neighbors);
  auto d_distances = raft::make_device_matrix<float, int64_t, row_major>(
    dev_res, n_rows_of_current_batch, n_neighbors);

  cuvs::neighbors::search(
    dev_res, ann_if, search_params, query_partition, d_neighbors.view(), d_distances.view());

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

template <typename AnnIndexType, typename T, typename IdxT>
void search(const raft::device_resources& handle,
            const index<AnnIndexType, T, IdxT>& index,
            const cuvs::neighbors::search_params* search_params,
            raft::host_matrix_view<const T, int64_t, row_major> queries,
            raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch)
{
  const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);

  int64_t n_rows      = queries.extent(0);
  int64_t n_cols      = queries.extent(1);
  int64_t n_neighbors = neighbors.extent(1);

  if (index.mode_ == REPLICATED) {
    cuvs::neighbors::mg::replicated_search_mode search_mode;
    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      const cuvs::neighbors::mg::search_params<ivf_flat::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg::search_params<ivf_flat::search_params>*>(
          search_params);
      search_mode = mg_search_params->search_mode;
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      const cuvs::neighbors::mg::search_params<ivf_pq::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg::search_params<ivf_pq::search_params>*>(
          search_params);
      search_mode = mg_search_params->search_mode;
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      const cuvs::neighbors::mg::search_params<cagra::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg::search_params<cagra::search_params>*>(search_params);
      search_mode = mg_search_params->search_mode;
    }

    if (search_mode == LOAD_BALANCER) {
      int64_t n_rows_per_rank = raft::ceildiv(n_rows, (int64_t)index.num_ranks_);
      n_rows_per_batch =
        std::min(n_rows_per_batch, n_rows_per_rank);  // get at least num_ranks_ batches
      int64_t n_batches = raft::ceildiv(n_rows, (int64_t)n_rows_per_batch);
      if (n_batches <= 1) n_rows_per_batch = n_rows;

      RAFT_LOG_INFO(
        "REPLICATED SEARCH IN LOAD BALANCER MODE: %d*%drows", n_batches, n_rows_per_batch);

#pragma omp parallel for
      for (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        int rank                        = batch_idx % index.num_ranks_;  // alternate GPUs
        int64_t offset                  = batch_idx * n_rows_per_batch;
        int64_t query_offset            = offset * n_cols;
        int64_t output_offset           = offset * n_neighbors;
        int64_t n_rows_of_current_batch = std::min(n_rows_per_batch, n_rows - offset);

        run_search_batch(clique,
                         index,
                         rank,
                         search_params,
                         queries,
                         neighbors,
                         distances,
                         query_offset,
                         output_offset,
                         n_rows_of_current_batch,
                         n_cols,
                         n_neighbors);
      }
    } else if (search_mode == ROUND_ROBIN) {
      RAFT_LOG_INFO("REPLICATED SEARCH IN ROUND ROBIN MODE: %d*%drows", 1, n_rows);

      ASSERT(n_rows <= n_rows_per_batch,
             "In round-robin mode, n_rows must lower or equal to n_rows_per_batch");

      auto& rrc    = *index.round_robin_counter_;
      int64_t rank = rrc++;
      rank %= index.num_ranks_;

      run_search_batch(clique,
                       index,
                       rank,
                       search_params,
                       queries,
                       neighbors,
                       distances,
                       0,
                       0,
                       n_rows,
                       n_cols,
                       n_neighbors);
    }
  } else if (index.mode_ == SHARDED) {
    cuvs::neighbors::mg::sharded_merge_mode merge_mode;
    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      const cuvs::neighbors::mg::search_params<ivf_flat::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg::search_params<ivf_flat::search_params>*>(
          search_params);
      merge_mode = mg_search_params->merge_mode;
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      const cuvs::neighbors::mg::search_params<ivf_pq::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg::search_params<ivf_pq::search_params>*>(
          search_params);
      merge_mode = mg_search_params->merge_mode;
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      const cuvs::neighbors::mg::search_params<cagra::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg::search_params<cagra::search_params>*>(search_params);
      merge_mode = mg_search_params->merge_mode;
    }

    int64_t n_batches = raft::ceildiv(n_rows, (int64_t)n_rows_per_batch);
    if (n_batches <= 1) n_rows_per_batch = n_rows;

    if (merge_mode == MERGE_ON_ROOT_RANK) {
      RAFT_LOG_INFO("SHARDED SEARCH WITH MERGE_ON_ROOT_RANK MERGE MODE: %d*%drows",
                    n_batches,
                    n_rows_per_batch);
      sharded_search_with_direct_merge(clique,
                                       index,
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
                                     index,
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
void serialize(const raft::device_resources& handle,
               const index<AnnIndexType, T, IdxT>& index,
               const std::string& filename)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);

  serialize_scalar(handle, of, (int)index.mode_);
  serialize_scalar(handle, of, index.num_ranks_);

  for (int rank = 0; rank < index.num_ranks_; rank++) {
    int dev_id                            = clique.device_ids_[rank];
    const raft::device_resources& dev_res = clique.device_resources_[rank];
    RAFT_CUDA_TRY(cudaSetDevice(dev_id));
    auto& ann_if = index.ann_interfaces_[rank];
    cuvs::neighbors::serialize(dev_res, ann_if, of);
  }

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

}  // namespace cuvs::neighbors::mg::detail

namespace cuvs::neighbors::mg {
using namespace cuvs::neighbors;
using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT>
index<AnnIndexType, T, IdxT>::index(distribution_mode mode, int num_ranks_)
  : mode_(mode),
    num_ranks_(num_ranks_),
    round_robin_counter_(std::make_shared<std::atomic<int64_t>>(0))
{
}

template <typename AnnIndexType, typename T, typename IdxT>
index<AnnIndexType, T, IdxT>::index(const raft::device_resources& handle,
                                    const std::string& filename)
  : round_robin_counter_(std::make_shared<std::atomic<int64_t>>(0))
{
  cuvs::neighbors::mg::detail::deserialize(handle, *this, filename);
}
}  // namespace cuvs::neighbors::mg
