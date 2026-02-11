/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../../distance/sparse_distance.cuh"
#include <cuvs/distance/distance.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/unary_op.cuh>

#include <cuvs/neighbors/knn_merge_parts.hpp>
#include <cuvs/selection/select_k.hpp>

#include <raft/sparse/coo.hpp>
#include <raft/sparse/csr.hpp>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/op/slice.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace cuvs::neighbors::detail {

template <typename ValueIdx, typename value_t>  // NOLINT(readability-identifier-naming)
struct csr_batcher_t {
  csr_batcher_t(ValueIdx batch_size,
                ValueIdx n_rows,
                const ValueIdx* csr_indptr,
                const ValueIdx* csr_indices,
                const value_t* csr_data)
    : batch_start_(0),
      batch_stop_(0),
      batch_rows_(0),
      total_rows_(n_rows),
      batch_size_(batch_size),
      csr_indptr_(csr_indptr),
      csr_indices_(csr_indices),
      csr_data_(csr_data),
      batch_csr_start_offset_(0),
      batch_csr_stop_offset_(0)
  {
  }

  void set_batch(int batch_num)
  {
    batch_start_ = batch_num * batch_size_;
    batch_stop_  = batch_start_ + batch_size_ - 1;  // zero-based indexing

    if (batch_stop_ >= total_rows_) batch_stop_ = total_rows_ - 1;  // zero-based indexing

    batch_rows_ = (batch_stop_ - batch_start_) + 1;
  }

  auto get_batch_csr_indptr_nnz(ValueIdx* batch_indptr, cudaStream_t stream) -> ValueIdx
  {
    raft::sparse::op::csr_row_slice_indptr(batch_start_,
                                           batch_stop_,
                                           csr_indptr_,
                                           batch_indptr,
                                           &batch_csr_start_offset_,
                                           &batch_csr_stop_offset_,
                                           stream);

    return batch_csr_stop_offset_ - batch_csr_start_offset_;
  }

  void get_batch_csr_indices_data(ValueIdx* csr_indices, value_t* csr_data, cudaStream_t stream)
  {
    raft::sparse::op::csr_row_slice_populate(batch_csr_start_offset_,
                                             batch_csr_stop_offset_,
                                             csr_indices_,
                                             csr_data_,
                                             csr_indices,
                                             csr_data,
                                             stream);
  }

  auto batch_rows() const -> ValueIdx { return batch_rows_; }

  auto batch_start() const -> ValueIdx { return batch_start_; }

  auto batch_stop() const -> ValueIdx { return batch_stop_; }

 private:
  ValueIdx batch_size_;
  ValueIdx batch_start_;
  ValueIdx batch_stop_;
  ValueIdx batch_rows_;

  ValueIdx total_rows_;

  const ValueIdx* csr_indptr_;
  const ValueIdx* csr_indices_;
  const value_t* csr_data_;

  ValueIdx batch_csr_start_offset_;
  ValueIdx batch_csr_stop_offset_;
};

template <typename ValueIdx, typename value_t>  // NOLINT(readability-identifier-naming)
class sparse_knn_t {
 public:
  sparse_knn_t(const ValueIdx* idxIndptr_,
               const ValueIdx* idxIndices_,
               const value_t* idxData_,
               size_t idxNNZ_,
               int n_idx_rows_,
               int n_idx_cols_,
               const ValueIdx* queryIndptr_,
               const ValueIdx* queryIndices_,
               const value_t* queryData_,
               size_t queryNNZ_,
               int n_query_rows_,
               int n_query_cols_,
               ValueIdx* output_indices_,
               value_t* output_dists_,
               int k_,
               raft::resources const& handle_,
               size_t batch_size_index_             = 2 << 14,  // approx 1M
               size_t batch_size_query_             = 2 << 14,
               cuvs::distance::DistanceType metric_ = cuvs::distance::DistanceType::L2Expanded,
               float metricArg_                     = 0)
    : idx_indptr_(idxIndptr_),
      idx_indices_(idxIndices_),
      idx_data_(idxData_),
      idx_nnz_(idxNNZ_),
      n_idx_rows_(n_idx_rows_),
      n_idx_cols_(n_idx_cols_),
      query_indptr_(queryIndptr_),
      query_indices_(queryIndices_),
      query_data_(queryData_),
      query_nnz_(queryNNZ_),
      n_query_rows_(n_query_rows_),
      n_query_cols_(n_query_cols_),
      output_indices_(output_indices_),
      output_dists_(output_dists_),
      k_(k_),
      handle(handle_),
      batch_size_index_(batch_size_index_),
      batch_size_query_(batch_size_query_),
      metric_(metric_),
      metric_arg_(metricArg_)
  {
  }

  void run()
  {
    int n_batches_query = raft::ceildiv(static_cast<size_t>(n_query_rows_), batch_size_query_);
    csr_batcher_t<ValueIdx, value_t> query_batcher(
      batch_size_query_, n_query_rows_, query_indptr_, query_indices_, query_data_);

    size_t rows_processed = 0;

    for (int i = 0; i < n_batches_query; i++) {
      /**
       * Compute index batch info
       */
      query_batcher.set_batch(i);

      /**
       * Slice CSR to rows in batch
       */

      rmm::device_uvector<ValueIdx> query_batch_indptr(query_batcher.batch_rows() + 1,
                                                       raft::resource::get_cuda_stream(handle));

      ValueIdx n_query_batch_nnz = query_batcher.get_batch_csr_indptr_nnz(
        query_batch_indptr.data(), raft::resource::get_cuda_stream(handle));

      rmm::device_uvector<ValueIdx> query_batch_indices(n_query_batch_nnz,
                                                        raft::resource::get_cuda_stream(handle));
      rmm::device_uvector<value_t> query_batch_data(n_query_batch_nnz,
                                                    raft::resource::get_cuda_stream(handle));

      query_batcher.get_batch_csr_indices_data(query_batch_indices.data(),
                                               query_batch_data.data(),
                                               raft::resource::get_cuda_stream(handle));

      // A 3-partition temporary merge space to scale the batching. 2 parts for subsequent
      // batches and 1 space for the results of the merge, which get copied back to the top
      rmm::device_uvector<ValueIdx> merge_buffer_indices(0,
                                                         raft::resource::get_cuda_stream(handle));
      rmm::device_uvector<value_t> merge_buffer_dists(0, raft::resource::get_cuda_stream(handle));

      value_t* dists_merge_buffer_ptr;
      ValueIdx* indices_merge_buffer_ptr;

      int n_batches_idx = raft::ceildiv(static_cast<size_t>(n_idx_rows_), batch_size_index_);
      csr_batcher_t<ValueIdx, value_t> idx_batcher(
        batch_size_index_, n_idx_rows_, idx_indptr_, idx_indices_, idx_data_);

      for (int j = 0; j < n_batches_idx; j++) {
        idx_batcher.set_batch(j);

        merge_buffer_indices.resize(query_batcher.batch_rows() * k_ * 3,
                                    raft::resource::get_cuda_stream(handle));
        merge_buffer_dists.resize(query_batcher.batch_rows() * k_ * 3,
                                  raft::resource::get_cuda_stream(handle));

        /**
         * Slice CSR to rows in batch
         */
        rmm::device_uvector<ValueIdx> idx_batch_indptr(idx_batcher.batch_rows() + 1,
                                                       raft::resource::get_cuda_stream(handle));
        rmm::device_uvector<ValueIdx> idx_batch_indices(0, raft::resource::get_cuda_stream(handle));
        rmm::device_uvector<value_t> idx_batch_data(0, raft::resource::get_cuda_stream(handle));

        ValueIdx idx_batch_nnz = idx_batcher.get_batch_csr_indptr_nnz(
          idx_batch_indptr.data(), raft::resource::get_cuda_stream(handle));

        idx_batch_indices.resize(idx_batch_nnz, raft::resource::get_cuda_stream(handle));
        idx_batch_data.resize(idx_batch_nnz, raft::resource::get_cuda_stream(handle));

        idx_batcher.get_batch_csr_indices_data(
          idx_batch_indices.data(), idx_batch_data.data(), raft::resource::get_cuda_stream(handle));

        /**
         * Compute distances
         */
        uint64_t dense_size = static_cast<uint64_t>(idx_batcher.batch_rows()) *
                              static_cast<uint64_t>(query_batcher.batch_rows());
        rmm::device_uvector<value_t> batch_dists(dense_size,
                                                 raft::resource::get_cuda_stream(handle));

        RAFT_CUDA_TRY(cudaMemset(batch_dists.data(), 0, batch_dists.size() * sizeof(value_t)));

        compute_distances(idx_batcher,
                          query_batcher,
                          idx_batch_nnz,
                          n_query_batch_nnz,
                          idx_batch_indptr.data(),
                          idx_batch_indices.data(),
                          idx_batch_data.data(),
                          query_batch_indptr.data(),
                          query_batch_indices.data(),
                          query_batch_data.data(),
                          batch_dists.data());

        // Build batch indices array
        rmm::device_uvector<ValueIdx> batch_indices(batch_dists.size(),
                                                    raft::resource::get_cuda_stream(handle));

        // populate batch indices array
        ValueIdx batch_rows = query_batcher.batch_rows(), batch_cols = idx_batcher.batch_rows();

        raft::sparse::iota_fill(
          batch_indices.data(), batch_rows, batch_cols, raft::resource::get_cuda_stream(handle));

        /**
         * Perform k_-selection on batch & merge with other k_-selections
         */
        size_t merge_buffer_offset = batch_rows * k_;
        dists_merge_buffer_ptr     = merge_buffer_dists.data() + merge_buffer_offset;
        indices_merge_buffer_ptr   = merge_buffer_indices.data() + merge_buffer_offset;

        perform_k_selection(idx_batcher,
                            query_batcher,
                            batch_dists.data(),
                            batch_indices.data(),
                            dists_merge_buffer_ptr,
                            indices_merge_buffer_ptr);

        value_t* dists_merge_buffer_tmp_ptr    = dists_merge_buffer_ptr;
        ValueIdx* indices_merge_buffer_tmp_ptr = indices_merge_buffer_ptr;

        // Merge results of difference batches if necessary
        if (idx_batcher.batch_start() > 0) {
          size_t merge_buffer_tmp_out  = batch_rows * k_ * 2;
          dists_merge_buffer_tmp_ptr   = merge_buffer_dists.data() + merge_buffer_tmp_out;
          indices_merge_buffer_tmp_ptr = merge_buffer_indices.data() + merge_buffer_tmp_out;

          merge_batches(idx_batcher,
                        query_batcher,
                        merge_buffer_dists.data(),
                        merge_buffer_indices.data(),
                        dists_merge_buffer_tmp_ptr,
                        indices_merge_buffer_tmp_ptr);
        }

        // copy merged output back into merge buffer partition for next iteration
        raft::copy_async<ValueIdx>(merge_buffer_indices.data(),
                                   indices_merge_buffer_tmp_ptr,
                                   batch_rows * k_,
                                   raft::resource::get_cuda_stream(handle));
        raft::copy_async<value_t>(merge_buffer_dists.data(),
                                  dists_merge_buffer_tmp_ptr,
                                  batch_rows * k_,
                                  raft::resource::get_cuda_stream(handle));
      }

      // Copy final merged batch to output array
      raft::copy_async<ValueIdx>(output_indices_ + (rows_processed * k_),
                                 merge_buffer_indices.data(),
                                 query_batcher.batch_rows() * k_,
                                 raft::resource::get_cuda_stream(handle));
      raft::copy_async<value_t>(output_dists_ + (rows_processed * k_),
                                merge_buffer_dists.data(),
                                query_batcher.batch_rows() * k_,
                                raft::resource::get_cuda_stream(handle));

      rows_processed += query_batcher.batch_rows();
    }
  }

 private:
  void merge_batches(csr_batcher_t<ValueIdx, value_t>& idx_batcher,
                     csr_batcher_t<ValueIdx, value_t>& query_batcher,
                     value_t* merge_buffer_dists,
                     ValueIdx* merge_buffer_indices,
                     value_t* out_dists,
                     ValueIdx* out_indices)
  {
    // build translation buffer to shift resulting indices by the batch
    std::vector<ValueIdx> id_ranges;
    id_ranges.push_back(0);
    id_ranges.push_back(idx_batcher.batch_start());

    rmm::device_uvector<ValueIdx> trans(id_ranges.size(), raft::resource::get_cuda_stream(handle));
    raft::update_device(
      trans.data(), id_ranges.data(), id_ranges.size(), raft::resource::get_cuda_stream(handle));

    // combine merge buffers only if there's more than 1 partition to combine
    auto rows = query_batcher.batch_rows();
    knn_merge_parts(
      handle,
      raft::make_device_matrix_view<const value_t, int64_t>(merge_buffer_dists, rows, 2 * k_),
      raft::make_device_matrix_view<const ValueIdx, int64_t>(merge_buffer_indices, rows, 2 * k_),
      raft::make_device_matrix_view<value_t, int64_t>(out_dists, rows, k_),
      raft::make_device_matrix_view<ValueIdx, int64_t>(out_indices, rows, k_),
      raft::make_device_vector_view<ValueIdx, int64_t>(trans.data(), id_ranges.size()));
  }

  void perform_k_selection(csr_batcher_t<ValueIdx, value_t> idx_batcher,
                           csr_batcher_t<ValueIdx, value_t> query_batcher,
                           value_t* batch_dists,
                           ValueIdx* batch_indices,
                           value_t* out_dists,
                           ValueIdx* out_indices)
  {
    // populate batch indices array
    ValueIdx batch_rows = query_batcher.batch_rows(), batch_cols = idx_batcher.batch_rows();

    // build translation buffer to shift resulting indices by the batch
    std::vector<ValueIdx> id_ranges;
    id_ranges.push_back(0);
    id_ranges.push_back(idx_batcher.batch_start());

    // in the case where the number of idx rows in the batch is < k_, we
    // want to adjust k_.
    ValueIdx n_neighbors = std::min(static_cast<ValueIdx>(k_), batch_cols);

    bool ascending = cuvs::distance::is_min_close(metric_);

    // kernel to slice first (min) k_ cols and copy into batched merge buffer
    cuvs::selection::select_k(
      handle,
      raft::make_device_matrix_view<const value_t, int64_t>(batch_dists, batch_rows, batch_cols),
      raft::make_device_matrix_view<const ValueIdx, int64_t>(batch_indices, batch_rows, batch_cols),
      raft::make_device_matrix_view<value_t, int64_t>(out_dists, batch_rows, n_neighbors),
      raft::make_device_matrix_view<ValueIdx, int64_t>(out_indices, batch_rows, n_neighbors),
      ascending,
      true);
  }

  void compute_distances(csr_batcher_t<ValueIdx, value_t>& idx_batcher,
                         csr_batcher_t<ValueIdx, value_t>& query_batcher,
                         size_t idx_batch_nnz,
                         size_t query_batch_nnz,
                         ValueIdx* idx_batch_indptr,
                         ValueIdx* idx_batch_indices,
                         value_t* idx_batch_data,
                         ValueIdx* query_batch_indptr,
                         ValueIdx* query_batch_indices,
                         value_t* query_batch_data,
                         value_t* batch_dists)
  {
    /**
     * Compute distances
     */
    cuvs::distance::detail::sparse::distances_config_t<ValueIdx, value_t> dist_config(handle);
    dist_config.b_nrows = idx_batcher.batch_rows();
    dist_config.b_ncols = n_idx_cols_;
    dist_config.b_nnz   = idx_batch_nnz;

    dist_config.b_indptr  = idx_batch_indptr;
    dist_config.b_indices = idx_batch_indices;
    dist_config.b_data    = idx_batch_data;

    dist_config.a_nrows = query_batcher.batch_rows();
    dist_config.a_ncols = n_query_cols_;
    dist_config.a_nnz   = query_batch_nnz;

    dist_config.a_indptr  = query_batch_indptr;
    dist_config.a_indices = query_batch_indices;
    dist_config.a_data    = query_batch_data;

    cuvs::distance::pairwise_distance(batch_dists, dist_config, metric_, metric_arg_);
  }

  const ValueIdx *idx_indptr_, *idx_indices_, *query_indptr_, *query_indices_;
  ValueIdx* output_indices_;
  const value_t *idx_data_, *query_data_;
  value_t* output_dists_;

  size_t idx_nnz_, query_nnz_, batch_size_index_, batch_size_query_;

  cuvs::distance::DistanceType metric_;

  float metric_arg_;

  int n_idx_rows_, n_idx_cols_, n_query_rows_, n_query_cols_, k_;

  raft::resources const& handle;  // NOLINT(readability-identifier-naming)
};

};  // namespace cuvs::neighbors::detail
