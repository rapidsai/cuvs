/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

template <class T, class IdxT>
merge_result<T, IdxT> merge(raft::resources const& handle,
                            const cagra::index_params& params,
                            std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                            const cuvs::neighbors::filtering::base_filter& row_filter)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type   = typename cagra_index_t::dataset_index_type;

  std::size_t dim              = 0;
  std::size_t new_dataset_size = 0;
  int64_t stride               = -1;

  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bitmap,
               "Bitmap filter isn't supported inside cagra::merge");

  for (cagra_index_t* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());
        strided_dset != nullptr) {
      if (dim == 0) {
        dim    = index->dim();
        stride = strided_dset->stride();
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
        RAFT_EXPECTS(stride == strided_dset->stride(),
                     "Row stride of datasets in indices must be equal.");
      }
      new_dataset_size += index->size();
    } else if (auto* padded_dset =
                 dynamic_cast<const cuvs::neighbors::device_padded_dataset_view<T, ds_idx_type>*>(
                   &index->data());
               padded_dset != nullptr) {
      if (dim == 0) {
        dim    = index->dim();
        stride = padded_dset->stride();
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
        RAFT_EXPECTS(stride == padded_dset->stride(),
                     "Row stride of datasets in indices must be equal.");
      }
      new_dataset_size += index->size();
    } else if (dynamic_cast<const cuvs::neighbors::empty_dataset<int64_t>*>(&index->data()) !=
               nullptr) {
      RAFT_FAIL(
        "cagra::merge only supports an index to which the dataset is attached. Please check if the "
        "index was built with index_param.attach_dataset_on_build = true, or if a dataset was "
        "attached after the build.");
    } else {
      RAFT_FAIL("cagra::merge only supports an uncompressed dataset index");
    }
  }

  // Destination leading dimension in elements. Use the same row pitch as the inputs so merged rows
  // stay alignment-safe (same contract as make_padded_dataset / device_padded_dataset_view). Using
  // ld == dim would pack rows tightly and can break 16-byte vectorized loads when dim * sizeof(T)
  // is not a multiple of lcm(16, sizeof(T)).
  auto merge_dataset = [&](T* dst, std::size_t dst_ld) {
    IdxT row_offset = 0;
    for (cagra_index_t* index : indices) {
      const T* src_ptr   = nullptr;
      std::size_t n_rows = 0;
      if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());
          strided_dset != nullptr) {
        src_ptr = strided_dset->view().data_handle();
        n_rows  = static_cast<std::size_t>(strided_dset->n_rows());
      } else if (auto* padded_dset =
                   dynamic_cast<const cuvs::neighbors::device_padded_dataset_view<T, ds_idx_type>*>(
                     &index->data());
                 padded_dset != nullptr) {
        src_ptr = padded_dset->view().data_handle();
        n_rows  = static_cast<std::size_t>(padded_dset->n_rows());
      } else {
        RAFT_FAIL("cagra::merge: unexpected dataset type while copying rows");
      }
      raft::copy_matrix(dst + static_cast<std::size_t>(row_offset) * dst_ld,
                        dst_ld,
                        src_ptr,
                        static_cast<std::size_t>(stride),
                        dim,
                        n_rows,
                        raft::resource::get_cuda_stream(handle));

      row_offset += IdxT(index->data().n_rows());
    }
  };

  try {
    auto updated_dataset = raft::make_device_matrix<T, int64_t>(
      handle, int64_t(new_dataset_size), static_cast<int64_t>(stride));
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      updated_dataset.data_handle(), 0, updated_dataset.size() * sizeof(T), stream));

    merge_dataset(updated_dataset.data_handle(), static_cast<std::size_t>(stride));

    if (row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset) {
      auto actual_filter =
        dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(
          row_filter);
      auto filtered_row_count = actual_filter.view().count(handle);

      // Convert the filter to a CSR matrix (so that we can pass indices to raft::copy_rows)
      auto indices_csr = raft::make_device_csr_matrix<uint32_t, int64_t, int64_t, int64_t>(
        handle, 1, new_dataset_size);
      indices_csr.initialize_sparsity(filtered_row_count);

      actual_filter.view().to_csr(handle, indices_csr);

      // Get the indices array from the csr matrix. Note that this returns a raft::span object
      // and we need to pass as device_vector_view, which is a 1D mdspan (instead of a span)
      // so we need to translate here (and adjust to be const)
      auto indices      = indices_csr.structure_view().get_indices();
      auto indices_view = raft::make_device_vector_view<const int64_t, int64_t>(
        indices.data(), static_cast<int64_t>(indices.size()));

      auto filtered_dataset = raft::make_device_matrix<T, int64_t>(
        handle, filtered_row_count, static_cast<int64_t>(stride));
      RAFT_CUDA_TRY(cudaMemsetAsync(
        filtered_dataset.data_handle(), 0, filtered_dataset.size() * sizeof(T), stream));
      raft::matrix::copy_rows(handle,
                              raft::make_const_mdspan(updated_dataset.view()),
                              filtered_dataset.view(),
                              indices_view);

      cuvs::neighbors::device_padded_dataset_view<T, int64_t> dv(
        raft::make_const_mdspan(filtered_dataset.view()), static_cast<uint32_t>(dim));
      auto build_res = cagra::detail::build<T, IdxT>(handle, params, dv);
      RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
      return cagra::merge_result<T, IdxT>{std::move(build_res.idx), std::move(filtered_dataset)};
    } else {
      cuvs::neighbors::device_padded_dataset_view<T, int64_t> dv(
        raft::make_const_mdspan(updated_dataset.view()), static_cast<uint32_t>(dim));
      auto build_res = cagra::detail::build<T, IdxT>(handle, params, dv);
      RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
      return cagra::merge_result<T, IdxT>{std::move(build_res.idx), std::move(updated_dataset)};
    }
  } catch (std::bad_alloc& e) {
    // We don't currently support the cpu memory fallback with filtered merge, since the
    // 'raft::matrix::copy_rows' only supports gpu memory
    RAFT_EXPECTS(row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::None,
                 "Filtered merge isn't available on cpu memory");

    RAFT_LOG_DEBUG("cagra::merge: using host memory for merged dataset");

    auto updated_dataset =
      raft::make_host_matrix<T, std::int64_t>(std::int64_t(new_dataset_size), std::int64_t(dim));

    merge_dataset(updated_dataset.data_handle(), dim);

    auto host_view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      updated_dataset.data_handle(), updated_dataset.extent(0), updated_dataset.extent(1));
    auto ace_res = cagra::detail::build_ace<T, IdxT>(handle, params, host_view);
    if (ace_res.dataset.has_value()) {
      return cagra::merge_result<T, IdxT>{std::move(ace_res.idx), std::move(*ace_res.dataset)};
    }
    return cagra::merge_result<T, IdxT>{std::move(ace_res.idx),
                                        raft::make_device_matrix<T, int64_t>(handle, 0, dim)};
  }
}

}  // namespace cuvs::neighbors::cagra::detail
