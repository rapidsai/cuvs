/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/cagra.hpp>

#include "cagra_build.cuh"

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

#include <vector>

namespace cuvs::neighbors::cagra::detail {

template <class T, class IdxT>
merged_dataset compute_merged_dataset_layout(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> const& indices,
  cuvs::neighbors::filtering::base_filter const& row_filter)
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
    using VT       = cuvs::neighbors::any_dataset_view_types<T, ds_idx_type>;
    auto const& va = index->data().as_variant();
    if (std::holds_alternative<typename VT::strided_view>(va)) {
      auto const& v = std::get<typename VT::strided_view>(va);
      if (dim == 0) {
        dim    = index->dim();
        stride = static_cast<int64_t>(v.stride());
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
        RAFT_EXPECTS(stride == static_cast<int64_t>(v.stride()),
                     "Row stride of datasets in indices must be equal.");
      }
      new_dataset_size += index->size();
    } else if (std::holds_alternative<typename VT::padded_view>(va)) {
      auto const& v = std::get<typename VT::padded_view>(va);
      if (dim == 0) {
        dim    = index->dim();
        stride = static_cast<int64_t>(v.stride());
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
        RAFT_EXPECTS(stride == static_cast<int64_t>(v.stride()),
                     "Row stride of datasets in indices must be equal.");
      }
      new_dataset_size += index->size();
    } else if (std::holds_alternative<typename VT::empty_view>(va)) {
      RAFT_FAIL(
        "cagra::merge only supports an index to which the dataset is attached. Please check if the "
        "index was built with index_param.attach_dataset_on_build = true, or if a dataset was "
        "attached after the build.");
    } else {
      RAFT_FAIL("cagra::merge only supports an uncompressed dataset index");
    }
  }

  merged_dataset layout{};
  layout.merged_rows     = static_cast<int64_t>(new_dataset_size);
  layout.stride_elements = stride;
  layout.dim             = static_cast<uint32_t>(dim);
  layout.bitset_filtered =
    (row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset);
  if (layout.bitset_filtered) {
    auto const& actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(row_filter);
    layout.filtered_rows = actual_filter.view().count(handle);
  } else {
    layout.filtered_rows = layout.merged_rows;
  }
  return layout;
}

template <class T, class IdxT>
cuvs::neighbors::cagra::index<T, IdxT> merge(
  raft::resources const& handle,
  const cagra::index_params& params,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
  merged_dataset_storage<T, IdxT>& storage,
  const cuvs::neighbors::filtering::base_filter& row_filter)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type   = typename cagra_index_t::dataset_index_type;

  auto const expected = compute_merged_dataset_layout<T, IdxT>(handle, indices, row_filter);
  RAFT_EXPECTS(expected.merged_rows == storage.layout.merged_rows &&
                 expected.filtered_rows == storage.layout.filtered_rows &&
                 expected.stride_elements == storage.layout.stride_elements &&
                 expected.dim == storage.layout.dim &&
                 expected.bitset_filtered == storage.layout.bitset_filtered,
               "merged_dataset_storage.layout does not match indices and row_filter (use the same "
               "arguments as "
               "make_merged_dataset).");

  auto merged_storage = storage.merged_storage.view();
  RAFT_EXPECTS(merged_storage.extent(0) == storage.layout.merged_rows,
               "merged_storage rows (%ld) must equal layout.merged_rows (%ld)",
               long(merged_storage.extent(0)),
               long(storage.layout.merged_rows));
  RAFT_EXPECTS(merged_storage.extent(1) == storage.layout.stride_elements,
               "merged_storage stride (%ld) must equal layout.stride_elements (%ld)",
               long(merged_storage.extent(1)),
               long(storage.layout.stride_elements));

  std::optional<raft::device_matrix_view<T, int64_t, raft::row_major>> filtered_view{};
  if (storage.layout.bitset_filtered) {
    RAFT_EXPECTS(storage.filtered_storage.has_value(),
                 "Bitset-filtered merge requires merged_dataset_storage.filtered_storage.");
    filtered_view = storage.filtered_storage->view();
    RAFT_EXPECTS(filtered_view->extent(0) == storage.layout.filtered_rows,
                 "filtered_storage rows (%ld) must equal layout.filtered_rows (%ld)",
                 long(filtered_view->extent(0)),
                 long(storage.layout.filtered_rows));
    RAFT_EXPECTS(filtered_view->extent(1) == storage.layout.stride_elements,
                 "filtered_storage stride (%ld) must equal layout.stride_elements (%ld)",
                 long(filtered_view->extent(1)),
                 long(storage.layout.stride_elements));
  } else {
    RAFT_EXPECTS(!storage.filtered_storage.has_value(),
                 "Non-bitset merge requires merged_dataset_storage.filtered_storage be unset.");
  }

  auto merge_dataset = [&](T* dst, std::size_t dst_ld) {
    IdxT row_offset = 0;
    for (cagra_index_t* index : indices) {
      const T* src_ptr   = nullptr;
      std::size_t n_rows = 0;
      using VTm          = cuvs::neighbors::any_dataset_view_types<T, ds_idx_type>;
      auto const& vam    = index->data().as_variant();
      if (std::holds_alternative<typename VTm::strided_view>(vam)) {
        auto const& v = std::get<typename VTm::strided_view>(vam);
        src_ptr       = v.view().data_handle();
        n_rows        = static_cast<std::size_t>(v.n_rows());
      } else if (std::holds_alternative<typename VTm::padded_view>(vam)) {
        auto const& v = std::get<typename VTm::padded_view>(vam);
        src_ptr       = v.view().data_handle();
        n_rows        = static_cast<std::size_t>(v.n_rows());
      } else {
        RAFT_FAIL("cagra::merge: unexpected dataset type while copying rows");
      }
      raft::copy_matrix(dst + static_cast<std::size_t>(row_offset) * dst_ld,
                        dst_ld,
                        src_ptr,
                        static_cast<std::size_t>(storage.layout.stride_elements),
                        static_cast<std::size_t>(storage.layout.dim),
                        n_rows,
                        raft::resource::get_cuda_stream(handle));

      row_offset += IdxT(index->data().n_rows());
    }
  };

  cudaStream_t stream     = raft::resource::get_cuda_stream(handle);
  const auto merged_bytes = static_cast<std::size_t>(merged_storage.size()) * sizeof(T);
  RAFT_CUDA_TRY(cudaMemsetAsync(merged_storage.data_handle(), 0, merged_bytes, stream));

  merge_dataset(merged_storage.data_handle(),
                static_cast<std::size_t>(storage.layout.stride_elements));

  if (storage.layout.bitset_filtered) {
    auto actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(row_filter);

    auto indices_csr = raft::make_device_csr_matrix<uint32_t, int64_t, int64_t, int64_t>(
      handle, 1, static_cast<std::size_t>(storage.layout.merged_rows));
    indices_csr.initialize_sparsity(storage.layout.filtered_rows);

    actual_filter.view().to_csr(handle, indices_csr);

    auto csr_indices  = indices_csr.structure_view().get_indices();
    auto indices_view = raft::make_device_vector_view<const int64_t, int64_t>(
      csr_indices.data(), static_cast<int64_t>(csr_indices.size()));

    auto& filtered_storage = *filtered_view;
    RAFT_CUDA_TRY(cudaMemsetAsync(filtered_storage.data_handle(),
                                  0,
                                  static_cast<std::size_t>(filtered_storage.size()) * sizeof(T),
                                  stream));

    raft::matrix::copy_rows(
      handle, raft::make_const_mdspan(merged_storage), filtered_storage, indices_view);

    cuvs::neighbors::device_padded_dataset_view<T, int64_t> dv(
      raft::make_const_mdspan(filtered_storage), storage.layout.dim);
    auto index = ::cuvs::neighbors::cagra::detail::build_from_device_matrix<T, IdxT>(
      handle, params, cuvs::neighbors::any_dataset_view<T, int64_t>(dv));
    RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
    return index;
  }

  cuvs::neighbors::device_padded_dataset_view<T, int64_t> dv(
    raft::make_const_mdspan(merged_storage), storage.layout.dim);
  auto index = ::cuvs::neighbors::cagra::detail::build_from_device_matrix<T, IdxT>(
    handle, params, cuvs::neighbors::any_dataset_view<T, int64_t>(dv));
  RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
  return index;
}

}  // namespace cuvs::neighbors::cagra::detail

namespace cuvs::neighbors::cagra {

template <class T, class IdxT>
merged_dataset_storage<T, IdxT> make_merged_dataset(
  raft::resources const& res,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> const& indices,
  cuvs::neighbors::filtering::base_filter const& row_filter)
{
  merged_dataset layout = detail::compute_merged_dataset_layout<T, IdxT>(res, indices, row_filter);
  auto merged_storage =
    raft::make_device_matrix<T, int64_t>(res, layout.merged_rows, layout.stride_elements);
  std::optional<raft::device_matrix<T, int64_t, raft::row_major>> filtered_storage;
  if (layout.bitset_filtered) {
    filtered_storage.emplace(
      raft::make_device_matrix<T, int64_t>(res, layout.filtered_rows, layout.stride_elements));
  }
  return {layout, std::move(merged_storage), std::move(filtered_storage)};
}

template <class T, class IdxT>
void adopt_merge_storage_into_index_deprecated(index<T, IdxT>& idx,
                                               merged_dataset_storage<T, IdxT>&& storage)
{
  if (storage.layout.bitset_filtered) {
    RAFT_EXPECTS(storage.filtered_storage.has_value(),
                 "adopt_merge_storage_into_index_deprecated: missing filtered_storage.");
    auto padded = std::make_unique<device_padded_dataset<T, int64_t>>(
      std::move(*storage.filtered_storage), storage.layout.dim);
    storage.filtered_storage.reset();
    adopt_owning_padded_dataset_into_index(idx, std::move(padded));
  } else {
    auto padded = std::make_unique<device_padded_dataset<T, int64_t>>(
      std::move(storage.merged_storage), storage.layout.dim);
    adopt_owning_padded_dataset_into_index(idx, std::move(padded));
  }
}

/** @brief Implementation of deprecated public `merge` (4-arg, owning dataset on index). */
template <class T, class IdxT>
index<T, IdxT> merge_owning_deprecated(raft::resources const& handle,
                                       const index_params& params,
                                       std::vector<index<T, IdxT>*>& indices,
                                       const filtering::base_filter& row_filter)
{
  auto storage = make_merged_dataset<T, IdxT>(handle, indices, row_filter);
  auto merged  = detail::merge<T, IdxT>(handle, params, indices, storage, row_filter);
  adopt_merge_storage_into_index_deprecated(merged, std::move(storage));
  return merged;
}

}  // namespace cuvs::neighbors::cagra
