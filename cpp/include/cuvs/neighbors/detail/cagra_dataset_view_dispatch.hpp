/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>

#include <memory>

namespace cuvs::neighbors::cagra::detail {

/**
 * @brief Dispatcher: select a concrete `dataset_view` and return an owned clone for
 *        `cagra::index` storage (`unique_ptr<dataset_view>`).
 *
 * Copies only the view object (metadata / pointers), not GPU vector data. Supported roots:
 * `empty_dataset_view`, `indirect_dataset_view`, `device_padded_dataset_view`,
 * `non_owning_dataset`.
 */
template <typename T, typename IdxT>
auto cagra_index_dataset_view_dispatcher(const cuvs::neighbors::dataset_view<IdxT>& root)
  -> std::unique_ptr<cuvs::neighbors::dataset_view<IdxT>>
{
  namespace nb = cuvs::neighbors;
  if (auto* p = dynamic_cast<const nb::empty_dataset_view<IdxT>*>(&root)) {
    return std::make_unique<nb::empty_dataset_view<IdxT>>(p->dim());
  }
  if (auto* p = dynamic_cast<const nb::indirect_dataset_view<IdxT>*>(&root)) {
    return std::make_unique<nb::indirect_dataset_view<IdxT>>(*p);
  }
  if (auto* p = dynamic_cast<const nb::device_padded_dataset_view<T, IdxT>*>(&root)) {
    return std::make_unique<nb::device_padded_dataset_view<T, IdxT>>(*p);
  }
  if (auto* p = dynamic_cast<const nb::non_owning_dataset<T, IdxT>*>(&root)) {
    return std::make_unique<nb::non_owning_dataset<T, IdxT>>(p->view());
  }
  RAFT_FAIL(
    "Unsupported dataset_view for CAGRA index. Use empty_dataset_view, indirect_dataset_view, "
    "device_padded_dataset_view, or non_owning_dataset.");
}

/**
 * @brief Centralized dispatch: convert a supported `dataset_view<int64_t>` to
 *        `device_padded_dataset_view` for existing graph-build code paths.
 *
 * Does not copy vector data; only builds a padded view over the same device memory. For
 * `attach_dataset_on_build`, `detail::build` still passes the original `dataset_view` to the
 * index constructor.
 */
template <typename T>
auto convert_dataset_view_to_padded_for_graph_build(
  const cuvs::neighbors::dataset_view<int64_t>& root)
  -> cuvs::neighbors::device_padded_dataset_view<T, int64_t>
{
  namespace nb = cuvs::neighbors;
  if (auto* p = dynamic_cast<const nb::device_padded_dataset_view<T, int64_t>*>(&root)) {
    return *p;
  }
  if (auto* p = dynamic_cast<const nb::non_owning_dataset<T, int64_t>*>(&root)) {
    auto sv             = p->view();
    const int64_t pitch = sv.stride(0) > 0 ? sv.stride(0) : sv.extent(1);
    auto rm =
      raft::make_device_matrix_view<const T, int64_t>(sv.data_handle(), sv.extent(0), pitch);
    return nb::device_padded_dataset_view<T, int64_t>(rm, p->dim());
  }
  if (auto* ind = dynamic_cast<const nb::indirect_dataset_view<int64_t>*>(&root)) {
    const auto* t = ind->target();
    if (auto* dp = dynamic_cast<const nb::device_padded_dataset<T, int64_t>*>(t)) {
      return dp->as_dataset_view();
    }
    if (auto* str = dynamic_cast<const nb::strided_dataset<T, int64_t>*>(t)) {
      auto sv             = str->view();
      const int64_t pitch = static_cast<int64_t>(str->stride());
      auto rm =
        raft::make_device_matrix_view<const T, int64_t>(sv.data_handle(), sv.extent(0), pitch);
      return nb::device_padded_dataset_view<T, int64_t>(rm, str->dim());
    }
    RAFT_FAIL(
      "cagra::build: indirect_dataset_view must refer to an uncompressed device dataset for graph "
      "construction.");
  }
  if (dynamic_cast<const nb::empty_dataset_view<int64_t>*>(&root) != nullptr) {
    RAFT_FAIL("cagra::build: empty dataset.");
  }
  RAFT_FAIL(
    "cagra::build: unsupported dataset_view for graph construction. Use "
    "device_padded_dataset_view, "
    "non_owning_dataset, or indirect_dataset_view to uncompressed device storage.");
}

}  // namespace cuvs::neighbors::cagra::detail
