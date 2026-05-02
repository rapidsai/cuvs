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
 * CAGRA row width (in elements) must match `cagra_required_row_width` for the logical feature
 * dimension — same contract as `make_padded_dataset_view` (16-byte default row alignment, not
 * "round pitch to a multiple of 16 elements").
 */
template <typename T>
void expect_cagra_row_width_for_graph(uint32_t logical_dim, int64_t pitch)
{
  static constexpr uint32_t k_default_row_align_bytes = 16;
  const uint32_t need =
    cuvs::neighbors::cagra_required_row_width<T>(logical_dim, k_default_row_align_bytes);
  RAFT_EXPECTS(
    pitch == static_cast<int64_t>(need),
    "convert_dataset_view_to_padded_for_graph_build: row width in elements (pitch) must match "
    "CAGRA's required width for this element type and logical dimension (expected %u, got %ld; "
    "logical dim %u). Use make_padded_dataset_view() or make_padded_dataset() with the same "
    "default alignment as CAGRA graph build.",
    static_cast<unsigned>(need),
    static_cast<long>(pitch),
    static_cast<unsigned>(logical_dim));
}

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
 * `attach_dataset_on_build`, `build_from_device_matrix` still passes the original `dataset_view`
 * to the index constructor.
 */
template <typename T>
auto convert_dataset_view_to_padded_for_graph_build(
  const cuvs::neighbors::dataset_view<int64_t>& root)
  -> cuvs::neighbors::device_padded_dataset_view<T, int64_t>
{
  namespace nb = cuvs::neighbors;
  if (auto* p = dynamic_cast<const nb::device_padded_dataset_view<T, int64_t>*>(&root)) {
    expect_cagra_row_width_for_graph<T>(p->dim(), static_cast<int64_t>(p->stride()));
    return *p;
  }
  if (auto* p = dynamic_cast<const nb::non_owning_dataset<T, int64_t>*>(&root)) {
    auto sv             = p->view();
    const int64_t pitch = sv.stride(0) > 0 ? sv.stride(0) : sv.extent(1);
    expect_cagra_row_width_for_graph<T>(p->dim(), pitch);
    auto rm =
      raft::make_device_matrix_view<const T, int64_t>(sv.data_handle(), sv.extent(0), pitch);
    return nb::device_padded_dataset_view<T, int64_t>(rm, p->dim());
  }
  if (auto* ind = dynamic_cast<const nb::indirect_dataset_view<int64_t>*>(&root)) {
    const auto* t = ind->target();
    if (auto* dp = dynamic_cast<const nb::device_padded_dataset<T, int64_t>*>(t)) {
      expect_cagra_row_width_for_graph<T>(dp->dim(), static_cast<int64_t>(dp->stride()));
      return dp->as_dataset_view();
    }
    if (auto* str = dynamic_cast<const nb::strided_dataset<T, int64_t>*>(t)) {
      auto sv             = str->view();
      const int64_t pitch = static_cast<int64_t>(str->stride());
      expect_cagra_row_width_for_graph<T>(str->dim(), pitch);
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
