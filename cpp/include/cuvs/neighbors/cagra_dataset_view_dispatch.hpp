/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file cagra_dataset_view_dispatch.hpp
 * @brief Template helpers for concrete CAGRA dataset views (no variant dispatch).
 */

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/dataset_view_concepts.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>

namespace cuvs::neighbors::cagra {

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
    "logical dim %u). Use make_device_padded_dataset_view() or make_device_padded_dataset() with "
    "the same "
    "default alignment as CAGRA graph build.",
    static_cast<unsigned>(need),
    static_cast<long>(pitch),
    static_cast<unsigned>(logical_dim));
}

template <typename T, typename IdxT>
  requires is_padded_dataset_view_v<padded_dataset_view_t<T, IdxT>>
auto convert_dataset_view_to_padded_for_graph_build(padded_dataset_view_t<T, IdxT> const& view)
  -> padded_dataset_view_t<T, IdxT>
{
  expect_cagra_row_width_for_graph<T>(view.dim(), static_cast<int64_t>(view.stride()));
  return view;
}

template <typename T, typename IdxT>
  requires is_empty_dataset_view_v<empty_dataset_view_t<IdxT>>
auto convert_dataset_view_to_padded_for_graph_build(empty_dataset_view_t<IdxT> const&)
  -> padded_dataset_view_t<T, IdxT>
{
  RAFT_FAIL("cagra::build: empty dataset.");
}

template <typename T, typename IdxT, typename MathT>
  requires is_vpq_dataset_view_v<vpq_dataset_view_t<MathT, IdxT>>
auto convert_dataset_view_to_padded_for_graph_build(vpq_dataset_view_t<MathT, IdxT> const&)
  -> padded_dataset_view_t<T, IdxT>
{
  RAFT_FAIL(
    "cagra::build: VPQ-compressed dataset cannot be converted to padded dense rows for graph "
    "construction.");
}

template <typename T, typename IdxT>
auto dataset_view_to_strided_device_matrix(padded_dataset_view_t<T, IdxT> const& view)
  -> raft::device_matrix_view<const T, int64_t, raft::layout_stride>
{
  return raft::make_device_strided_matrix_view<const T, int64_t>(
    view.view().data_handle(), view.n_rows(), view.dim(), view.stride());
}

template <typename T, typename IdxT>
auto dataset_view_to_strided_device_matrix(vpq_dataset_view_t<half, IdxT> const& view)
  -> raft::device_matrix_view<const T, int64_t, raft::layout_stride>
{
  auto d = view.dim();
  return raft::make_device_strided_matrix_view<const T, int64_t>(nullptr, 0, d, d);
}

template <typename T, typename IdxT>
auto dataset_view_to_strided_device_matrix(vpq_dataset_view_t<float, IdxT> const& view)
  -> raft::device_matrix_view<const T, int64_t, raft::layout_stride>
{
  auto d = view.dim();
  return raft::make_device_strided_matrix_view<const T, int64_t>(nullptr, 0, d, d);
}

template <typename T, typename IdxT>
auto dataset_view_to_strided_device_matrix(empty_dataset_view_t<IdxT> const& view)
  -> raft::device_matrix_view<const T, int64_t, raft::layout_stride>
{
  auto d = view.dim();
  return raft::make_device_strided_matrix_view<const T, int64_t>(nullptr, 0, d, d);
}

}  // namespace cuvs::neighbors::cagra
