/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>

#include <memory>
#include <type_traits>

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
 * @brief Store a heap copy of CAGRA's dataset view handle (variant copy; same device pointers).
 */
template <typename T, typename IdxT>
auto clone_any_dataset_view_for_cagra_index(any_dataset_view<T, IdxT> const& root)
  -> std::unique_ptr<any_dataset_view<T, IdxT>>
{
  return std::make_unique<any_dataset_view<T, IdxT>>(root);
}

/**
 * @brief Map `any_owning_dataset` storage to `any_dataset_view<T, IdxT>` for CAGRA index
 *        `update_dataset` (element type \p T must match the owning variant member).
 */
template <typename T, typename IdxT>
auto any_owning_dataset_to_index_view(any_owning_dataset<IdxT>& owner) -> any_dataset_view<T, IdxT>
{
  namespace nb = cuvs::neighbors;
  using OT     = nb::any_owning_dataset_types<IdxT>;
  auto& store  = owner.as_variant();

  if (std::holds_alternative<typename OT::empty_owning>(store)) {
    auto const& e = std::get<typename OT::empty_owning>(store);
    return any_dataset_view<T, IdxT>(
      typename nb::any_dataset_view_types<T, IdxT>::empty_view(e.dim()));
  }

  if constexpr (std::is_same_v<T, float>) {
    if (std::holds_alternative<typename OT::padded_f32_owning>(store)) {
      return any_dataset_view<T, IdxT>(
        std::get<typename OT::padded_f32_owning>(store).as_dataset_view());
    }
    if (std::holds_alternative<typename OT::strided_f32_owning>(store)) {
      return any_dataset_view<T, IdxT>(
        nb::strided_dataset_view<T, IdxT>(std::get<typename OT::strided_f32_owning>(store).view()));
    }
    if (std::holds_alternative<typename OT::vpq_f32_owning>(store)) {
      auto& vpq = std::get<typename OT::vpq_f32_owning>(store);
      return any_dataset_view<T, IdxT>(nb::make_indirect_dataset_view(&vpq));
    }
  } else if constexpr (std::is_same_v<T, half>) {
    if (std::holds_alternative<typename OT::padded_f16_owning>(store)) {
      return any_dataset_view<T, IdxT>(
        std::get<typename OT::padded_f16_owning>(store).as_dataset_view());
    }
    if (std::holds_alternative<typename OT::strided_f16_owning>(store)) {
      return any_dataset_view<T, IdxT>(
        nb::strided_dataset_view<T, IdxT>(std::get<typename OT::strided_f16_owning>(store).view()));
    }
    if (std::holds_alternative<typename OT::vpq_f16_owning>(store)) {
      auto& vpq = std::get<typename OT::vpq_f16_owning>(store);
      return any_dataset_view<T, IdxT>(nb::make_indirect_dataset_view(&vpq));
    }
  } else if constexpr (std::is_same_v<T, int8_t>) {
    if (std::holds_alternative<typename OT::padded_i8_owning>(store)) {
      return any_dataset_view<T, IdxT>(
        std::get<typename OT::padded_i8_owning>(store).as_dataset_view());
    }
    if (std::holds_alternative<typename OT::strided_i8_owning>(store)) {
      return any_dataset_view<T, IdxT>(
        nb::strided_dataset_view<T, IdxT>(std::get<typename OT::strided_i8_owning>(store).view()));
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    if (std::holds_alternative<typename OT::padded_u8_owning>(store)) {
      return any_dataset_view<T, IdxT>(
        std::get<typename OT::padded_u8_owning>(store).as_dataset_view());
    }
    if (std::holds_alternative<typename OT::strided_u8_owning>(store)) {
      return any_dataset_view<T, IdxT>(
        nb::strided_dataset_view<T, IdxT>(std::get<typename OT::strided_u8_owning>(store).view()));
    }
  } else {
    RAFT_FAIL(
      "cagra::index: any_owning_dataset_to_index_view: unsupported index element type T (expected "
      "float, half, int8_t, or uint8_t).");
  }

  RAFT_FAIL(
    "cagra::index: any_owning_dataset variant does not match index element type T, or unsupported "
    "alternative.");
}

/**
 * @brief Dispatch on `any_dataset_view` alternatives and produce `device_padded_dataset_view` for
 *        graph-build paths.
 */
template <typename T>
auto convert_dataset_view_to_padded_for_graph_build(any_dataset_view<T, int64_t> const& root)
  -> cuvs::neighbors::device_padded_dataset_view<T, int64_t>
{
  namespace nb   = cuvs::neighbors;
  using VT       = nb::any_dataset_view_types<T, int64_t>;
  auto const& va = root.as_variant();
  if (std::holds_alternative<typename VT::empty_view>(va)) {
    RAFT_FAIL("cagra::build: empty dataset.");
  }
  if (std::holds_alternative<typename VT::indirect_view>(va)) {
    auto const& v = std::get<typename VT::indirect_view>(va);
    RAFT_EXPECTS(
      v.get_indirect_target_type() == nb::indirect_padded_type_for_element<T>(),
      "cagra::build: indirect_dataset_view target must be device padded storage matching index "
      "element type T for graph construction.");
    auto* dp = static_cast<nb::padded_dataset<T, int64_t> const*>(v.raw_target());
    expect_cagra_row_width_for_graph<T>(dp->dim(), static_cast<int64_t>(dp->stride()));
    return dp->as_dataset_view();
  }
  if (std::holds_alternative<typename VT::padded_view>(va)) {
    auto const& v = std::get<typename VT::padded_view>(va);
    expect_cagra_row_width_for_graph<T>(v.dim(), static_cast<int64_t>(v.stride()));
    return v;
  }
  if (std::holds_alternative<typename VT::strided_view>(va)) {
    auto const& v       = std::get<typename VT::strided_view>(va);
    auto sv             = v.view();
    const int64_t pitch = sv.stride(0) > 0 ? sv.stride(0) : sv.extent(1);
    expect_cagra_row_width_for_graph<T>(v.dim(), pitch);
    auto rm =
      raft::make_device_matrix_view<const T, int64_t>(sv.data_handle(), sv.extent(0), pitch);
    return nb::device_padded_dataset_view<T, int64_t>(rm, v.dim());
  }
  RAFT_FAIL("cagra::build: unsupported dataset view for graph construction.");
}

}  // namespace cuvs::neighbors::cagra::detail
