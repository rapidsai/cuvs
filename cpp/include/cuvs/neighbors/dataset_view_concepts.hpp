/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file dataset_view_concepts.hpp
 * @brief Compile-time contracts for CAGRA (and shared) dataset view types.
 *
 * These replace runtime `std::variant` dispatch: each `DatasetViewT` is a concrete
 * `dataset_view<Container, …>` specialization known at compile time.
 */

#include <cuvs/neighbors/common.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace cuvs::neighbors {

/** Any non-owning dataset view exposing row count and logical dimension. */
template <typename V, typename IdxT = int64_t>
concept cagra_dataset_view = requires(V const& v) {
  { v.n_rows() } -> std::convertible_to<IdxT>;
  { v.dim() } -> std::convertible_to<uint32_t>;
};

template <typename DataT, typename IdxT>
using padded_dataset_view_t = device_padded_dataset_view<DataT, IdxT>;

template <typename MathT, typename IdxT>
using vpq_dataset_view_t = vpq_dataset_view<MathT, IdxT>;

template <typename IdxT>
using empty_dataset_view_t = empty_dataset_view<IdxT>;

enum class dataset_view_kind {
  empty,
  padded,
  vpq_f16,
  vpq_f32,
};

template <typename V>
struct dataset_view_kind_of;

template <typename IdxT>
struct dataset_view_kind_of<empty_dataset_view<IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::empty;
};

template <typename DataT, typename IdxT>
struct dataset_view_kind_of<padded_dataset_view<DataT, IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::padded;
};

template <typename IdxT>
struct dataset_view_kind_of<vpq_dataset_view<half, IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::vpq_f16;
};

template <typename IdxT>
struct dataset_view_kind_of<vpq_dataset_view<float, IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::vpq_f32;
};

template <typename V>
using dataset_view_type_t = std::remove_cvref_t<V>;

template <typename V>
inline constexpr dataset_view_kind dataset_view_kind_v =
  dataset_view_kind_of<dataset_view_type_t<V>>::value;

template <typename V>
inline constexpr bool is_empty_dataset_view_v = dataset_view_kind_v<V> == dataset_view_kind::empty;

template <typename V>
inline constexpr bool is_padded_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::padded;

template <typename V>
inline constexpr bool is_vpq_f16_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f16;

template <typename V>
inline constexpr bool is_vpq_f32_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f32;

template <typename V>
inline constexpr bool is_vpq_dataset_view_v =
  is_vpq_f16_dataset_view_v<V> || is_vpq_f32_dataset_view_v<V>;

/** Element type `T` for `cagra::build(res, params, dataset_view)` (deduced, not a template arg). */
template <typename V, typename = void>
struct cagra_view_element_type;

template <typename DataT, typename IdxT>
struct cagra_view_element_type<padded_dataset_view_t<DataT, IdxT>> {
  using type = DataT;
};

template <typename MathT, typename IdxT>
struct cagra_view_element_type<vpq_dataset_view_t<MathT, IdxT>> {
  using type = MathT;
};

template <typename V>
using cagra_view_element_type_t = typename cagra_view_element_type<dataset_view_type_t<V>>::type;

}  // namespace cuvs::neighbors
