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

enum class dataset_view_kind {
  // TODO(removal): Remove `unknown` once all deprecated host_matrix_view / device_matrix_view /
  // mdspan overloads are deleted. It exists solely so that overload resolution on the deprecated
  // build(host_matrix_view) / build(device_matrix_view) shims does not cause a hard error when
  // the compiler evaluates is_host/device_dataset_view_v for a plain mdspan type.
  unknown,
  device_empty,
  host_empty,
  device_padded,
  host_padded,
  device_vpq_f16,
  host_vpq_f16,
  device_vpq_f32,
  host_vpq_f32,
};

/** Primary template returns `unknown` so traits safely return `false` for non-dataset-view types.
 */
template <typename V>
struct dataset_view_kind_of {
  static constexpr dataset_view_kind value = dataset_view_kind::unknown;
};

template <typename IdxT>
struct dataset_view_kind_of<device_empty_dataset_view<IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::device_empty;
};

template <typename IdxT>
struct dataset_view_kind_of<host_empty_dataset_view<IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::host_empty;
};

template <typename DataT, typename IdxT>
struct dataset_view_kind_of<device_padded_dataset_view<DataT, IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::device_padded;
};

template <typename DataT, typename IdxT>
struct dataset_view_kind_of<host_padded_dataset_view<DataT, IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::host_padded;
};

template <typename IdxT>
struct dataset_view_kind_of<device_vpq_dataset_view<half, IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::device_vpq_f16;
};

template <typename IdxT>
struct dataset_view_kind_of<device_vpq_dataset_view<float, IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::device_vpq_f32;
};

template <typename IdxT>
struct dataset_view_kind_of<host_vpq_dataset_view<half, IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::host_vpq_f16;
};

template <typename IdxT>
struct dataset_view_kind_of<host_vpq_dataset_view<float, IdxT>> {
  static constexpr dataset_view_kind value = dataset_view_kind::host_vpq_f32;
};

template <typename V>
using dataset_view_type_t = std::remove_cvref_t<V>;

template <typename V>
inline constexpr dataset_view_kind dataset_view_kind_v =
  dataset_view_kind_of<dataset_view_type_t<V>>::value;

template <typename V>
inline constexpr bool is_device_empty_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::device_empty;

template <typename V>
inline constexpr bool is_host_empty_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::host_empty;

/** True for any empty dataset view (device or host). */
template <typename V>
inline constexpr bool is_empty_dataset_view_v =
  is_device_empty_dataset_view_v<V> || is_host_empty_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_padded_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::device_padded;

template <typename V>
inline constexpr bool is_host_padded_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::host_padded;

/** True for either `device_padded_dataset_view` or `host_padded_dataset_view`. */
template <typename V>
inline constexpr bool is_padded_dataset_view_v =
  is_device_padded_dataset_view_v<V> || is_host_padded_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_f16_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::device_vpq_f16;

template <typename V>
inline constexpr bool is_host_vpq_f16_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::host_vpq_f16;

template <typename V>
inline constexpr bool is_vpq_f16_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_host_vpq_f16_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_f32_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::device_vpq_f32;

template <typename V>
inline constexpr bool is_host_vpq_f32_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::host_vpq_f32;

template <typename V>
inline constexpr bool is_vpq_f32_dataset_view_v =
  is_device_vpq_f32_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_device_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_dataset_view_v =
  is_host_vpq_f16_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_vpq_dataset_view_v =
  is_device_vpq_dataset_view_v<V> || is_host_vpq_dataset_view_v<V>;

/** True for any device-resident dataset view. */
template <typename V>
inline constexpr bool is_device_dataset_view_v =
  is_device_empty_dataset_view_v<V> || is_device_padded_dataset_view_v<V> ||
  is_device_vpq_dataset_view_v<V>;

/** True for any host-resident dataset view. */
template <typename V>
inline constexpr bool is_host_dataset_view_v =
  is_host_empty_dataset_view_v<V> || is_host_padded_dataset_view_v<V> ||
  is_host_vpq_dataset_view_v<V>;

/**
 * True when a host view `H` and device view `D` represent the same storage kind and differ
 * only in residency (host vs. device). Used to constrain `attach_device_dataset_on_host_index`.
 */
template <typename HostViewT, typename DeviceViewT>
inline constexpr bool compatible_host_device_dataset_views_v =
  (is_host_padded_dataset_view_v<HostViewT> && is_device_padded_dataset_view_v<DeviceViewT>) ||
  (is_host_vpq_f16_dataset_view_v<HostViewT> && is_device_vpq_f16_dataset_view_v<DeviceViewT>) ||
  (is_host_vpq_f32_dataset_view_v<HostViewT> && is_device_vpq_f32_dataset_view_v<DeviceViewT>) ||
  (is_host_empty_dataset_view_v<HostViewT> && is_device_empty_dataset_view_v<DeviceViewT>);

/** Maps a host dataset view type to its device-resident counterpart. */
template <typename HostViewT>
struct device_counterpart;

template <typename DataT, typename IdxT>
struct device_counterpart<host_padded_dataset_view<DataT, IdxT>> {
  using type = device_padded_dataset_view<DataT, IdxT>;
};

template <typename MathT, typename IdxT>
struct device_counterpart<host_vpq_dataset_view<MathT, IdxT>> {
  using type = device_vpq_dataset_view<MathT, IdxT>;
};

template <typename IdxT>
struct device_counterpart<host_empty_dataset_view<IdxT>> {
  using type = device_empty_dataset_view<IdxT>;
};

template <typename HostViewT>
using device_counterpart_t = typename device_counterpart<dataset_view_type_t<HostViewT>>::type;

/** Element type `T` for `cagra::build(res, params, dataset_view)` (deduced, not a template arg). */
template <typename V, typename = void>
struct cagra_view_element_type;

template <typename DataT, typename IdxT>
struct cagra_view_element_type<device_padded_dataset_view<DataT, IdxT>> {
  using type = DataT;
};

template <typename DataT, typename IdxT>
struct cagra_view_element_type<host_padded_dataset_view<DataT, IdxT>> {
  using type = DataT;
};

template <typename MathT, typename IdxT>
struct cagra_view_element_type<device_vpq_dataset_view<MathT, IdxT>> {
  using type = MathT;
};

template <typename V>
using cagra_view_element_type_t = typename cagra_view_element_type<dataset_view_type_t<V>>::type;

}  // namespace cuvs::neighbors
