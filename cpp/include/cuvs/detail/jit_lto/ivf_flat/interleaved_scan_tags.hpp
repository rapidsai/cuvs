/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::ivf_flat::detail {

// Tag types for data types
struct tag_f {};
struct tag_h {};
struct tag_sc {};
struct tag_uc {};

// Tag types for accumulator types
struct tag_acc_f {};
struct tag_acc_h {};
struct tag_acc_i {};
struct tag_acc_ui {};

// Tag types for index types
struct tag_idx_l {};

template <typename T>
struct tag_abbrev;
template <>
struct tag_abbrev<tag_f> {
  static constexpr char const* value = "f";
};
template <>
struct tag_abbrev<tag_h> {
  static constexpr char const* value = "h";
};
template <>
struct tag_abbrev<tag_sc> {
  static constexpr char const* value = "sc";
};
template <>
struct tag_abbrev<tag_uc> {
  static constexpr char const* value = "uc";
};
template <>
struct tag_abbrev<tag_acc_f> {
  static constexpr char const* value = "f";
};
template <>
struct tag_abbrev<tag_acc_h> {
  static constexpr char const* value = "h";
};
template <>
struct tag_abbrev<tag_acc_i> {
  static constexpr char const* value = "i";
};
template <>
struct tag_abbrev<tag_acc_ui> {
  static constexpr char const* value = "ui";
};
template <>
struct tag_abbrev<tag_idx_l> {
  static constexpr char const* value = "l";
};

// Tag types for filter subtypes
struct tag_filter_bitset_impl {};
struct tag_filter_none_impl {};

// Tag types for sample filter types with full template info
template <typename IdxTag, typename FilterImplTag>
struct tag_filter {};

// Tag types for distance metrics with full template info
struct tag_metric_euclidean {};
struct tag_metric_inner_product {};

template <int Veclen, typename TTag, typename AccTTag>
struct tag_metric_custom_udf {};

template <typename>
inline constexpr bool is_tag_metric_custom_udf_v = false;
template <int Veclen, typename TTag, typename AccTTag>
inline constexpr bool is_tag_metric_custom_udf_v<tag_metric_custom_udf<Veclen, TTag, AccTTag>> =
  true;

// Tag types for post-processing
struct tag_post_identity {};
struct tag_post_sqrt {};
struct tag_post_compose {};

}  // namespace cuvs::neighbors::ivf_flat::detail
