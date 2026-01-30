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

// Tag types for filter subtypes
struct tag_filter_bitset_impl {};
struct tag_filter_none_impl {};

// Tag types for sample filter types with full template info
template <typename IdxTag, typename FilterImplTag>
struct tag_filter {};

// Tag types for distance metrics with full template info
template <int Veclen, typename TTag, typename AccTTag>
struct tag_metric_euclidean {};

template <int Veclen, typename TTag, typename AccTTag>
struct tag_metric_inner_product {};

// Tag types for post-processing
struct tag_post_identity {};
struct tag_post_sqrt {};
struct tag_post_compose {};

}  // namespace cuvs::neighbors::ivf_flat::detail
