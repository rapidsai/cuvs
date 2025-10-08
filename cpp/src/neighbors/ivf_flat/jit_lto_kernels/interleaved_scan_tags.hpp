/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
