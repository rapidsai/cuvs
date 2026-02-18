/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::cagra::detail {

// Tag types for data types
struct tag_f {};   // float
struct tag_h {};   // __half
struct tag_sc {};  // int8_t
struct tag_uc {};  // uint8_t

// Tag types for index types
struct tag_idx_ui {};  // uint32_t
struct tag_idx_l {};   // int64_t

// Tag types for distance types
struct tag_dist_f {};  // float

// Tag types for distance metrics
struct tag_metric_l2 {};
struct tag_metric_inner_product {};
struct tag_metric_cosine {};
struct tag_metric_hamming {};

// Tag types for team sizes
struct tag_team_8 {};
struct tag_team_16 {};
struct tag_team_32 {};

// Tag types for dataset block dimensions
struct tag_dim_128 {};
struct tag_dim_256 {};
struct tag_dim_512 {};

// Tag types for sample filter types
struct tag_filter_none {};
struct tag_filter_bitset {};

// Tag types for VPQ parameters
struct tag_pq_bits_8 {};
struct tag_pq_len_2 {};
struct tag_pq_len_4 {};
struct tag_codebook_half {};

}  // namespace cuvs::neighbors::cagra::detail
