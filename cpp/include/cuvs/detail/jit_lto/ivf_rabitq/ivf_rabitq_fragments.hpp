/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::ivf_rabitq::detail {

template <bool WithEx>
struct fragment_tag_compute_inner_products_with_lut {};

template <bool WithEx>
struct fragment_tag_compute_inner_products_with_lut_block_sort {};

template <bool WithEx>
struct fragment_tag_compute_inner_products_with_lut16_opt {};

template <bool WithEx>
struct fragment_tag_compute_inner_products_with_lut16_opt_block_sort {};

template <bool WithEx>
struct fragment_tag_compute_inner_products_with_bitwise {};

template <bool WithEx>
struct fragment_tag_compute_inner_products_with_bitwise_block_sort {};

template <int EX_BITS>
struct fragment_tag_extract_code {};

template <int EX_BITS>
struct fragment_tag_compute_ip2_from_long_codes_warp {};

struct tag_lut_dtype_f32 {};
struct tag_lut_dtype_f16 {};

template <typename LutDtypeTag>
struct fragment_tag_compute_lut_ip_for_vec {};

template <int NumBits>
struct fragment_tag_compute_bitwise_quantized_ip_for_vec {};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
