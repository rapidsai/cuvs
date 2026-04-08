/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_pq::detail {

struct tag_out_f {};
struct tag_out_h {};

struct tag_lut_f {};
struct tag_lut_h {};
struct tag_lut_fp8_signed {};
struct tag_lut_fp8_unsigned {};

struct tag_filter_none {};
struct tag_filter_bitset {};

template <typename OutTag, typename LutTag>
struct fragment_tag_compute_similarity {};

template <typename LutTag, bool EnableSMemLut, uint32_t PqBits>
struct fragment_tag_prepare_lut {};

template <typename OutTag, bool kManageLocalTopK>
struct fragment_tag_store_calculated_distances {};

template <bool PrecompBaseDiff>
struct fragment_tag_precompute_base_diff {};

template <typename LutTag, bool PrecompBaseDiff, uint32_t PqBits>
struct fragment_tag_create_lut {};

template <typename OutTag, typename LutTag, int Capacity>
struct fragment_tag_compute_distances {};

template <typename FilterTag>
struct fragment_tag_sample_filter {};

template <uint32_t PqBits>
struct fragment_tag_get_line_width {};

template <typename OutTag, typename LutTag, uint32_t PqBits>
struct fragment_tag_compute_score {};

}  // namespace cuvs::neighbors::ivf_pq::detail
