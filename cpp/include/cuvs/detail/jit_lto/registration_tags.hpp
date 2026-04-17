/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::detail::jit_lto {

struct tag_f {};
struct tag_h {};
struct tag_sc {};
struct tag_uc {};
struct tag_idx_l {};
struct tag_filter_none {};
struct tag_filter_bitset {};

}  // namespace cuvs::detail::jit_lto

namespace cuvs::neighbors::cagra::detail {

using cuvs::detail::jit_lto::tag_f;
using cuvs::detail::jit_lto::tag_filter_bitset;
using cuvs::detail::jit_lto::tag_filter_none;
using cuvs::detail::jit_lto::tag_h;
using cuvs::detail::jit_lto::tag_idx_l;
using cuvs::detail::jit_lto::tag_sc;
using cuvs::detail::jit_lto::tag_uc;

struct tag_idx_ui {};
struct tag_dist_f {};
struct tag_metric_l2 {};
struct tag_metric_inner_product {};
struct tag_metric_cosine {};
struct tag_metric_hamming {};
struct tag_team_8 {};
struct tag_team_16 {};
struct tag_team_32 {};
struct tag_dim_128 {};
struct tag_dim_256 {};
struct tag_dim_512 {};
struct tag_pq_bits_0 {};
struct tag_pq_bits_8 {};
struct tag_pq_len_0 {};
struct tag_pq_len_2 {};
struct tag_pq_len_4 {};
struct tag_codebook_none {};
struct tag_codebook_half {};
struct tag_metric_l1 {};
struct tag_norm_noop {};
struct tag_norm_cosine {};

}  // namespace cuvs::neighbors::cagra::detail

namespace cuvs::neighbors::ivf_flat::detail {

using cuvs::detail::jit_lto::tag_f;
using cuvs::detail::jit_lto::tag_filter_bitset;
using cuvs::detail::jit_lto::tag_filter_none;
using cuvs::detail::jit_lto::tag_h;
using cuvs::detail::jit_lto::tag_idx_l;
using cuvs::detail::jit_lto::tag_sc;
using cuvs::detail::jit_lto::tag_uc;

struct tag_i8 {};
struct tag_u8 {};

struct tag_acc_f {};
struct tag_acc_h {};
struct tag_acc_i32 {};
struct tag_acc_u32 {};

struct tag_metric_euclidean {};
struct tag_metric_inner_product {};
struct tag_metric_custom_udf {};

struct tag_post_process_identity {};
struct tag_post_process_sqrt {};
struct tag_post_process_compose {};

template <typename DataTag, typename AccTag, typename IdxTag, int Capacity, bool Ascending>
struct fragment_tag_interleaved_scan {};

template <typename DataTag, typename AccTag, bool ComputeNorm, int Veclen>
struct fragment_tag_load_and_compute_dist {};

template <typename DataTag, typename AccTag, typename MetricTag, int Veclen>
struct fragment_tag_metric {};

template <typename IndexTag, typename FilterTag>
struct fragment_tag_filter {};

template <typename PostLambdaTag>
struct fragment_tag_post_lambda {};

}  // namespace cuvs::neighbors::ivf_flat::detail
