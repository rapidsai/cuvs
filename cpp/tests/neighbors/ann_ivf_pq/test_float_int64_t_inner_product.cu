/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IVF-PQ test that runs only Inner Product metric parameter sets.
 * Use this to verify IP-specific behavior (e.g. normalization for unnormalized data).
 */

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

using f32_f32_i64_ip        = ivf_pq_test<float, float, int64_t>;
using f32_f32_i64_ip_filter = ivf_pq_filter_test<float, float, int64_t>;

// Only build + search (no host input, extend, serialize, precomputed variants).
// inner_product_strict_recall_test(): n_lists=256, n_probes=20; fails without IP normalization.
TEST_BUILD_SEARCH(f32_f32_i64_ip)
INSTANTIATE(f32_f32_i64_ip, inner_product_strict_recall_test());

TEST_BUILD_SEARCH(f32_f32_i64_ip_filter)
INSTANTIATE(f32_f32_i64_ip_filter, inner_product_strict_recall_test());

}  // namespace cuvs::neighbors::ivf_pq
