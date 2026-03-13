/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IVF-PQ test that runs only CosineExpanded metric parameter sets.
 */

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

using f32_f32_i64_cosine        = ivf_pq_test<float, float, int64_t>;
using f32_f32_i64_cosine_filter = ivf_pq_filter_test<float, float, int64_t>;

TEST_BUILD_HOST_INPUT_SEARCH(f32_f32_i64_cosine)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_f32_i64_cosine)
TEST_BUILD_EXTEND_SEARCH(f32_f32_i64_cosine)
TEST_BUILD_SERIALIZE_SEARCH(f32_f32_i64_cosine)
TEST_BUILD_PRECOMPUTED(f32_f32_i64_cosine)
// Only enum_variety_cosine() sets metric = CosineExpanded; defaults/small_dims/big_dims use L2.
INSTANTIATE(f32_f32_i64_cosine, enum_variety_cosine());

TEST_BUILD_SEARCH(f32_f32_i64_cosine_filter)
INSTANTIATE(f32_f32_i64_cosine_filter, enum_variety_cosine());

}  // namespace cuvs::neighbors::ivf_pq
