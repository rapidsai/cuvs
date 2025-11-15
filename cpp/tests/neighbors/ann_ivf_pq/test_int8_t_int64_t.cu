/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

using f32_i08_i64        = ivf_pq_test<float, int8_t, int64_t>;
using f32_i08_i64_filter = ivf_pq_filter_test<float, int8_t, int64_t>;

TEST_BUILD_SEARCH(f32_i08_i64)
TEST_BUILD_HOST_INPUT_SEARCH(f32_i08_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_i08_i64)
TEST_BUILD_SERIALIZE_SEARCH(f32_i08_i64)
TEST_BUILD_PRECOMPUTED(f32_i08_i64)
INSTANTIATE(f32_i08_i64,
            defaults() + big_dims() + var_k() + enum_variety_l2() + enum_variety_ip() +
              enum_variety_cosine());

TEST_BUILD_SEARCH(f32_i08_i64_filter)
INSTANTIATE(f32_i08_i64_filter,
            defaults() + big_dims() + var_k() + enum_variety_l2() + enum_variety_ip() +
              enum_variety_cosine());
}  // namespace cuvs::neighbors::ivf_pq
