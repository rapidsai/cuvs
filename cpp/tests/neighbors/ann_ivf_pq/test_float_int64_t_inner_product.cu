/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IVF-PQ tests that run only Inner Product metric parameter sets (enum_variety_ip).
 * Same test types and fixtures as the full NEIGHBORS_ANN_IVF_PQ_TEST, but only IP cases.
 * Float, int8, and uint8 in one file. Use: ./gtests/NEIGHBORS_ANN_IVF_PQ_INNER_PRODUCT_TEST
 */

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

// float
using f32_f32_i64        = ivf_pq_test<float, float, int64_t>;
using f32_f32_i64_filter = ivf_pq_filter_test<float, float, int64_t>;

TEST_BUILD_HOST_INPUT_SEARCH(f32_f32_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_f32_i64)
TEST_BUILD_EXTEND_SEARCH(f32_f32_i64)
TEST_BUILD_SERIALIZE_SEARCH(f32_f32_i64)
INSTANTIATE(f32_f32_i64, enum_variety_ip());

TEST_BUILD_SEARCH(f32_f32_i64_filter)
INSTANTIATE(f32_f32_i64_filter, enum_variety_ip());

// int8
using f32_i08_i64        = ivf_pq_test<float, int8_t, int64_t>;
using f32_i08_i64_filter = ivf_pq_filter_test<float, int8_t, int64_t>;

TEST_BUILD_SEARCH(f32_i08_i64)
TEST_BUILD_HOST_INPUT_SEARCH(f32_i08_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_i08_i64)
TEST_BUILD_SERIALIZE_SEARCH(f32_i08_i64)
INSTANTIATE(f32_i08_i64, enum_variety_ip());

TEST_BUILD_SEARCH(f32_i08_i64_filter)
INSTANTIATE(f32_i08_i64_filter, enum_variety_ip());

// uint8
using f32_u08_i64        = ivf_pq_test<float, uint8_t, int64_t>;
using f32_u08_i64_filter = ivf_pq_filter_test<float, uint8_t, int64_t>;

TEST_BUILD_SEARCH(f32_u08_i64)
TEST_BUILD_HOST_INPUT_SEARCH(f32_u08_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_u08_i64)
TEST_BUILD_EXTEND_SEARCH(f32_u08_i64)
INSTANTIATE(f32_u08_i64, enum_variety_ip());

TEST_BUILD_SEARCH(f32_u08_i64_filter)
INSTANTIATE(f32_u08_i64_filter, enum_variety_ip());

}  // namespace cuvs::neighbors::ivf_pq
