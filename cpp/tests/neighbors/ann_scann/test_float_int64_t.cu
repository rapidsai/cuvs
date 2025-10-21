/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_scann.cuh"

namespace cuvs::neighbors::experimental::scann {

using f32_i64 = scann_test<float, int64_t>;

TEST_BUILD(f32_i64)
TEST_BUILD_HOST_INPUT(f32_i64)
TEST_BUILD_HOST_INPUT_OVERLAP(f32_i64);

INSTANTIATE(f32_i64,
            defaults() + small_dims_all_pq_bits() + big_dims_all_pq_bits() + bf16() + avq() +
              soar());

}  // namespace cuvs::neighbors::experimental::scann
