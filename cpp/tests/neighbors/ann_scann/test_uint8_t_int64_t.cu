/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_scann.cuh"

#include <raft/matrix/init.cuh>

namespace cuvs::neighbors::scann {

using u8_i64 = scann_test<uint8_t, int64_t>;

TEST_BUILD(u8_i64)
TEST_BUILD_HOST_INPUT(u8_i64)
TEST_BUILD_HOST_INPUT_OVERLAP(u8_i64);

INSTANTIATE(u8_i64,
            defaults() + small_dims_all_pq_bits() + big_dims_all_pq_bits() + bf16() + bf16_avq() +
              avq() + soar() + two_level());

}  // namespace cuvs::neighbors::scann
