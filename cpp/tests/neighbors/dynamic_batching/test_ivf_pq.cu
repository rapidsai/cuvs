/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../dynamic_batching.cuh"

#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::dynamic_batching {

using ivf_pq_f16 =
  dynamic_batching_test<half, int64_t, ivf_pq::index<int64_t>, ivf_pq::build, ivf_pq::search>;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST_P(ivf_pq_f16, defaults)  // NOLINT(readability-identifier-naming)
{
  build_params_upsm.n_lists = std::round(std::sqrt(ps.n_rows));
  search_params_upsm.n_probes =
    std::max<uint32_t>(std::min<uint32_t>(build_params_upsm.n_lists, 10),
                       raft::div_rounding_up_safe<uint32_t>(build_params_upsm.n_lists, 50));
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching,
                        ivf_pq_f16,
                        ::testing::ValuesIn(inputs));  // NOLINT(readability-identifier-naming)

}  // namespace cuvs::neighbors::dynamic_batching
