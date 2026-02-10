/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../dynamic_batching.cuh"

#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::dynamic_batching {

using cagra_F32 = dynamic_batching_test<float,
                                        uint32_t,
                                        cagra::index<float, uint32_t>,
                                        cagra::build,
                                        cagra::search>;

using cagra_U8 = dynamic_batching_test<uint8_t,
                                       uint32_t,
                                       cagra::index<uint8_t, uint32_t>,
                                       cagra::build,
                                       cagra::search>;

template <typename fixture>  // NOLINT(readability-identifier-naming)
static void set_default_cagra_params(fixture& that)
{
  that.build_params_upsm.intermediate_graph_degree = 128;
  that.build_params_upsm.graph_degree              = 64;
  that.search_params_upsm.itopk_size =
    std::clamp<int64_t>(raft::bound_by_power_of_two(that.ps.k) * 16, 128, 512);
}

TEST_P(
  cagra_F32,  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
  single_cta)  // NOLINT(readability-identifier-naming,google-readability-avoid-underscore-in-googletest-name)
{
  set_default_cagra_params(*this);
  search_params_upsm.algo = cagra::search_algo::SINGLE_CTA;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(
  cagra_F32,  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
  multi_cta)  // NOLINT(readability-identifier-naming,google-readability-avoid-underscore-in-googletest-name)
{
  set_default_cagra_params(*this);
  search_params_upsm.algo = cagra::search_algo::MULTI_CTA;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(
  cagra_F32,  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
  multi_kernel)  // NOLINT(readability-identifier-naming,google-readability-avoid-underscore-in-googletest-name)
{
  set_default_cagra_params(*this);
  search_params_upsm.algo = cagra::search_algo::MULTI_KERNEL;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(
  cagra_U8,  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
  defaults)  // NOLINT(readability-identifier-naming,google-readability-avoid-underscore-in-googletest-name)
{
  set_default_cagra_params(*this);
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching,
                        cagra_F32,
                        ::testing::ValuesIn(inputs));  // NOLINT(readability-identifier-naming)
INSTANTIATE_TEST_CASE_P(dynamic_batching,
                        cagra_U8,
                        ::testing::ValuesIn(inputs));  // NOLINT(readability-identifier-naming)

}  // namespace cuvs::neighbors::dynamic_batching
