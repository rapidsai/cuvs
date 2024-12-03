/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

template <typename fixture>
static void set_default_cagra_params(fixture& that)
{
  that.build_params_upsm.intermediate_graph_degree = 128;
  that.build_params_upsm.graph_degree              = 64;
  that.search_params_upsm.itopk_size =
    std::clamp<int64_t>(raft::bound_by_power_of_two(that.ps.k) * 16, 128, 512);
}

TEST_P(cagra_F32, single_cta)
{
  set_default_cagra_params(*this);
  search_params_upsm.algo = cagra::search_algo::SINGLE_CTA;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(cagra_F32, multi_cta)
{
  set_default_cagra_params(*this);
  search_params_upsm.algo = cagra::search_algo::MULTI_CTA;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(cagra_F32, multi_kernel)
{
  set_default_cagra_params(*this);
  search_params_upsm.algo = cagra::search_algo::MULTI_KERNEL;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(cagra_U8, defaults)
{
  set_default_cagra_params(*this);
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching, cagra_F32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(dynamic_batching, cagra_U8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::dynamic_batching
