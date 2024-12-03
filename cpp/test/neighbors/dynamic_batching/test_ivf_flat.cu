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

#include <cuvs/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors::dynamic_batching {

using ivf_flat_i8 = dynamic_batching_test<uint8_t,
                                          int64_t,
                                          ivf_flat::index<uint8_t, int64_t>,
                                          ivf_flat::build,
                                          ivf_flat::search>;

TEST_P(ivf_flat_i8, defaults)
{
  build_params_upsm.n_lists = std::round(std::sqrt(ps.n_rows));
  search_params_upsm.n_probes =
    std::max<uint32_t>(std::min<uint32_t>(build_params_upsm.n_lists, 10),
                       raft::div_rounding_up_safe<uint32_t>(build_params_upsm.n_lists, 50));
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching, ivf_flat_i8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::dynamic_batching
