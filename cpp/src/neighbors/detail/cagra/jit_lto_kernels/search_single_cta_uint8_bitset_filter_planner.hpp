/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#pragma once

#include <iostream>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.h>
#include <cuvs/detail/jit_lto/MakeFragmentKey.h>

template <int N1, int N2, typename... Args>
struct SearchSingleCtaBitsetFilterPlanner : AlgorithmPlanner {
  SearchSingleCtaBitsetFilterPlanner()
    : AlgorithmPlanner(
        "search_single_cta_uint8_bitset_filter_" + std::to_string(N1) + "_" + std::to_string(N2),
        make_fragment_key<Args...>())
  {
    std::cout << "In the planner" << std::endl;
  }
};
