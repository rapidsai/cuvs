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

#pragma once

#include <raft/core/bitset.hpp>

namespace cuvs::core {
/* To use bitset functions containing CUDA code, include <raft/core/bitset.cuh> */

template <typename bitset_t, typename index_t>
using bitset_view = raft::core::bitset_view<bitset_t, index_t>;

template <typename bitset_t, typename index_t>
using bitset = raft::core::bitset<bitset_t, index_t>;

}  // end namespace cuvs::core
