/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "cuvs/neighbors/common.hpp"
#include <cstdint>

namespace cuvs::neighbors::bang {

/**
 * @defgroup bang_cpp_index_params bang index wrapper params
 * @{
 */

/**
 * @brief Hierarchy for HNSW index when converting from CAGRA index
 *
 * NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only index.
 */

}  // namespace cuvs::neighbors::bang