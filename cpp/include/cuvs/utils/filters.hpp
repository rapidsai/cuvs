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

#include <cuvs/core/bitmap.hpp>
#include <cuvs/core/bitset.hpp>
#include <faiss/impl/IDSelector.h>

namespace cuvs::utils {

/**
 * @brief Convert a Faiss IDSelectorRange to a cuvs::core::bitset_view
 *
 * @param selector The Faiss IDSelectorRange to convert
 * @param bitset The cuvs::core::bitset_view to store the result
 */
void convert_to_bitset(raft::resources const& res,
                       const faiss::IDSelectorRange& selector,
                       cuvs::core::bitset_view<uint32_t, uint32_t> bitset);

/**
 * @brief Convert a Faiss IDSelector to a cuvs::core::bitset_view
 *
 * @param selector The Faiss IDSelector to convert
 * @param bitset The cuvs::core::bitset_view to store the result
 */
void convert_to_bitset(raft::resources const& res,
                       const faiss::IDSelector& selector,
                       cuvs::core::bitset_view<uint32_t, uint32_t> bitset);

/**
 * @brief Convert a Faiss IDSelector to a cuvs::core::bitmap
 *
 * @param selector The Faiss IDSelector to convert
 * @param bitmap The cuvs::core::bitmap to store the result
 */
void convert_to_bitmap(raft::resources const& res,
                       const faiss::IDSelector& selector,
                       cuvs::core::bitmap_view<uint32_t, uint32_t> bitmap);

}  // namespace cuvs::utils
