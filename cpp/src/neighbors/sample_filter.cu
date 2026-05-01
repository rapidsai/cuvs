/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_filter.cuh"
#include <cuvs/core/export.hpp>

namespace cuvs::neighbors::filtering {

template struct CUVS_EXPORT bitmap_filter<uint32_t, uint32_t>;
template struct CUVS_EXPORT bitmap_filter<const uint32_t, int64_t>;
template struct CUVS_EXPORT bitmap_filter<uint32_t, int64_t>;
template struct CUVS_EXPORT bitmap_filter<uint64_t, int64_t>;

template struct CUVS_EXPORT bitset_filter<uint8_t, uint32_t>;
template struct CUVS_EXPORT bitset_filter<uint16_t, uint32_t>;
template struct CUVS_EXPORT bitset_filter<uint32_t, uint32_t>;
template struct CUVS_EXPORT bitset_filter<uint32_t, int64_t>;
template struct CUVS_EXPORT bitset_filter<uint64_t, int64_t>;
}  // namespace cuvs::neighbors::filtering
