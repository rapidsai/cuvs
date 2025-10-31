/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_filter.cuh"

namespace cuvs::neighbors::filtering {

template struct bitmap_filter<uint32_t, uint32_t>;
template struct bitmap_filter<const uint32_t, int64_t>;
template struct bitmap_filter<uint32_t, int64_t>;
template struct bitmap_filter<uint64_t, int64_t>;

template struct bitset_filter<uint8_t, uint32_t>;
template struct bitset_filter<uint16_t, uint32_t>;
template struct bitset_filter<uint32_t, uint32_t>;
template struct bitset_filter<uint32_t, int64_t>;
template struct bitset_filter<uint64_t, int64_t>;
}  // namespace cuvs::neighbors::filtering
