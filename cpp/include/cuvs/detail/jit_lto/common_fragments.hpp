/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::detail {

struct tag_filter_none {};
struct tag_filter_bitset {};

struct tag_bitset_u32 {};

struct tag_index_i64 {};

template <typename BitsetTag, typename IndexTag, typename FilterTag>
struct fragment_tag_sample_filter {};

}  // namespace cuvs::neighbors::detail
