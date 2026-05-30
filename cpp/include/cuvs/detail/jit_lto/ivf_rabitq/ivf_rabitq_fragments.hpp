/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::ivf_rabitq::detail {

template <bool WithEx>
struct fragment_tag_compute_inner_products_with_lut {};

template <bool WithEx>
struct fragment_tag_compute_inner_products_with_lut_block_sort {};

struct fragment_tag_extract_code {};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
