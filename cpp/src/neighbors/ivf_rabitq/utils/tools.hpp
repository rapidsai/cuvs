/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::ivf_rabitq::detail {

inline constexpr size_t div_rd_up(size_t x, size_t y)
{
  return (x / y) + static_cast<size_t>((x % y) != 0);
}

inline constexpr size_t rd_up_to_multiple_of(size_t x, size_t y) { return y * (div_rd_up(x, y)); }

}  // namespace cuvs::neighbors::ivf_rabitq::detail
