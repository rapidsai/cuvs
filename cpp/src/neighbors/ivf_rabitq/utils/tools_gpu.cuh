/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/1/25.
//

#pragma once

namespace cuvs::neighbors::ivf_rabitq::detail {

inline constexpr size_t div_rd_up_new(size_t x, size_t y)
{
  return (x / y) + static_cast<size_t>((x % y) != 0);
}

inline constexpr size_t rd_up_to_multiple_of_new(size_t x, size_t y)
{
  return y * (div_rd_up_new(x, y));
}

std::vector<cudaStream_t> create_cuda_streams(size_t num_streams);

void delete_cuda_streams(std::vector<cudaStream_t>& streams);

}  // namespace cuvs::neighbors::ivf_rabitq::detail
