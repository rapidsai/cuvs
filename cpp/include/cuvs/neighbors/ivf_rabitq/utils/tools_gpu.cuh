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

//
// Created by Stardust on 4/1/25.
//

#ifndef EXRABITQ_TOOLS_GPU_CUH
#define EXRABITQ_TOOLS_GPU_CUH

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

#endif  // EXRABITQ_TOOLS_GPU_CUH
