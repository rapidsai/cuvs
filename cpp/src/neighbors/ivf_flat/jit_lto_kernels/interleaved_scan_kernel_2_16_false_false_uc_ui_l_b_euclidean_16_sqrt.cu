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

#include "../ivf_flat_interleaved_scan.cuh"

#ifdef BUILD_KERNEL

template __global__ void cuvs::neighbors::ivf_flat::detail::interleaved_scan_kernel<
  2,
  16,
  false,
  false,
  unsigned char,
  unsigned int,
  long,
  cuvs::neighbors::filtering::
    ivf_to_sample_filter<long, cuvs::neighbors::filtering::bitset_filter<unsigned int, long>>,
  cuvs::neighbors::ivf_flat::detail::euclidean_dist<16, unsigned char, unsigned int>,
  raft::sqrt_op>(cuvs::neighbors::ivf_flat::detail::euclidean_dist<16, unsigned char, unsigned int>,
                 raft::sqrt_op,
                 unsigned int,
                 unsigned char const*,
                 unsigned int const*,
                 unsigned char const* const*,
                 unsigned int const*,
                 unsigned int,
                 unsigned int,
                 unsigned int,
                 unsigned int,
                 unsigned int const*,
                 unsigned int,
                 cuvs::neighbors::filtering::ivf_to_sample_filter<
                   long,
                   cuvs::neighbors::filtering::bitset_filter<unsigned int, long>>,
                 unsigned int*,
                 float*);

#else

#include "interleaved_scan_kernel_2_16_false_false_uc_ui_l_b_euclidean_16_sqrt.h"
#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>

__attribute__((__constructor__)) static void
register_interleaved_scan_kernel_2_16_false_false_uc_ui_l_b_euclidean_16_sqrt()
{
  registerAlgorithm<
    unsigned char,
    unsigned int,
    long,
    cuvs::neighbors::filtering::
      ivf_to_sample_filter<long, cuvs::neighbors::filtering::bitset_filter<unsigned int, long>>,
    cuvs::neighbors::ivf_flat::detail::euclidean_dist<16, unsigned char, unsigned int>,
    raft::sqrt_op>(
    "interleaved_scan_kernel_2_16_false_false",
    embedded_interleaved_scan_kernel_2_16_false_false_uc_ui_l_b_euclidean_16_sqrt,
    sizeof(embedded_interleaved_scan_kernel_2_16_false_false_uc_ui_l_b_euclidean_16_sqrt));
}

#endif
