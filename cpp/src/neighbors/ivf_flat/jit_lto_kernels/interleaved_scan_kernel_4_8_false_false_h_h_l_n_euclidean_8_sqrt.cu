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
  4,
  8,
  false,
  false,
  __half,
  __half,
  long,
  cuvs::neighbors::filtering::ivf_to_sample_filter<long,
                                                   cuvs::neighbors::filtering::none_sample_filter>,
  cuvs::neighbors::ivf_flat::detail::euclidean_dist<8, __half, __half>,
  raft::sqrt_op>(
  cuvs::neighbors::ivf_flat::detail::euclidean_dist<8, __half, __half>,
  raft::sqrt_op,
  unsigned int,
  __half const*,
  unsigned int const*,
  __half const* const*,
  unsigned int const*,
  unsigned int,
  unsigned int,
  unsigned int,
  unsigned int,
  unsigned int const*,
  unsigned int,
  cuvs::neighbors::filtering::ivf_to_sample_filter<long,
                                                   cuvs::neighbors::filtering::none_sample_filter>,
  unsigned int*,
  float*);

#else

#include "interleaved_scan_kernel_4_8_false_false_h_h_l_n_euclidean_8_sqrt.h"
#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>

__attribute__((__constructor__)) static void
register_interleaved_scan_kernel_4_8_false_false_h_h_l_n_euclidean_8_sqrt()
{
  registerAlgorithm<__half,
                    __half,
                    long,
                    cuvs::neighbors::filtering::
                      ivf_to_sample_filter<long, cuvs::neighbors::filtering::none_sample_filter>,
                    cuvs::neighbors::ivf_flat::detail::euclidean_dist<8, __half, __half>,
                    raft::sqrt_op>(
    "interleaved_scan_kernel_4_8_false_false",
    embedded_interleaved_scan_kernel_4_8_false_false_h_h_l_n_euclidean_8_sqrt,
    sizeof(embedded_interleaved_scan_kernel_4_8_false_false_h_h_l_n_euclidean_8_sqrt));
}

#endif
