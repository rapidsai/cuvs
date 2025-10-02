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
  1,
  16,
  false,
  false,
  signed char,
  int,
  long,
  cuvs::neighbors::filtering::ivf_to_sample_filter<long,
                                                   cuvs::neighbors::filtering::none_sample_filter>,
  cuvs::neighbors::ivf_flat::detail::euclidean_dist<16, signed char, int>,
  raft::identity_op>(
  cuvs::neighbors::ivf_flat::detail::euclidean_dist<16, signed char, int>,
  raft::identity_op,
  unsigned int,
  signed char const*,
  unsigned int const*,
  signed char const* const*,
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

#include "interleaved_scan_kernel_1_16_false_false_sc_i_l_n_euclidean_16_id.h"
#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>

__attribute__((__constructor__)) static void
register_interleaved_scan_kernel_1_16_false_false_sc_i_l_n_euclidean_16_id()
{
  registerAlgorithm<signed char,
                    int,
                    long,
                    cuvs::neighbors::filtering::
                      ivf_to_sample_filter<long, cuvs::neighbors::filtering::none_sample_filter>,
                    cuvs::neighbors::ivf_flat::detail::euclidean_dist<16, signed char, int>,
                    raft::identity_op>(
    "interleaved_scan_kernel_1_16_false_false",
    embedded_interleaved_scan_kernel_1_16_false_false_sc_i_l_n_euclidean_16_id,
    sizeof(embedded_interleaved_scan_kernel_1_16_false_false_sc_i_l_n_euclidean_16_id));
}

#endif
