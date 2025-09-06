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

#include "../search_single_cta_inst.cuh"
#include "kernel_tags.hpp"

#ifdef BUILD_KERNEL

template __global__ void cuvs::neighbors::cagra::detail::single_cta_search::search_kernel<
  64u,
  64u,
  1u,
  cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<unsigned char, unsigned int, float>,
  cuvs::neighbors::cagra::detail::CagraSampleFilterWithQueryIdOffset<
    cuvs::neighbors::filtering::bitset_filter<unsigned int, long>>>(
  unsigned long,
  cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<unsigned char, unsigned int, float>::
    DISTANCE_T*,
  unsigned int,
  cuvs::neighbors::cagra::detail::
    dataset_descriptor_base_t<unsigned char, unsigned int, float> const*,
  cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<unsigned char, unsigned int, float>::
    DATA_T const*,
  cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<unsigned char, unsigned int, float>::
    INDEX_T const*,
  unsigned int,
  unsigned int,
  unsigned long,
  cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<unsigned char, unsigned int, float>::
    INDEX_T const*,
  unsigned int,
  cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<unsigned char, unsigned int, float>::
    INDEX_T*,
  unsigned int,
  unsigned int,
  unsigned int,
  unsigned int,
  unsigned int*,
  unsigned int,
  unsigned int,
  unsigned int,
  cuvs::neighbors::cagra::detail::CagraSampleFilterWithQueryIdOffset<
    cuvs::neighbors::filtering::bitset_filter<unsigned int, long>>);

#else

#include "embedded_fatbins.h"
#include <cuvs/detail/jit_lto/RegisterKernelFragment.h>

__attribute__((__constructor__)) static void register_search_single_cta_uint8_bitset_filter_64_64()
{
  registerAlgorithm<
    cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<unsigned char, unsigned int, float>,
    _64_tag,
    _64_tag>("search_single_cta_uint8_bitset_filter_64_64",
             embedded_search_single_cta_uint8_bitset_filter_64_64,
             sizeof(embedded_search_single_cta_uint8_bitset_filter_64_64));
}

#endif
