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

#include <cuvs/core/bitmap.hpp>
#include <cuvs/core/bitset.hpp>
#include <faiss/impl/IDSelector.h>
#include <omp.h>
#include <raft/core/bitset.cuh>
#include <raft/core/copy.cuh>
#include <raft/core/host_mdarray.hpp>

namespace cuvs::util {

/**
 * @brief CUDA kernel to set a range of bits in a bitset to true
 *
 * @param bitset_data Pointer to the bitset data
 * @param imin Starting index
 * @param imax Ending index
 * @param n_elements_to_set Number of elements to set
 */
template <typename bitset_t>
RAFT_KERNEL set_range_kernel(bitset_t* bitset_data,
                             uint32_t imin,
                             uint32_t imax,
                             uint32_t n_elements_to_set)
{
  uint32_t idx         = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t nbits = sizeof(bitset_t) * 8;

  uint32_t current_index = (imin / nbits) + idx;
  bitset_t mask          = 0;
  if (idx < n_elements_to_set) {
    if (n_elements_to_set == 1) {
      // Special case: range is within a single element
      int bit_offset = imin % nbits;
      mask           = (bitset_t{1} << bit_offset) - 1;
      bit_offset     = imax % nbits;
      mask           = mask ^ ((bitset_t{1} << bit_offset) - 1);
    } else if (idx == 0) {
      // First element: set bits from imin to end
      int bit_offset = imin % nbits;
      mask           = ~((bitset_t{1} << bit_offset) - 1);
    } else if (idx == n_elements_to_set - 1) {
      // Last element: set bits from start to imax
      int bit_offset = imax % nbits;
      mask           = (bitset_t{1} << bit_offset) - 1;
    } else {
      // Middle elements: set all bits
      mask = ~mask;
    }
    atomicOr(&bitset_data[current_index], mask);
  }
}

/**
 * @brief Convert a Faiss IDSelectorRange to a cuvs::core::bitset_view
 *
 * @param selector The Faiss IDSelectorRange to convert
 * @param bitset The cuvs::core::bitset_view to store the result
 */
void convert_to_bitset(raft::resources const& res,
                       const faiss::IDSelectorRange& selector,
                       cuvs::core::bitset_view<uint32_t, uint32_t> bitset)
{
  RAFT_EXPECTS(bitset.size() >= selector.imax,
               "IDSelectorRange is out of range for the given bitset");
  const uint32_t nbits = sizeof(uint32_t) * 8;
  auto original_nbits  = bitset.get_original_nbits();
  if (original_nbits == 0) { original_nbits = nbits; }
  uint32_t imin = selector.imin;
  uint32_t imax = selector.imax;

  uint32_t n_elements_to_set = 1 + (imax + original_nbits) / original_nbits;
  n_elements_to_set -= (imin + original_nbits) / original_nbits;
  auto stream = raft::resource::get_cuda_stream(res);

  const int threads_per_block = 256;
  const int blocks            = (n_elements_to_set + threads_per_block - 1) / threads_per_block;

  if (nbits == original_nbits) {
    set_range_kernel<uint32_t><<<blocks, threads_per_block, 0, stream>>>(
      (uint32_t*)bitset.data(), imin, imax, n_elements_to_set);
  } else if (original_nbits == 8) {
    set_range_kernel<uint8_t><<<blocks, threads_per_block, 0, stream>>>(
      (uint8_t*)bitset.data(), imin, imax, n_elements_to_set);
  } else if (original_nbits == 64) {
    set_range_kernel<uint64_t><<<blocks, threads_per_block, 0, stream>>>(
      (uint64_t*)bitset.data(), imin, imax, n_elements_to_set);
  } else {
    throw std::invalid_argument("Unsupported original_nbits");
  }
}

/**
 * @brief Convert a Faiss IDSelectorRange to a cuvs::core::bitset_view
 *
 * @param selector The Faiss IDSelectorRange to convert
 * @param bitset The cuvs::core::bitset_view to store the result
 */
void convert_to_bitset(raft::resources const& res,
                       const faiss::IDSelectorArray& selector,
                       cuvs::core::bitset_view<uint32_t, uint32_t> bitset)
{
  uint32_t n            = selector.n;
  auto d_indexes_to_set = raft::make_device_vector<faiss::idx_t, uint32_t>(res, n);
  raft::copy(res,
             d_indexes_to_set.view(),
             raft::make_host_vector_view<const faiss::idx_t, uint32_t>(selector.ids, n));
  thrust::for_each_n(
    raft::resource::get_thrust_policy(res),
    d_indexes_to_set.data_handle(),
    n,
    [bitset] __device__(const faiss::idx_t sample_index) { bitset.set(sample_index, true); });
}

void convert_to_bitset_bruteforce(raft::resources const& res,
                                  const faiss::IDSelector& selector,
                                  cuvs::core::bitset_view<uint32_t, uint32_t> bitset,
                                  int num_threads = 0)
{
  auto bitset_cpu = raft::make_host_vector<uint32_t, uint32_t>(bitset.n_elements());
  auto nbits      = sizeof(uint32_t) * 8;
  if (num_threads == 0) num_threads = omp_get_max_threads();
#pragma omp parallel for num_threads(num_threads)
  for (uint32_t i = 0; i < bitset.n_elements(); i++) {
    uint32_t element = uint32_t{0};
    for (uint32_t j = 0; j < nbits; j++) {
      if (i * nbits + j < bitset.size() && selector.is_member(i * nbits + j)) {
        element |= (uint32_t{1} << j);
      }
    }
    bitset_cpu(i) = element;
  }
  raft::copy(res, bitset.to_mdspan(), bitset_cpu.view());
  raft::resource::sync_stream(res);
}

/**
 * @brief Convert a Faiss IDSelector to a cuvs::core::bitset_view
 *
 * @param selector The Faiss IDSelector to convert
 * @param bitset The cuvs::core::bitset_view to store the result
 * @param num_threads Number of threads to use for the conversion. If 0, the number of threads is
 * set to the number of available threads.
 */
void convert_to_bitset(raft::resources const& res,
                       const faiss::IDSelector& selector,
                       cuvs::core::bitset_view<uint32_t, uint32_t> bitset,
                       int num_threads)
{
  // If the selector is simple, we can use the specialized functions
  // Otherwise use the brute force method
  try {
    auto range_selector = dynamic_cast<const faiss::IDSelectorRange&>(selector);
    convert_to_bitset(res, range_selector, bitset);
    return;
  } catch (const std::bad_cast& e) {
  }
  try {
    auto array_selector = dynamic_cast<const faiss::IDSelectorArray&>(selector);
    convert_to_bitset(res, array_selector, bitset);
    return;
  } catch (const std::bad_cast& e) {
  }
  convert_to_bitset_bruteforce(res, selector, bitset, num_threads);
}
}  // namespace cuvs::util
