/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#pragma once

#include "utils.hpp"

// TODO: This shouldn't be invoking anything from detail outside of neighbors/
#include <raft/core/detail/macros.hpp>
#include <raft/util/device_atomics.cuh>

#include <cstdint>

#define HASHMAP_LINEAR_PROBING

// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored
// #pragma GCC diagnostic pop
namespace cuvs::neighbors::cagra::detail {
namespace hashmap {

RAFT_INLINE_FUNCTION uint32_t get_size(const uint32_t bitlen) { return 1U << bitlen; }

template <class IdxT>
RAFT_DEVICE_INLINE_FUNCTION void init(IdxT* const table,
                                      const unsigned bitlen,
                                      unsigned FIRST_TID = 0)
{
  if (threadIdx.x < FIRST_TID) return;
  for (unsigned i = threadIdx.x - FIRST_TID; i < get_size(bitlen); i += blockDim.x - FIRST_TID) {
    table[i] = utils::get_max_value<IdxT>();
  }
}

template <class IdxT, unsigned SUPPORT_REMOVE = 0>
RAFT_DEVICE_INLINE_FUNCTION uint32_t insert(IdxT* const table,
                                            const uint32_t bitlen,
                                            const IdxT key)
{
  // Open addressing is used for collision resolution
  const uint32_t size     = get_size(bitlen);
  const uint32_t bit_mask = size - 1;
#ifdef HASHMAP_LINEAR_PROBING
  // Linear probing
  IdxT index                = (key ^ (key >> bitlen)) & bit_mask;
  constexpr uint32_t stride = 1;
#else
  // Double hashing
  uint32_t index        = key & bit_mask;
  const uint32_t stride = (key >> bitlen) * 2 + 1;
#endif
  constexpr IdxT hashval_empty = ~static_cast<IdxT>(0);
  const IdxT removed_key       = key | utils::gen_index_msb_1_mask<IdxT>::value;
  for (unsigned i = 0; i < size; i++) {
    const IdxT old = atomicCAS(&table[index], hashval_empty, key);
    if (old == hashval_empty) {
      return 1;
    } else if (old == key) {
      return 0;
    } else if (SUPPORT_REMOVE) {
      // Checks if this key has been removed before.
      const uint32_t old = atomicCAS(&table[index], removed_key, key);
      if (old == removed_key) {
        return 1;
      } else if (old == key) {
        return 0;
      }
    }
    index = (index + stride) & bit_mask;
  }
  return 0;
}

template <class IdxT, unsigned SUPPORT_REMOVE = 0>
RAFT_DEVICE_INLINE_FUNCTION uint32_t search(IdxT* table, const uint32_t bitlen, const IdxT key)
{
  const uint32_t size     = get_size(bitlen);
  const uint32_t bit_mask = size - 1;
#ifdef HASHMAP_LINEAR_PROBING
  // Linear probing
  IdxT index                = (key ^ (key >> bitlen)) & bit_mask;
  constexpr uint32_t stride = 1;
#else
  // Double hashing
  IdxT index            = key & bit_mask;
  const uint32_t stride = (key >> bitlen) * 2 + 1;
#endif
  constexpr IdxT hashval_empty = ~static_cast<IdxT>(0);
  const IdxT removed_key       = key | utils::gen_index_msb_1_mask<IdxT>::value;
  for (unsigned i = 0; i < size; i++) {
    const IdxT val = table[index];
    if (val == key) {
      return 1;
    } else if (val == hashval_empty) {
      return 0;
    } else if (SUPPORT_REMOVE) {
      // Check if this key has been removed.
      if (val == removed_key) { return 0; }
    }
    index = (index + stride) & bit_mask;
  }
  return 0;
}

template <class IdxT>
RAFT_DEVICE_INLINE_FUNCTION uint32_t remove(IdxT* table, const uint32_t bitlen, const IdxT key)
{
  const uint32_t size     = get_size(bitlen);
  const uint32_t bit_mask = size - 1;
#ifdef HASHMAP_LINEAR_PROBING
  // Linear probing
  IdxT index                = (key ^ (key >> bitlen)) & bit_mask;
  constexpr uint32_t stride = 1;
#else
  // Double hashing
  IdxT index            = key & bit_mask;
  const uint32_t stride = (key >> bitlen) * 2 + 1;
#endif
  constexpr IdxT hashval_empty = ~static_cast<IdxT>(0);
  const IdxT removed_key       = key | utils::gen_index_msb_1_mask<IdxT>::value;
  for (unsigned i = 0; i < size; i++) {
    // To remove a key, set the MSB to 1.
    const uint32_t old = atomicCAS(&table[index], key, removed_key);
    if (old == key) {
      return 1;
    } else if (old == hashval_empty) {
      return 0;
    }
    index = (index + stride) & bit_mask;
  }
  return 0;
}

template <class IdxT, unsigned SUPPORT_REMOVE = 0>
RAFT_DEVICE_INLINE_FUNCTION uint32_t
insert(unsigned team_size, IdxT* const table, const uint32_t bitlen, const IdxT key)
{
  IdxT ret = 0;
  if (threadIdx.x % team_size == 0) { ret = insert(table, bitlen, key); }
  for (unsigned offset = 1; offset < team_size; offset *= 2) {
    ret |= __shfl_xor_sync(0xffffffff, ret, offset);
  }
  return ret;
}

}  // namespace hashmap
}  // namespace cuvs::neighbors::cagra::detail
