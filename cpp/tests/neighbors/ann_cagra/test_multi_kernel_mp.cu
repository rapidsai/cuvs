/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Focused unit tests for the partition-aware variants of the MULTI_KERNEL CAGRA expansion
// kernels that don't require a dataset_descriptor_host. Validates that the mp launchers produce
// per-(query, partition) output slices identical to running the corresponding single-partition
// kernel for each partition's slice in turn.

#include <neighbors/detail/cagra/search_multi_kernel.cuh>

#include <gtest/gtest.h>

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace cuvs::neighbors::cagra::detail::multi_kernel_search {

class mp_simple_kernels_test : public ::testing::Test {
 protected:
  raft::resources res;
};

// remove_parent_bit_mp: clears MSB of each row's top-k indices. Validates the mp variant's
// row arithmetic (row = partition_id * num_queries + query_id) by running it against multiple
// sequential single-partition calls and comparing the full buffer afterward.
TEST_F(mp_simple_kernels_test, remove_parent_bit_mp_matches_sequential)
{
  using INDEX_T                     = uint32_t;
  constexpr uint32_t num_partitions = 3;
  constexpr uint32_t num_queries    = 4;
  constexpr uint32_t num_topk       = 16;
  constexpr uint32_t ld             = 24;  // ld > num_topk: trailing entries must not be touched

  constexpr INDEX_T msb_mask = INDEX_T{1} << (sizeof(INDEX_T) * 8 - 1);
  constexpr size_t total     = static_cast<size_t>(num_partitions) * num_queries * ld;

  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  std::vector<INDEX_T> host_input(total);
  std::mt19937 rng(0xdeadbeef);
  std::uniform_int_distribution<INDEX_T> dist(0, std::numeric_limits<INDEX_T>::max());
  for (auto& v : host_input) {
    v = dist(rng);
  }

  rmm::device_uvector<INDEX_T> dev_seq(total, stream);
  rmm::device_uvector<INDEX_T> dev_mp(total, stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    dev_seq.data(), host_input.data(), total * sizeof(INDEX_T), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    dev_mp.data(), host_input.data(), total * sizeof(INDEX_T), cudaMemcpyHostToDevice, stream));

  for (uint32_t p = 0; p < num_partitions; p++) {
    INDEX_T* part_slice = dev_seq.data() + static_cast<size_t>(p) * num_queries * ld;
    remove_parent_bit<INDEX_T>(num_queries, num_topk, part_slice, ld, stream);
  }

  remove_parent_bit_mp<INDEX_T>(num_queries, num_partitions, num_topk, dev_mp.data(), ld, stream);

  std::vector<INDEX_T> host_seq(total);
  std::vector<INDEX_T> host_mp(total);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    host_seq.data(), dev_seq.data(), total * sizeof(INDEX_T), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    host_mp.data(), dev_mp.data(), total * sizeof(INDEX_T), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(res);

  for (uint32_t p = 0; p < num_partitions; p++) {
    for (uint32_t q = 0; q < num_queries; q++) {
      const size_t row_offset = (static_cast<size_t>(p) * num_queries + q) * ld;
      // First num_topk entries: MSB cleared
      for (uint32_t i = 0; i < num_topk; i++) {
        ASSERT_EQ(host_seq[row_offset + i], host_mp[row_offset + i])
          << "mismatch at partition " << p << ", query " << q << ", slot " << i;
        ASSERT_EQ(host_mp[row_offset + i] & msb_mask, 0u)
          << "MSB not cleared at partition " << p << ", query " << q << ", slot " << i;
      }
      // Trailing entries [num_topk, ld): untouched, must match the original input
      for (uint32_t i = num_topk; i < ld; i++) {
        const size_t idx = row_offset + i;
        ASSERT_EQ(host_mp[idx], host_input[idx])
          << "trailing entry modified at partition " << p << ", query " << q << ", slot " << i;
        ASSERT_EQ(host_seq[idx], host_input[idx])
          << "(reference) trailing entry modified at partition " << p << ", query " << q
          << ", slot " << i;
      }
    }
  }
}

// pickup_next_parents_mp: scans each row's parent_candidates for unused entries (MSB == 0),
// picks the first parent_list_size of them, sets their MSB to mark as used, and writes their
// positions to parent_list. Also clears terminate_flag if any (query, partition) found new
// parents. Validates the mp variant matches the sequential reference for all of these effects.
TEST_F(mp_simple_kernels_test, pickup_next_parents_mp_matches_sequential)
{
  using INDEX_T                             = uint32_t;
  constexpr uint32_t num_partitions         = 3;
  constexpr uint32_t num_queries            = 4;
  constexpr uint32_t parent_candidates_size = 32;
  constexpr uint32_t parent_list_size       = 4;
  constexpr uint32_t lds                    = parent_candidates_size;
  constexpr uint32_t ldd                    = parent_list_size;
  constexpr uint32_t hash_bitlen       = 8;  // 256-entry hashmap, unused w/ small_hash_bitlen=0
  constexpr uint32_t small_hash_bitlen = 0;  // bypass the small-hash insertion branch

  constexpr INDEX_T msb_mask = INDEX_T{1} << (sizeof(INDEX_T) * 8 - 1);
  const size_t rows          = static_cast<size_t>(num_partitions) * num_queries;
  const size_t cand_total    = rows * lds;
  const size_t list_total    = rows * ldd;
  const size_t hash_total    = rows * (1u << hash_bitlen);

  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  // Per-row parent candidates: random uint32_t with MSB cleared, then sprinkle a few entries
  // with MSB set ("already used"). Distribution ensures plenty of unused parents per row so
  // both the picking logic and the terminate_flag clear paths exercise.
  std::vector<INDEX_T> host_cand(cand_total);
  std::mt19937 rng(0xc0ffee);
  std::uniform_int_distribution<INDEX_T> dist(0, (INDEX_T{1} << 30) - 1);
  for (auto& v : host_cand) {
    v = dist(rng);
  }
  std::uniform_int_distribution<int> used_dist(0, 7);
  for (auto& v : host_cand) {
    if (used_dist(rng) == 0) { v |= msb_mask; }
  }

  rmm::device_uvector<INDEX_T> dev_cand_seq(cand_total, stream);
  rmm::device_uvector<INDEX_T> dev_cand_mp(cand_total, stream);
  rmm::device_uvector<INDEX_T> dev_list_seq(list_total, stream);
  rmm::device_uvector<INDEX_T> dev_list_mp(list_total, stream);
  rmm::device_uvector<INDEX_T> dev_hashmap(hash_total,
                                           stream);  // contents irrelevant w/ small_hash_bitlen=0
  rmm::device_uvector<uint32_t> dev_term_seq(1, stream);
  rmm::device_uvector<uint32_t> dev_term_mp(1, stream);

  RAFT_CUDA_TRY(cudaMemcpyAsync(dev_cand_seq.data(),
                                host_cand.data(),
                                cand_total * sizeof(INDEX_T),
                                cudaMemcpyHostToDevice,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(dev_cand_mp.data(),
                                host_cand.data(),
                                cand_total * sizeof(INDEX_T),
                                cudaMemcpyHostToDevice,
                                stream));

  // Initialize terminate_flag to 1 (as the caller does in operator()).
  const uint32_t one = 1;
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(dev_term_seq.data(), &one, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(dev_term_mp.data(), &one, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

  // Sequential reference: one call per partition, sharing the same terminate_flag.
  for (uint32_t p = 0; p < num_partitions; p++) {
    INDEX_T* cand_slice = dev_cand_seq.data() + static_cast<size_t>(p) * num_queries * lds;
    INDEX_T* list_slice = dev_list_seq.data() + static_cast<size_t>(p) * num_queries * ldd;
    INDEX_T* hash_slice =
      dev_hashmap.data() + static_cast<size_t>(p) * num_queries * (1u << hash_bitlen);
    pickup_next_parents<INDEX_T>(cand_slice,
                                 lds,
                                 parent_candidates_size,
                                 num_queries,
                                 hash_slice,
                                 hash_bitlen,
                                 small_hash_bitlen,
                                 list_slice,
                                 ldd,
                                 parent_list_size,
                                 dev_term_seq.data(),
                                 stream);
  }

  // mp variant: one call across all partitions.
  pickup_next_parents_mp<INDEX_T>(dev_cand_mp.data(),
                                  lds,
                                  parent_candidates_size,
                                  num_queries,
                                  num_partitions,
                                  dev_hashmap.data(),
                                  hash_bitlen,
                                  small_hash_bitlen,
                                  dev_list_mp.data(),
                                  ldd,
                                  parent_list_size,
                                  dev_term_mp.data(),
                                  stream);

  std::vector<INDEX_T> host_cand_seq(cand_total);
  std::vector<INDEX_T> host_cand_mp(cand_total);
  std::vector<INDEX_T> host_list_seq(list_total);
  std::vector<INDEX_T> host_list_mp(list_total);
  uint32_t host_term_seq = 0;
  uint32_t host_term_mp  = 0;

  RAFT_CUDA_TRY(cudaMemcpyAsync(host_cand_seq.data(),
                                dev_cand_seq.data(),
                                cand_total * sizeof(INDEX_T),
                                cudaMemcpyDeviceToHost,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(host_cand_mp.data(),
                                dev_cand_mp.data(),
                                cand_total * sizeof(INDEX_T),
                                cudaMemcpyDeviceToHost,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(host_list_seq.data(),
                                dev_list_seq.data(),
                                list_total * sizeof(INDEX_T),
                                cudaMemcpyDeviceToHost,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(host_list_mp.data(),
                                dev_list_mp.data(),
                                list_total * sizeof(INDEX_T),
                                cudaMemcpyDeviceToHost,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    &host_term_seq, dev_term_seq.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    &host_term_mp, dev_term_mp.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(res);

  ASSERT_EQ(host_term_seq, host_term_mp) << "terminate_flag differs after mp vs sequential";
  // With ~7/8 of candidates unused, every (query, partition) row has new parents and the
  // flag should be cleared to 0.
  ASSERT_EQ(host_term_mp, 0u) << "expected terminate_flag = 0 given the input setup";

  for (uint32_t p = 0; p < num_partitions; p++) {
    for (uint32_t q = 0; q < num_queries; q++) {
      const size_t cand_row = (static_cast<size_t>(p) * num_queries + q) * lds;
      const size_t list_row = (static_cast<size_t>(p) * num_queries + q) * ldd;
      for (uint32_t i = 0; i < parent_candidates_size; i++) {
        ASSERT_EQ(host_cand_seq[cand_row + i], host_cand_mp[cand_row + i])
          << "candidate mismatch at partition " << p << ", query " << q << ", slot " << i;
      }
      for (uint32_t i = 0; i < parent_list_size; i++) {
        ASSERT_EQ(host_list_seq[list_row + i], host_list_mp[list_row + i])
          << "parent_list mismatch at partition " << p << ", query " << q << ", slot " << i;
      }
    }
  }
}

}  // namespace cuvs::neighbors::cagra::detail::multi_kernel_search
