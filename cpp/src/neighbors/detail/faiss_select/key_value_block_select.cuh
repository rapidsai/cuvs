/**
 * SPDX-FileCopyrightText: Copyright (c) Facebook, Inc. and its affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

#include "MergeNetworkUtils.cuh"
#include "Select.cuh"

// TODO(snanditale): Need to think further about the impact (and new boundaries created) on the
// registers because this will change the max k that can be processed. One solution might be to
// break up k into multiple batches for larger k.

namespace cuvs::neighbors::detail::faiss_select {

// `Dir` true, produce largest values.
// `Dir` false, produce smallest values.
template <typename K,
          typename V,
          bool Dir,
          typename Comp,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
struct key_value_block_select {
  static constexpr int kNumWarps          = ThreadsPerBlock / WarpSize;
  static constexpr int kTotalWarpSortSize = NumWarpQ;

  __device__ inline key_value_block_select(
    K initKVal, K initVKey, V initVVal, K* smemK, KeyValuePair<K, V>* smemV, int k)
    : init_k(initKVal),
      init_vk(initVKey),
      init_vv(initVVal),

      warp_k_top(initKVal),
      warp_k_top_r_dist(initKVal),
      shared_k(smemK),
      shared_v(smemV),
      k_minus1(k - 1)
  {
    static_assert(utils::is_power_of2(ThreadsPerBlock), "threads must be a power-of-2");
    static_assert(utils::is_power_of2(NumWarpQ), "warp queue must be power-of-2");

    // Fill the per-thread queue keys with the default value
#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      thread_k[i]       = init_k;
      thread_v[i].key   = init_vk;
      thread_v[i].value = init_vv;
    }

    int lane_id = raft::laneId();
    int warp_id = threadIdx.x / WarpSize;
    warp_k      = shared_k + warp_id * kTotalWarpSortSize;
    warp_v      = shared_v + warp_id * kTotalWarpSortSize;

    // Fill warp queue (only the actual queue space is fine, not where
    // we write the per-thread queues for merging)
    for (int i = lane_id; i < NumWarpQ; i += WarpSize) {
      warp_k[i]       = init_k;
      warp_v[i].key   = init_vk;
      warp_v[i].value = init_vv;
    }

    warpFence();
  }

  __device__ inline void add_thread_q(K k, K vk, V vv)
  {
    if (Dir ? Comp::gt(k, warp_k_top) : Comp::lt(k, warp_k_top)) {
      // Rotate right
#pragma unroll
      for (int i = NumThreadQ - 1; i > 0; --i) {
        thread_k[i]       = thread_k[i - 1];
        thread_v[i].key   = thread_v[i - 1].key;
        thread_v[i].value = thread_v[i - 1].value;
      }

      thread_k[0]       = k;
      thread_v[0].key   = vk;
      thread_v[0].value = vv;
      ++num_vals;
    }
  }

  __device__ inline void check_thread_q()
  {
    bool need_sort = (num_vals == NumThreadQ);

#if CUDA_VERSION >= 9000
    need_sort = __any_sync(0xffffffff, need_sort);
#else
    need_sort = __any(need_sort);
#endif

    if (!need_sort) {
      // no lanes have triggered a sort
      return;
    }

    // This has a trailing warpFence
    merge_warp_q();

    // Any top-k elements have been merged into the warp queue; we're
    // free to reset the thread queues
    num_vals = 0;

#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      thread_k[i]       = init_k;
      thread_v[i].key   = init_vk;
      thread_v[i].value = init_vv;
    }

    // We have to beat at least this element
    warp_k_top        = warp_k[k_minus1];
    warp_k_top_r_dist = warp_v[k_minus1].key;

    warpFence();
  }

  /// This function handles sorting and merging together the
  /// per-thread queues with the warp-wide queue, creating a sorted
  /// list across both
  __device__ inline void merge_warp_q()
  {
    int lane_id = raft::laneId();

    // Sort all of the per-thread queues
    warp_sort_any_registers<K, KeyValuePair<K, V>, NumThreadQ, !Dir, Comp>(thread_k, thread_v);

    constexpr int kNumWarpQRegisters = NumWarpQ / WarpSize;
    K warp_k_registers[kNumWarpQRegisters];
    KeyValuePair<K, V> warp_v_registers[kNumWarpQRegisters];

#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warp_k_registers[i]       = warp_k[i * WarpSize + lane_id];
      warp_v_registers[i].key   = warp_v[i * WarpSize + lane_id].key;
      warp_v_registers[i].value = warp_v[i * WarpSize + lane_id].value;
    }

    warpFence();

    // The warp queue is already sorted, and now that we've sorted the
    // per-thread queue, merge both sorted lists together, producing
    // one sorted list
    warp_merge_any_registers<K,
                             KeyValuePair<K, V>,
                             kNumWarpQRegisters,
                             NumThreadQ,
                             !Dir,
                             Comp,
                             false>(warp_k_registers, warp_v_registers, thread_k, thread_v);

    // Write back out the warp queue
#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warp_k[i * WarpSize + lane_id]       = warp_k_registers[i];
      warp_v[i * WarpSize + lane_id].key   = warp_v_registers[i].key;
      warp_v[i * WarpSize + lane_id].value = warp_v_registers[i].value;
    }

    warpFence();
  }

  /// WARNING: all threads in a warp must participate in this.
  /// Otherwise, you must call the constituent parts separately.
  __device__ inline void add(K k, K vk, V vv)
  {
    add_thread_q(k, vk, vv);
    check_thread_q();
  }

  __device__ inline void reduce()
  {
    // Have all warps dump and merge their queues; this will produce
    // the final per-warp results
    merge_warp_q();

    // block-wide dep; thus far, all warps have been completely
    // independent
    __syncthreads();

    // All warp queues are contiguous in smem.
    // Now, we have kNumWarps lists of NumWarpQ elements.
    // This is a power of 2.
    final_block_merge<kNumWarps, ThreadsPerBlock, K, KeyValuePair<K, V>, NumWarpQ, Dir, Comp>::
      merge(shared_k, shared_v);

    // The block-wide merge has a trailing syncthreads
  }

  // Default element key
  const K init_k;

  // Default element value
  const K init_vk;
  const V init_vv;

  // Number of valid elements in our thread queue
  int num_vals{0};

  // The k-th highest (Dir) or lowest (!Dir) element
  K warp_k_top;

  K warp_k_top_r_dist;

  // Thread queue values
  K thread_k[NumThreadQ];
  KeyValuePair<K, V> thread_v[NumThreadQ];

  // Queues for all warps
  K* shared_k;
  KeyValuePair<K, V>* shared_v;

  // Our warp's queue (points into shared_k/shared_v)
  // warp_k[0] is highest (Dir) or lowest (!Dir)
  K* warp_k;
  KeyValuePair<K, V>* warp_v;

  // This is a cached k-1 value
  int k_minus1;
};

}  // namespace cuvs::neighbors::detail::faiss_select
