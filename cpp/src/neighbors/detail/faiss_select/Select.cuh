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

#include "Comparators.cuh"
#include "MergeNetworkBlock.cuh"
#include "MergeNetworkWarp.cuh"
#include <raft/core/kvp.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_utils.cuh>

namespace cuvs::neighbors::detail::faiss_select {

// Import raft utilities needed by this file
using raft::KeyValuePair;
using raft::max_op;
using raft::min_op;
using raft::shfl;
using raft::warpFence;
using raft::warpReduce;
using raft::WarpSize;

// Specialization for block-wide monotonic merges producing a merge sort
// since what we really want is a constexpr loop expansion
template <int NumWarps,
          int NumThreads,
          typename K,
          typename V,
          int NumWarpQ,
          bool Dir,
          typename Comp>
struct final_block_merge {};

template <int NumThreads, typename K, typename V, int NumWarpQ, bool Dir, typename Comp>
struct final_block_merge<1, NumThreads, K, V, NumWarpQ, Dir, Comp> {
  static inline __device__ void merge(K* shared_k, V* shared_v)
  {
    // no merge required; single warp
  }
};

template <int NumThreads, typename K, typename V, int NumWarpQ, bool Dir, typename Comp>
struct final_block_merge<2, NumThreads, K, V, NumWarpQ, Dir, Comp> {
  static inline __device__ void merge(K* shared_k, V* shared_v)
  {
    // Final merge doesn't need to fully merge the second list
    block_merge<NumThreads, K, V, NumThreads / (WarpSize * 2), NumWarpQ, !Dir, Comp, false>(
      shared_k, shared_v);
  }
};

template <int NumThreads, typename K, typename V, int NumWarpQ, bool Dir, typename Comp>
struct final_block_merge<4, NumThreads, K, V, NumWarpQ, Dir, Comp> {
  static inline __device__ void merge(K* shared_k, V* shared_v)
  {
    block_merge<NumThreads, K, V, NumThreads / (WarpSize * 2), NumWarpQ, !Dir, Comp>(shared_k,
                                                                                     shared_v);
    // Final merge doesn't need to fully merge the second list
    block_merge<NumThreads, K, V, NumThreads / (WarpSize * 4), NumWarpQ * 2, !Dir, Comp, false>(
      shared_k, shared_v);
  }
};

template <int NumThreads, typename K, typename V, int NumWarpQ, bool Dir, typename Comp>
struct final_block_merge<8, NumThreads, K, V, NumWarpQ, Dir, Comp> {
  static inline __device__ void merge(K* shared_k, V* shared_v)
  {
    block_merge<NumThreads, K, V, NumThreads / (WarpSize * 2), NumWarpQ, !Dir, Comp>(shared_k,
                                                                                     shared_v);
    block_merge<NumThreads, K, V, NumThreads / (WarpSize * 4), NumWarpQ * 2, !Dir, Comp>(shared_k,
                                                                                         shared_v);
    // Final merge doesn't need to fully merge the second list
    block_merge<NumThreads, K, V, NumThreads / (WarpSize * 8), NumWarpQ * 4, !Dir, Comp, false>(
      shared_k, shared_v);
  }
};

// `Dir` true, produce largest values.
// `Dir` false, produce smallest values.
template <typename K,
          typename V,
          bool Dir,
          typename Comp,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
struct block_select {
  static constexpr int kNumWarps          = ThreadsPerBlock / WarpSize;
  static constexpr int kTotalWarpSortSize = NumWarpQ;

  __device__ inline block_select(K initKVal, V initVVal, K* smemK, V* smemV, int k)
    : init_k(initKVal),
      init_v(initVVal),

      warp_k_top(initKVal),
      shared_k(smemK),
      shared_v(smemV),
      k_minus1(k - 1)
  {
    static_assert(utils::is_power_of2(ThreadsPerBlock), "threads must be a power-of-2");
    static_assert(utils::is_power_of2(NumWarpQ), "warp queue must be power-of-2");

    // Fill the per-thread queue keys with the default value
#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      thread_k[i] = init_k;
      thread_v[i] = init_v;
    }

    int lane_id = raft::laneId();
    int warp_id = threadIdx.x / WarpSize;
    warp_k      = shared_k + warp_id * kTotalWarpSortSize;
    warp_v      = shared_v + warp_id * kTotalWarpSortSize;

    // Fill warp queue (only the actual queue space is fine, not where
    // we write the per-thread queues for merging)
    for (int i = lane_id; i < NumWarpQ; i += WarpSize) {
      warp_k[i] = init_k;
      warp_v[i] = init_v;
    }

    warpFence();
  }

  __device__ inline void add_thread_q(K k, V v)
  {
    if (Dir ? Comp::gt(k, warp_k_top) : Comp::lt(k, warp_k_top)) {
      // Rotate right
#pragma unroll
      for (int i = NumThreadQ - 1; i > 0; --i) {
        thread_k[i] = thread_k[i - 1];
        thread_v[i] = thread_v[i - 1];
      }

      thread_k[0] = k;
      thread_v[0] = v;
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
      thread_k[i] = init_k;
      thread_v[i] = init_v;
    }

    // We have to beat at least this element
    warp_k_top = warp_k[k_minus1];

    warpFence();
  }

  /// This function handles sorting and merging together the
  /// per-thread queues with the warp-wide queue, creating a sorted
  /// list across both
  __device__ inline void merge_warp_q()
  {
    int lane_id = raft::laneId();

    // Sort all of the per-thread queues
    warp_sort_any_registers<K, V, NumThreadQ, !Dir, Comp>(thread_k, thread_v);

    constexpr int kNumWarpQRegisters = NumWarpQ / WarpSize;
    K warp_k_registers[kNumWarpQRegisters];
    V warp_v_registers[kNumWarpQRegisters];

#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warp_k_registers[i] = warp_k[i * WarpSize + lane_id];
      warp_v_registers[i] = warp_v[i * WarpSize + lane_id];
    }

    warpFence();

    // The warp queue is already sorted, and now that we've sorted the
    // per-thread queue, merge both sorted lists together, producing
    // one sorted list
    warp_merge_any_registers<K, V, kNumWarpQRegisters, NumThreadQ, !Dir, Comp, false>(
      warp_k_registers, warp_v_registers, thread_k, thread_v);

    // Write back out the warp queue
#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warp_k[i * WarpSize + lane_id] = warp_k_registers[i];
      warp_v[i * WarpSize + lane_id] = warp_v_registers[i];
    }

    warpFence();
  }

  /// WARNING: all threads in a warp must participate in this.
  /// Otherwise, you must call the constituent parts separately.
  __device__ inline void add(K k, V v)
  {
    add_thread_q(k, v);
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
    final_block_merge<kNumWarps, ThreadsPerBlock, K, V, NumWarpQ, Dir, Comp>::merge(shared_k,
                                                                                    shared_v);

    // The block-wide merge has a trailing syncthreads
  }

  // Default element key
  const K init_k;

  // Default element value
  const V init_v;

  // Number of valid elements in our thread queue
  int num_vals{0};

  // The k-th highest (Dir) or lowest (!Dir) element
  K warp_k_top;

  // Thread queue values
  K thread_k[NumThreadQ];
  V thread_v[NumThreadQ];

  // Queues for all warps
  K* shared_k;
  V* shared_v;

  // Our warp's queue (points into shared_k/shared_v)
  // warp_k[0] is highest (Dir) or lowest (!Dir)
  K* warp_k;
  V* warp_v;

  // This is a cached k-1 value
  int k_minus1;
};

/// Specialization for k == 1 (NumWarpQ == 1)
template <typename K, typename V, bool Dir, typename Comp, int NumThreadQ, int ThreadsPerBlock>
struct block_select<K, V, Dir, Comp, 1, NumThreadQ, ThreadsPerBlock> {
  static constexpr int kNumWarps = ThreadsPerBlock / WarpSize;

  __device__ inline block_select(K init_k, V init_v, K* smemK, V* smemV, int k)
    : thread_k(init_k), thread_v(init_v), shared_k(smemK), shared_v(smemV)
  {
  }

  __device__ inline void add_thread_q(K k, V v)
  {
    bool swap = Dir ? Comp::gt(k, thread_k) : Comp::lt(k, thread_k);
    thread_k  = swap ? k : thread_k;
    thread_v  = swap ? v : thread_v;
  }

  __device__ inline void check_thread_q()
  {
    // We don't need to do anything here, since the warp doesn't
    // cooperate until the end
  }

  __device__ inline void add(K k, V v) { add_thread_q(k, v); }

  __device__ inline void reduce()
  {
    // Reduce within the warp
    KeyValuePair<K, V> pair(thread_k, thread_v);

    if (Dir) {
      pair = warpReduce(pair, max_op{});
    } else {
      pair = warpReduce(pair, min_op{});
    }

    // Each warp writes out a single value
    int lane_id = raft::laneId();
    int warp_id = threadIdx.x / WarpSize;

    if (lane_id == 0) {
      shared_k[warp_id] = pair.key;
      shared_v[warp_id] = pair.value;
    }

    __syncthreads();

    // We typically use this for small blocks (<= 128), just having the
    // first thread in the block perform the reduction across warps is
    // faster
    if (threadIdx.x == 0) {
      thread_k = shared_k[0];
      thread_v = shared_v[0];

#pragma unroll
      for (int i = 1; i < kNumWarps; ++i) {
        K k = shared_k[i];
        V v = shared_v[i];

        bool swap = Dir ? Comp::gt(k, thread_k) : Comp::lt(k, thread_k);
        thread_k  = swap ? k : thread_k;
        thread_v  = swap ? v : thread_v;
      }

      // Hopefully a thread's smem reads/writes are ordered wrt
      // itself, so no barrier needed :)
      shared_k[0] = thread_k;
      shared_v[0] = thread_v;
    }

    // In case other threads wish to read this value
    __syncthreads();
  }

  // thread_k is lowest (Dir) or highest (!Dir)
  K thread_k;
  V thread_v;

  // Where we reduce in smem
  K* shared_k;
  V* shared_v;
};

//
// per-warp warp_select
//

// `Dir` true, produce largest values.
// `Dir` false, produce smallest values.
template <typename K,
          typename V,
          bool Dir,
          typename Comp,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
struct warp_select {
  static constexpr int kNumWarpQRegisters = NumWarpQ / WarpSize;

  __device__ inline warp_select(K initKVal, V initVVal, int k)
    : init_k(initKVal), init_v(initVVal), warp_k_top(initKVal), k_lane((k - 1) % WarpSize)
  {
    static_assert(utils::is_power_of2(ThreadsPerBlock), "threads must be a power-of-2");
    static_assert(utils::is_power_of2(NumWarpQ), "warp queue must be power-of-2");

    // Fill the per-thread queue keys with the default value
#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      thread_k[i] = init_k;
      thread_v[i] = init_v;
    }

    // Fill the warp queue with the default value
#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warp_k[i] = init_k;
      warp_v[i] = init_v;
    }
  }

  __device__ inline void add_thread_q(K k, V v)
  {
    if (Dir ? Comp::gt(k, warp_k_top) : Comp::lt(k, warp_k_top)) {
      // Rotate right
#pragma unroll
      for (int i = NumThreadQ - 1; i > 0; --i) {
        thread_k[i] = thread_k[i - 1];
        thread_v[i] = thread_v[i - 1];
      }

      thread_k[0] = k;
      thread_v[0] = v;
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

    merge_warp_q();

    // Any top-k elements have been merged into the warp queue; we're
    // free to reset the thread queues
    num_vals = 0;

#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      thread_k[i] = init_k;
      thread_v[i] = init_v;
    }

    // We have to beat at least this element
    warp_k_top = shfl(warp_k[kNumWarpQRegisters - 1], k_lane);
  }

  /// This function handles sorting and merging together the
  /// per-thread queues with the warp-wide queue, creating a sorted
  /// list across both
  __device__ inline void merge_warp_q()
  {
    // Sort all of the per-thread queues
    warp_sort_any_registers<K, V, NumThreadQ, !Dir, Comp>(thread_k, thread_v);

    // The warp queue is already sorted, and now that we've sorted the
    // per-thread queue, merge both sorted lists together, producing
    // one sorted list
    warp_merge_any_registers<K, V, kNumWarpQRegisters, NumThreadQ, !Dir, Comp, false>(
      warp_k, warp_v, thread_k, thread_v);
  }

  /// WARNING: all threads in a warp must participate in this.
  /// Otherwise, you must call the constituent parts separately.
  __device__ inline void add(K k, V v)
  {
    add_thread_q(k, v);
    check_thread_q();
  }

  __device__ inline void reduce()
  {
    // Have all warps dump and merge their queues; this will produce
    // the final per-warp results
    merge_warp_q();
  }

  /// Dump final k selected values for this warp out
  __device__ inline void write_out(K* outK, V* outV, int k)
  {
    int lane_id = raft::laneId();

#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      int idx = i * WarpSize + lane_id;

      if (idx < k) {
        outK[idx] = warp_k[i];
        outV[idx] = warp_v[i];
      }
    }
  }

  // Default element key
  const K init_k;

  // Default element value
  const V init_v;

  // Number of valid elements in our thread queue
  int num_vals{0};

  // The k-th highest (Dir) or lowest (!Dir) element
  K warp_k_top;

  // Thread queue values
  K thread_k[NumThreadQ];
  V thread_v[NumThreadQ];

  // warp_k[0] is highest (Dir) or lowest (!Dir)
  K warp_k[kNumWarpQRegisters];
  V warp_v[kNumWarpQRegisters];

  // This is what lane we should load an approximation (>=k) to the
  // kth element from the last register in the warp queue (i.e.,
  // warp_k[kNumWarpQRegisters - 1]).
  int k_lane;  // NOLINT(modernize-use-default-member-init)
};

/// Specialization for k == 1 (NumWarpQ == 1)
template <typename K, typename V, bool Dir, typename Comp, int NumThreadQ, int ThreadsPerBlock>
struct warp_select<K, V, Dir, Comp, 1, NumThreadQ, ThreadsPerBlock> {
  static constexpr int kNumWarps = ThreadsPerBlock / WarpSize;

  __device__ inline warp_select(K init_k, V init_v, int k) : thread_k(init_k), thread_v(init_v) {}

  __device__ inline void add_thread_q(K k, V v)
  {
    bool swap = Dir ? Comp::gt(k, thread_k) : Comp::lt(k, thread_k);
    thread_k  = swap ? k : thread_k;
    thread_v  = swap ? v : thread_v;
  }

  __device__ inline void check_thread_q()
  {
    // We don't need to do anything here, since the warp doesn't
    // cooperate until the end
  }

  __device__ inline void add(K k, V v) { add_thread_q(k, v); }

  __device__ inline void reduce()
  {
    // Reduce within the warp
    KeyValuePair<K, V> pair(thread_k, thread_v);

    if (Dir) {
      pair = warpReduce(pair, max_op{});
    } else {
      pair = warpReduce(pair, min_op{});
    }

    thread_k = pair.key;
    thread_v = pair.value;
  }

  /// Dump final k selected values for this warp out
  __device__ inline void write_out(K* outK, V* outV, int k)
  {
    if (raft::laneId() == 0) {
      *outK = thread_k;
      *outV = thread_v;
    }
  }

  // thread_k is lowest (Dir) or highest (!Dir)
  K thread_k;
  V thread_v;
};

}  // namespace cuvs::neighbors::detail::faiss_select
