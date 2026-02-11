/**
 * SPDX-FileCopyrightText: Copyright (c) Facebook, Inc. and its affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include "StaticUtils.h"
#include <raft/util/cuda_utils.cuh>

namespace cuvs::neighbors::detail::faiss_select {

// Import raft utilities needed by this file
using raft::shfl;
using raft::shfl_xor;
using raft::WarpSize;

//
// This file contains functions to:
//
// -perform bitonic merges on pairs of sorted lists, held in
// registers. Each list contains N * WarpSize (multiple of 32)
// elements for some N.
// The bitonic merge is implemented for arbitrary sizes;
// sorted list A of size N1 * WarpSize registers
// sorted list B of size N2 * WarpSize registers =>
// sorted list C if size (N1 + N2) * WarpSize registers. N1 and N2
// are >= 1 and don't have to be powers of 2.
//
// -perform bitonic sorts on a set of N * WarpSize key/value pairs
// held in registers, by using the above bitonic merge as a
// primitive.
// N can be an arbitrary N >= 1; i.e., the bitonic sort here supports
// odd sizes and doesn't require the input to be a power of 2.
//
// The sort or merge network is completely statically instantiated via
// template specialization / expansion and constexpr, and it uses warp
// shuffles to exchange values between warp lanes.
//
// A note about comparisons:
//
// For a sorting network of keys only, we only need one
// comparison (a < b). However, what we really need to know is
// if one lane chooses to exchange a value, then the
// corresponding lane should also do the exchange.
// Thus, if one just uses the negation !(x < y) in the higher
// lane, this will also include the case where (x == y). Thus, one
// lane in fact performs an exchange and the other doesn't, but
// because the only value being exchanged is equivalent, nothing has
// changed.
// So, you can get away with just one comparison and its negation.
//
// If we're sorting keys and values, where equivalent keys can
// exist, then this is a problem, since we want to treat (x, v1)
// as not equivalent to (x, v2).
//
// To remedy this, you can either compare with a lexicographic
// ordering (a.k < b.k || (a.k == b.k && a.v < b.v)), which since
// we're predicating all of the choices results in 3 comparisons
// being executed, or we can invert the selection so that there is no
// middle choice of equality; the other lane will likewise
// check that (b.k > a.k) (the higher lane has the values
// swapped). Then, the first lane swaps if and only if the
// second lane swaps; if both lanes have equivalent keys, no
// swap will be performed. This results in only two comparisons
// being executed.
//
// If you don't consider values as well, then this does not produce a
// consistent ordering among (k, v) pairs with equivalent keys but
// different values; for us, we don't really care about ordering or
// stability here.
//
// I have tried both re-arranging the order in the higher lane to get
// away with one comparison or adding the value to the check; both
// result in greater register consumption or lower speed than just
// performing both < and > comparisons with the variables, so I just
// stick with this.

// This function merges WarpSize / 2L lists in parallel using warp
// shuffles.
// It works on at most size-16 lists, as we need 32 threads for this
// shuffle merge.
//
// If IsBitonic is false, the first stage is reversed, so we don't
// need to sort directionally. It's still technically a bitonic sort.
template <typename K, typename V, int L, bool Dir, typename Comp, bool IsBitonic>
inline __device__ void warp_bitonic_merge_le16(K& k, V& v)
{
  static_assert(utils::is_power_of2(L), "L must be a power-of-2");
  static_assert(L <= WarpSize / 2, "merge list size must be <= 16");

  int lane_id = raft::laneId();

  if (!IsBitonic) {
    // Reverse the first comparison stage.
    // For example, merging a list of size 8 has the exchanges:
    // 0 <-> 15, 1 <-> 14, ...
    K other_k = shfl_xor(k, 2 * L - 1);
    V other_v = shfl_xor(v, 2 * L - 1);

    // Whether we are the lesser thread in the exchange
    bool small = !(lane_id & L);

    if (Dir) {
      // See the comment above how performing both of these
      // comparisons in the warp seems to win out over the
      // alternatives in practice
      bool s = small ? Comp::gt(k, other_k) : Comp::lt(k, other_k);
      assign(s, k, other_k);
      assign(s, v, other_v);

    } else {
      bool s = small ? Comp::lt(k, other_k) : Comp::gt(k, other_k);
      assign(s, k, other_k);
      assign(s, v, other_v);
    }
  }

#pragma unroll
  for (int stride = IsBitonic ? L : L / 2; stride > 0; stride /= 2) {
    K other_k = shfl_xor(k, stride);
    V other_v = shfl_xor(v, stride);

    // Whether we are the lesser thread in the exchange
    bool small = !(lane_id & stride);

    if (Dir) {
      bool s = small ? Comp::gt(k, other_k) : Comp::lt(k, other_k);
      assign(s, k, other_k);
      assign(s, v, other_v);

    } else {
      bool s = small ? Comp::lt(k, other_k) : Comp::gt(k, other_k);
      assign(s, k, other_k);
      assign(s, v, other_v);
    }
  }
}

// Template for performing a bitonic merge of an arbitrary set of
// registers
template <typename K, typename V, int N, bool Dir, typename Comp, bool Low, bool Pow2>
struct bitonic_merge_step {};

//
// Power-of-2 merge specialization
//

// All merges eventually call this
template <typename K, typename V, bool Dir, typename Comp, bool Low>
struct bitonic_merge_step<K, V, 1, Dir, Comp, Low, true> {
  static inline __device__ void merge(K k[1], V v[1])
  {
    // Use warp shuffles
    warp_bitonic_merge_le16<K, V, 16, Dir, Comp, true>(k[0], v[0]);
  }
};

template <typename K, typename V, int N, bool Dir, typename Comp, bool Low>
struct bitonic_merge_step<K, V, N, Dir, Comp, Low, true> {
  static inline __device__ void merge(K k[N], V v[N])
  {
    static_assert(utils::is_power_of2(N), "must be power of 2");
    static_assert(N > 1, "must be N > 1");

#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      K& ka = k[i];
      V& va = v[i];

      K& kb = k[i + N / 2];
      V& vb = v[i + N / 2];

      bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
      swap(s, ka, kb);
      swap(s, va, vb);
    }

    {
      K new_k[N / 2];
      V new_v[N / 2];

#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        new_k[i] = k[i];
        new_v[i] = v[i];
      }

      bitonic_merge_step<K, V, N / 2, Dir, Comp, true, true>::merge(new_k, new_v);

#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        k[i] = new_k[i];
        v[i] = new_v[i];
      }
    }

    {
      K new_k[N / 2];
      V new_v[N / 2];

#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        new_k[i] = k[i + N / 2];
        new_v[i] = v[i + N / 2];
      }

      bitonic_merge_step<K, V, N / 2, Dir, Comp, false, true>::merge(new_k, new_v);

#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        k[i + N / 2] = new_k[i];
        v[i + N / 2] = new_v[i];
      }
    }
  }
};

//
// Non-power-of-2 merge specialization
//

// Low recursion
template <typename K, typename V, int N, bool Dir, typename Comp>
struct bitonic_merge_step<K, V, N, Dir, Comp, true, false> {
  static inline __device__ void merge(K k[N], V v[N])
  {
    static_assert(!utils::is_power_of2(N), "must be non-power-of-2");
    static_assert(N >= 3, "must be N >= 3");

    constexpr int kNextHighestPowerOf2 = utils::next_highest_power_of2(N);

#pragma unroll
    for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
      K& ka = k[i];
      V& va = v[i];

      K& kb = k[i + kNextHighestPowerOf2 / 2];
      V& vb = v[i + kNextHighestPowerOf2 / 2];

      bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
      swap(s, ka, kb);
      swap(s, va, vb);
    }

    constexpr int kLowSize  = N - kNextHighestPowerOf2 / 2;
    constexpr int kHighSize = kNextHighestPowerOf2 / 2;
    {
      K new_k[kLowSize];
      V new_v[kLowSize];

#pragma unroll
      for (int i = 0; i < kLowSize; ++i) {
        new_k[i] = k[i];
        new_v[i] = v[i];
      }

      constexpr bool kLowIsPowerOf2 = utils::is_power_of2(N - kNextHighestPowerOf2 / 2);
      // FIXME: compiler doesn't like this expression? compiler bug?
      //      constexpr bool kLowIsPowerOf2 = utils::is_power_of2(kLowSize);
      bitonic_merge_step<K,
                         V,
                         kLowSize,
                         Dir,
                         Comp,
                         true,  // low
                         kLowIsPowerOf2>::merge(new_k, new_v);

#pragma unroll
      for (int i = 0; i < kLowSize; ++i) {
        k[i] = new_k[i];
        v[i] = new_v[i];
      }
    }

    {
      K new_k[kHighSize];
      V new_v[kHighSize];

#pragma unroll
      for (int i = 0; i < kHighSize; ++i) {
        new_k[i] = k[i + kLowSize];
        new_v[i] = v[i + kLowSize];
      }

      constexpr bool kHighIsPowerOf2 = utils::is_power_of2(kNextHighestPowerOf2 / 2);
      // FIXME: compiler doesn't like this expression? compiler bug?
      //      constexpr bool kHighIsPowerOf2 =
      //      utils::is_power_of2(kHighSize);
      bitonic_merge_step<K,
                         V,
                         kHighSize,
                         Dir,
                         Comp,
                         false,  // high
                         kHighIsPowerOf2>::merge(new_k, new_v);

#pragma unroll
      for (int i = 0; i < kHighSize; ++i) {
        k[i + kLowSize] = new_k[i];
        v[i + kLowSize] = new_v[i];
      }
    }
  }
};

// High recursion
template <typename K, typename V, int N, bool Dir, typename Comp>
struct bitonic_merge_step<K, V, N, Dir, Comp, false, false> {
  static inline __device__ void merge(K k[N], V v[N])
  {
    static_assert(!utils::is_power_of2(N), "must be non-power-of-2");
    static_assert(N >= 3, "must be N >= 3");

    constexpr int kNextHighestPowerOf2 = utils::next_highest_power_of2(N);

#pragma unroll
    for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
      K& ka = k[i];
      V& va = v[i];

      K& kb = k[i + kNextHighestPowerOf2 / 2];
      V& vb = v[i + kNextHighestPowerOf2 / 2];

      bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
      swap(s, ka, kb);
      swap(s, va, vb);
    }

    constexpr int kLowSize  = kNextHighestPowerOf2 / 2;
    constexpr int kHighSize = N - kNextHighestPowerOf2 / 2;
    {
      K new_k[kLowSize];
      V new_v[kLowSize];

#pragma unroll
      for (int i = 0; i < kLowSize; ++i) {
        new_k[i] = k[i];
        new_v[i] = v[i];
      }

      constexpr bool kLowIsPowerOf2 = utils::is_power_of2(kNextHighestPowerOf2 / 2);
      // FIXME: compiler doesn't like this expression? compiler bug?
      //      constexpr bool kLowIsPowerOf2 = utils::is_power_of2(kLowSize);
      bitonic_merge_step<K,
                         V,
                         kLowSize,
                         Dir,
                         Comp,
                         true,  // low
                         kLowIsPowerOf2>::merge(new_k, new_v);

#pragma unroll
      for (int i = 0; i < kLowSize; ++i) {
        k[i] = new_k[i];
        v[i] = new_v[i];
      }
    }

    {
      K new_k[kHighSize];
      V new_v[kHighSize];

#pragma unroll
      for (int i = 0; i < kHighSize; ++i) {
        new_k[i] = k[i + kLowSize];
        new_v[i] = v[i + kLowSize];
      }

      constexpr bool kHighIsPowerOf2 = utils::is_power_of2(N - kNextHighestPowerOf2 / 2);
      // FIXME: compiler doesn't like this expression? compiler bug?
      //      constexpr bool kHighIsPowerOf2 =
      //      utils::is_power_of2(kHighSize);
      bitonic_merge_step<K,
                         V,
                         kHighSize,
                         Dir,
                         Comp,
                         false,  // high
                         kHighIsPowerOf2>::merge(new_k, new_v);

#pragma unroll
      for (int i = 0; i < kHighSize; ++i) {
        k[i + kLowSize] = new_k[i];
        v[i + kLowSize] = new_v[i];
      }
    }
  }
};

/// Merges two sets of registers across the warp of any size;
/// i.e., merges a sorted k/v list of size WarpSize * N1 with a
/// sorted k/v list of size WarpSize * N2, where N1 and N2 are any
/// value >= 1
template <typename K, typename V, int N1, int N2, bool Dir, typename Comp, bool FullMerge = true>
inline __device__ void warp_merge_any_registers(K k1[N1], V v1[N1], K k2[N2], V v2[N2])
{
  constexpr int kSmallestN = N1 < N2 ? N1 : N2;

#pragma unroll
  for (int i = 0; i < kSmallestN; ++i) {
    K& ka = k1[N1 - 1 - i];
    V& va = v1[N1 - 1 - i];

    K& kb = k2[i];
    V& vb = v2[i];

    K other_ka;
    V other_va;

    if (FullMerge) {
      // We need the other values
      other_ka = shfl_xor(ka, WarpSize - 1);
      other_va = shfl_xor(va, WarpSize - 1);
    }

    K other_kb = shfl_xor(kb, WarpSize - 1);
    V other_vb = shfl_xor(vb, WarpSize - 1);

    // ka is always first in the list, so we needn't use our lane
    // in this comparison
    bool swapa = Dir ? Comp::gt(ka, other_kb) : Comp::lt(ka, other_kb);
    assign(swapa, ka, other_kb);
    assign(swapa, va, other_vb);

    // kb is always second in the list, so we needn't use our lane
    // in this comparison
    if (FullMerge) {
      bool swapb = Dir ? Comp::lt(kb, other_ka) : Comp::gt(kb, other_ka);
      assign(swapb, kb, other_ka);
      assign(swapb, vb, other_va);

    } else {
      // We don't care about updating elements in the second list
    }
  }

  bitonic_merge_step<K, V, N1, Dir, Comp, true, utils::is_power_of2(N1)>::merge(k1, v1);
  if (FullMerge) {
    // Only if we care about N2 do we need to bother merging it fully
    bitonic_merge_step<K, V, N2, Dir, Comp, false, utils::is_power_of2(N2)>::merge(k2, v2);
  }
}

// Recursive template that uses the above bitonic merge to perform a
// bitonic sort
template <typename K, typename V, int N, bool Dir, typename Comp>
struct bitonic_sort_step {
  static inline __device__ void sort(K k[N], V v[N])
  {
    static_assert(N > 1, "did not hit specialized case");

    // Sort recursively
    constexpr int kSizeA = N / 2;
    constexpr int kSizeB = N - kSizeA;

    K a_k[kSizeA];
    V a_v[kSizeA];

#pragma unroll
    for (int i = 0; i < kSizeA; ++i) {
      a_k[i] = k[i];
      a_v[i] = v[i];
    }

    bitonic_sort_step<K, V, kSizeA, Dir, Comp>::sort(a_k, a_v);

    K b_k[kSizeB];
    V b_v[kSizeB];

#pragma unroll
    for (int i = 0; i < kSizeB; ++i) {
      b_k[i] = k[i + kSizeA];
      b_v[i] = v[i + kSizeA];
    }

    bitonic_sort_step<K, V, kSizeB, Dir, Comp>::sort(b_k, b_v);

    // Merge halves
    warp_merge_any_registers<K, V, kSizeA, kSizeB, Dir, Comp>(a_k, a_v, b_k, b_v);

#pragma unroll
    for (int i = 0; i < kSizeA; ++i) {
      k[i] = a_k[i];
      v[i] = a_v[i];
    }

#pragma unroll
    for (int i = 0; i < kSizeB; ++i) {
      k[i + kSizeA] = b_k[i];
      v[i + kSizeA] = b_v[i];
    }
  }
};

// Single warp (N == 1) sorting specialization
template <typename K, typename V, bool Dir, typename Comp>
struct bitonic_sort_step<K, V, 1, Dir, Comp> {
  static inline __device__ void sort(K k[1], V v[1])
  {
    // Update this code if this changes
    // should go from 1 -> WarpSize in multiples of 2
    static_assert(WarpSize == 32, "unexpected warp size");

    warp_bitonic_merge_le16<K, V, 1, Dir, Comp, false>(k[0], v[0]);
    warp_bitonic_merge_le16<K, V, 2, Dir, Comp, false>(k[0], v[0]);
    warp_bitonic_merge_le16<K, V, 4, Dir, Comp, false>(k[0], v[0]);
    warp_bitonic_merge_le16<K, V, 8, Dir, Comp, false>(k[0], v[0]);
    warp_bitonic_merge_le16<K, V, 16, Dir, Comp, false>(k[0], v[0]);
  }
};

/// Sort a list of WarpSize * N elements in registers, where N is an
/// arbitrary >= 1
template <typename K, typename V, int N, bool Dir, typename Comp>
inline __device__ void warp_sort_any_registers(K k[N], V v[N])
{
  bitonic_sort_step<K, V, N, Dir, Comp>::sort(k, v);
}

}  // namespace cuvs::neighbors::detail::faiss_select
