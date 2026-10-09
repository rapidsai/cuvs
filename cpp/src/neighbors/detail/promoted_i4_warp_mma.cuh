/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>

#include <mma.h>

#include <cstdint>
#include <type_traits>

// Warp-level tensor-core dot products of i4 codes (0..15, two per byte) staged in shared memory as
// rows of RowStride bytes. A warp owns a WarpTile x WarpTile block of the output: accumulate() adds
// the dot products over one staged K range, store() writes the raw sums as raw_t. The u8 (wmma)
// and fp8 (mma.sync) variants share this interface, so callers can switch between them.

namespace cuvs::neighbors::nn_descent::detail {

// Masks one nibble half of a packed-i4 fragment into `dst`, normalised to 0..15. The high half is
// shifted down so both halves reach the MMA at the same scale and can share one accumulator.
template <bool High, typename FragT>
__device__ __forceinline__ void select_packed_nibble(const FragT& src, FragT& dst)
{
  constexpr int kWords = FragT::num_elements / 4;
  static_assert(FragT::num_elements % 4 == 0, "fragment must be a whole number of u32 words");
  const auto* w = reinterpret_cast<const uint32_t*>(src.x);
  auto* d       = reinterpret_cast<uint32_t*>(dst.x);
#pragma unroll
  for (int i = 0; i < kWords; ++i) {
    d[i] = High ? ((w[i] >> 4) & 0x0f0f0f0fu) : (w[i] & 0x0f0f0f0fu);
  }
}

// The staged bytes are packed i4; the MMA reads them as u8 and splits the nibbles here rather than
// leaving it to ptxas. u4 MMA is native only through sm_89, and past that ptxas emits this same
// split out of line, once per MMA. Inlining it keeps the IMMAs in the loop body on every arch, and
// the 16x16x16 tile u8 allows drops SUB_M/SUB_N to 1.
template <int WarpTile, int RowStride>
struct i4_warp_mma_u8 {
  using raw_t = int;

  static constexpr int MMA_M = 16;
  static constexpr int MMA_N = 16;
  // A k-step reads 16 packed bytes = 32 codes; each nibble half is a separate MMA over MMA_K.
  static constexpr int MMA_K       = 16;
  static constexpr int KSTEP_BYTES = MMA_K;
  static constexpr int SUB_M       = WarpTile / MMA_M;
  static constexpr int SUB_N       = WarpTile / MMA_N;
  static_assert(WarpTile % MMA_M == 0 && WarpTile % MMA_N == 0,
                "the warp tile must be a whole number of native MMA tiles");
  static_assert(RowStride % 16 == 0, "row stride must preserve 16-byte IMMA row alignment");

  using a_frag_t = nvcuda::wmma::
    fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K, uint8_t, nvcuda::wmma::row_major>;
  using b_frag_t = nvcuda::wmma::
    fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N, MMA_K, uint8_t, nvcuda::wmma::col_major>;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MMA_M, MMA_N, MMA_K, int> c[SUB_M][SUB_N];

  __device__ __forceinline__ i4_warp_mma_u8()
  {
    for (int msub = 0; msub < SUB_M; ++msub) {
      for (int nsub = 0; nsub < SUB_N; ++nsub) {
        nvcuda::wmma::fill_fragment(c[msub][nsub], 0);
      }
    }
  }

  // a_frag depends only on (msub, kk); b_frag depends only on (nsub, kk) -- load each once per kk
  // and reuse across the other sub-tile index, instead of reloading redundantly inside a full
  // msub x nsub x kk cross product.
  // Kept rolled: unrolling the k-steps keeps more fragment live ranges simultaneous and costs a
  // block per SM in registers, for some intra-warp ILP.
  template <int RowBytes>
  __device__ __forceinline__ void accumulate(const uint8_t (*a)[RowStride],
                                             const uint8_t (*b)[RowStride],
                                             int row0,
                                             int col0)
  {
#pragma unroll 1
    for (int kk = 0; kk < RowBytes / KSTEP_BYTES; ++kk) {
      a_frag_t a_frag[SUB_M];
      b_frag_t b_frag[SUB_N];
      for (int msub = 0; msub < SUB_M; ++msub) {
        nvcuda::wmma::load_matrix_sync(
          a_frag[msub], a[row0 + msub * MMA_M] + kk * KSTEP_BYTES, RowStride);
      }
      for (int nsub = 0; nsub < SUB_N; ++nsub) {
        nvcuda::wmma::load_matrix_sync(
          b_frag[nsub], b[col0 + nsub * MMA_N] + kk * KSTEP_BYTES, RowStride);
      }
      // One scratch fragment per operand, reused across the halves.
      a_frag_t a_half[SUB_M];
      b_frag_t b_half[SUB_N];
      auto accumulate_half = [&](auto high_tag) {
        constexpr bool High = decltype(high_tag)::value;
        for (int msub = 0; msub < SUB_M; ++msub) {
          select_packed_nibble<High>(a_frag[msub], a_half[msub]);
        }
        for (int nsub = 0; nsub < SUB_N; ++nsub) {
          select_packed_nibble<High>(b_frag[nsub], b_half[nsub]);
        }
        for (int msub = 0; msub < SUB_M; ++msub) {
          for (int nsub = 0; nsub < SUB_N; ++nsub) {
            nvcuda::wmma::mma_sync(c[msub][nsub], a_half[msub], b_half[nsub], c[msub][nsub]);
          }
        }
      };
      accumulate_half(std::false_type{});
      accumulate_half(std::true_type{});
    }
  }

  template <int Stride>
  __device__ __forceinline__ void store(raw_t* dst, int row0, int col0) const
  {
    for (int msub = 0; msub < SUB_M; ++msub) {
      for (int nsub = 0; nsub < SUB_N; ++nsub) {
        nvcuda::wmma::store_matrix_sync(dst + (row0 + msub * MMA_M) * Stride + col0 + nsub * MMA_N,
                                        c[msub][nsub],
                                        Stride,
                                        nvcuda::wmma::mem_row_major);
      }
    }
  }
};

// fp8 mma.sync needs sm_89+.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
// Expands the eight 4-bit codes packed in w to e4m3 bytes of the same value (0..15 are exact in
// e4m3): lo gets the codes in w's low 16 bits, hi the codes in its high 16 bits. prmt tables hold
// only 8 bytes, so 0..7 and 8..15 use separate tables; a selector with bit 3 set yields the
// selected byte's sign, 0 here, which zeroes the lanes the other table owns. prmt reads only the
// selector's low 16 bits, hence the shift for hi.
__device__ __forceinline__ void nibbles_to_e4m3(uint32_t w, uint32_t& lo, uint32_t& hi)
{
  constexpr uint32_t k0_3   = 0x44403800u;
  constexpr uint32_t k4_7   = 0x4e4c4a48u;
  constexpr uint32_t k8_11  = 0x53525150u;
  constexpr uint32_t k12_15 = 0x57565554u;
  auto lookup               = [](uint32_t sel) {
    uint32_t below_8, from_8;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(below_8) : "r"(k0_3), "r"(k4_7), "r"(sel));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(from_8) : "r"(k8_11), "r"(k12_15), "r"(sel ^ 0x8888u));
    return below_8 | from_8;
  };
  lo = lookup(w);
  hi = lookup(w >> 16);
}

// c += a * b over one m16n8k32 tile. The tensor core's fp8 accumulator keeps far fewer bits than
// f32, so chaining c through it drops low bits of the integer sums. Each MMA starts from zero
// instead -- one tile's sum fits exactly -- and the running sum is kept with ordinary f32 adds.
__device__ __forceinline__ void mma_e4m3_m16n8k32(float (&c)[4],
                                                  const uint32_t (&a)[4],
                                                  const uint32_t (&b)[2])
{
  float d[4];
  asm(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.0f));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c[i] += d[i];
  }
}

// Feeds the 4-bit codes to mma.sync as e4m3, expanded in registers from plain 32-bit LDS.
template <int WarpTile, int RowStride>
struct i4_warp_mma_fp8 {
  using raw_t = float;

  static constexpr int MMA_M = 16;
  static constexpr int MMA_N = 8;
  // A k-step reads 16 packed bytes = 32 codes, expanded to e4m3 and consumed by one MMA.
  static constexpr int KSTEP_BYTES = 16;
  static constexpr int SUB_M       = WarpTile / MMA_M;
  static constexpr int SUB_N       = WarpTile / MMA_N;
  static_assert(WarpTile % MMA_M == 0 && WarpTile % MMA_N == 0,
                "the warp tile must be a whole number of native MMA tiles");
  static_assert(RowStride % 4 == 0, "rows must stay 4-byte aligned for the u32 fragment loads");

  float c[SUB_M][SUB_N][4] = {};

  // m16n8k32 fragment coordinates: `group` is the A row (and +8) and the B column this lane holds;
  // `quad` picks its 4-byte slice of the k-step. A and B read the same slice, and expanding it to
  // (low 16 bits -> k 0..15, high -> k 16..31) the same way keeps their k orders aligned.
  // Kept rolled for the same register reason as the u8 path.
  template <int RowBytes>
  __device__ __forceinline__ void accumulate(const uint8_t (*a)[RowStride],
                                             const uint8_t (*b)[RowStride],
                                             int row0,
                                             int col0)
  {
    const int lane  = threadIdx.x % raft::warp_size();
    const int group = lane / 4;
    const int quad  = lane % 4;
#pragma unroll 1
    for (int kk = 0; kk < RowBytes / KSTEP_BYTES; ++kk) {
      const int byte0 = kk * KSTEP_BYTES + quad * 4;
      uint32_t a_frag[SUB_M][4];
      uint32_t b_frag[SUB_N][2];
      for (int msub = 0; msub < SUB_M; ++msub) {
        const int row     = row0 + msub * MMA_M + group;
        const uint32_t r0 = *reinterpret_cast<const uint32_t*>(a[row] + byte0);
        const uint32_t r8 = *reinterpret_cast<const uint32_t*>(a[row + 8] + byte0);
        nibbles_to_e4m3(r0, a_frag[msub][0], a_frag[msub][2]);
        nibbles_to_e4m3(r8, a_frag[msub][1], a_frag[msub][3]);
      }
      for (int nsub = 0; nsub < SUB_N; ++nsub) {
        const uint32_t w =
          *reinterpret_cast<const uint32_t*>(b[col0 + nsub * MMA_N + group] + byte0);
        nibbles_to_e4m3(w, b_frag[nsub][0], b_frag[nsub][1]);
      }
      for (int msub = 0; msub < SUB_M; ++msub) {
        for (int nsub = 0; nsub < SUB_N; ++nsub) {
          mma_e4m3_m16n8k32(c[msub][nsub], a_frag[msub], b_frag[nsub]);
        }
      }
    }
  }

  // Lane (group, quad) holds columns quad*2, +1 of rows group and group+8.
  template <int Stride>
  __device__ __forceinline__ void store(raw_t* dst, int row0, int col0) const
  {
    static_assert(Stride % 2 == 0, "float2 stores need even row strides");
    const int lane  = threadIdx.x % raft::warp_size();
    const int group = lane / 4;
    const int quad  = lane % 4;
    for (int msub = 0; msub < SUB_M; ++msub) {
      for (int nsub = 0; nsub < SUB_N; ++nsub) {
        for (int h = 0; h < 2; ++h) {
          const int row = row0 + msub * MMA_M + group + h * 8;
          const int col = col0 + nsub * MMA_N + quad * 2;
          *reinterpret_cast<float2*>(dst + row * Stride + col) =
            make_float2(c[msub][nsub][h * 2], c[msub][nsub][h * 2 + 1]);
        }
      }
    }
  }
};
#endif

}  // namespace cuvs::neighbors::nn_descent::detail
