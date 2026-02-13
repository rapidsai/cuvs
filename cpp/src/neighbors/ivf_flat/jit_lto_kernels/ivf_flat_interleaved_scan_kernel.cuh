/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../ivf_common.cuh"

#include <cuvs/neighbors/ivf_flat.hpp>

#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/util/device_loads_stores.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/vectorized.cuh>

// This header contains the kernel definition and should only be included
// when compiling JIT-LTO kernel fragments (when BUILD_KERNEL is defined).

namespace cuvs::neighbors::ivf_flat::detail {

static constexpr int kThreadsPerBlock = 128;

// These extern device functions are linked at runtime using JIT-LTO.
template <int Veclen, typename T, typename AccT>
extern __device__ void compute_dist(AccT& acc, AccT x, AccT y);

template <typename index_t>
extern __device__ bool sample_filter(index_t* const* const inds_ptrs,
                                     const uint32_t query_ix,
                                     const uint32_t cluster_ix,
                                     const uint32_t sample_ix,
                                     uint32_t* bitset_ptr,
                                     index_t bitset_len,
                                     index_t original_nbits);

template <typename T>
extern __device__ T post_process(T val);

/**
 * @brief Load a part of a vector from the index and from query, compute the (part of the) distance
 * between them, and aggregate it using the provided Lambda; one structure per thread, per query,
 * and per index item.
 *
 * @tparam kUnroll elements per loop (normally, kUnroll = WarpSize / Veclen)
 * @tparam Lambda computing the part of the distance for one dimension and aggregating it:
 *                void (AccT& acc, AccT x, AccT y)
 * @tparam Veclen size of the vectorized load
 * @tparam T type of the data in the query and the index
 * @tparam AccT type of the accumulated value (an optimization for 8bit values to be loaded as 32bit
 * values)
 */
template <int kUnroll, int Veclen, typename T, typename AccT, bool ComputeNorm>
struct loadAndComputeDist {
  AccT& dist;
  AccT& norm_query;
  AccT& norm_data;

  __device__ __forceinline__ loadAndComputeDist(AccT& dist, AccT& norm_query, AccT& norm_data)
    : dist(dist), norm_query(norm_query), norm_data(norm_data)
  {
  }

  /**
   * Load parts of vectors from the index and query and accumulates the partial distance.
   * This version assumes the query is stored in shared memory.
   * Every thread here processes exactly kUnroll * Veclen elements independently of others.
   */
  template <typename IdxT>
  __device__ __forceinline__ void runLoadShmemCompute(const T* const& data,
                                                      const T* query_shared,
                                                      IdxT loadIndex,
                                                      IdxT shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      T encV[Veclen];
      raft::ldg(encV, data + (loadIndex + j * kIndexGroupSize) * Veclen);
      T queryRegs[Veclen];
      raft::lds(queryRegs, &query_shared[shmemIndex + j * Veclen]);
#pragma unroll
      for (int k = 0; k < Veclen; ++k) {
        compute_dist<Veclen, T, AccT>(dist, queryRegs[k], encV[k]);
        if constexpr (ComputeNorm) {
          norm_query += queryRegs[k] * queryRegs[k];
          norm_data += encV[k] * encV[k];
        }
      }
    }
  }

  /**
   * Load parts of vectors from the index and query and accumulates the partial distance.
   * This version assumes the query is stored in the global memory and is different for every
   * thread. One warp loads exactly WarpSize query elements at once and then reshuffles them into
   * corresponding threads (`WarpSize / (kUnroll * Veclen)` elements per thread at once).
   */
  template <typename IdxT>
  __device__ __forceinline__ void runLoadShflAndCompute(const T*& data,
                                                        const T* query,
                                                        IdxT baseLoadIndex,
                                                        const int lane_id)
  {
    T queryReg               = query[baseLoadIndex + lane_id];
    constexpr int stride     = kUnroll * Veclen;
    constexpr int totalIter  = raft::WarpSize / stride;
    constexpr int gmemStride = stride * kIndexGroupSize;
#pragma unroll
    for (int i = 0; i < totalIter; ++i, data += gmemStride) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        T encV[Veclen];
        raft::ldg(encV, data + (lane_id + j * kIndexGroupSize) * Veclen);
        const int d = (i * kUnroll + j) * Veclen;
#pragma unroll
        for (int k = 0; k < Veclen; ++k) {
          T q = raft::shfl(queryReg, d + k, raft::WarpSize);
          compute_dist<Veclen, T, AccT>(dist, q, encV[k]);
          if constexpr (ComputeNorm) {
            norm_query += q * q;
            norm_data += encV[k] * encV[k];
          }
        }
      }
    }
  }

  /**
   * Load parts of vectors from the index and query and accumulates the partial distance.
   * This version augments `runLoadShflAndCompute` when `dim` is not a multiple of `WarpSize`.
   */
  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const T*& data, const T* query, const int lane_id, const int dim, const int dimBlocks)
  {
    const int loadDim     = dimBlocks + lane_id;
    T queryReg            = loadDim < dim ? query[loadDim] : T{0};
    const int loadDataIdx = lane_id * Veclen;
    for (int d = 0; d < dim - dimBlocks; d += Veclen, data += kIndexGroupSize * Veclen) {
      T enc[Veclen];
      raft::ldg(enc, data + loadDataIdx);
#pragma unroll
      for (int k = 0; k < Veclen; k++) {
        T q = raft::shfl(queryReg, d + k, raft::WarpSize);
        compute_dist<Veclen, T, AccT>(dist, q, enc[k]);
        if constexpr (ComputeNorm) {
          norm_query += q * q;
          norm_data += enc[k] * enc[k];
        }
      }
    }
  }
};

// This handles uint8_t 8, 16 Veclens
template <int kUnroll, int uint8_veclen, bool ComputeNorm>
struct loadAndComputeDist<kUnroll, uint8_veclen, uint8_t, uint32_t, ComputeNorm> {
  uint32_t& dist;
  uint32_t& norm_query;
  uint32_t& norm_data;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist,
                                                uint32_t& norm_query,
                                                uint32_t& norm_data)
    : dist(dist), norm_query(norm_query), norm_data(norm_data)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
    constexpr int veclen_int = uint8_veclen / 4;  // converting uint8_t veclens to int
    loadIndex                = loadIndex * veclen_int;
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t encV[veclen_int];
      raft::ldg(
        encV,
        reinterpret_cast<unsigned const*>(data) + loadIndex + j * kIndexGroupSize * veclen_int);
      uint32_t queryRegs[veclen_int];
      raft::lds(queryRegs,
                reinterpret_cast<unsigned const*>(query_shared + shmemIndex) + j * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        compute_dist<uint8_veclen, uint8_t, uint32_t>(dist, queryRegs[k], encV[k]);
        if constexpr (ComputeNorm) {
          norm_query = raft::dp4a(queryRegs[k], queryRegs[k], norm_query);
          norm_data  = raft::dp4a(encV[k], encV[k], norm_data);
        }
      }
    }
  }
  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    constexpr int veclen_int = uint8_veclen / 4;  // converting uint8_t veclens to int
    uint32_t queryReg =
      (lane_id < 8) ? reinterpret_cast<unsigned const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int stride = kUnroll * uint8_veclen;

#pragma unroll
    for (int i = 0; i < raft::WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        uint32_t encV[veclen_int];
        raft::ldg(
          encV,
          reinterpret_cast<unsigned const*>(data) + (lane_id + j * kIndexGroupSize) * veclen_int);
        const int d = (i * kUnroll + j) * veclen_int;
#pragma unroll
        for (int k = 0; k < veclen_int; ++k) {
          uint32_t q = raft::shfl(queryReg, d + k, raft::WarpSize);
          compute_dist<uint8_veclen, uint8_t, uint32_t>(dist, q, encV[k]);
          if constexpr (ComputeNorm) {
            norm_query = raft::dp4a(q, q, norm_query);
            norm_data  = raft::dp4a(encV[k], encV[k], norm_data);
          }
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int lane_id,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen_int = uint8_veclen / 4;
    const int loadDim        = dimBlocks + lane_id * 4;  // Here 4 is for 1 - int
    uint32_t queryReg = loadDim < dim ? reinterpret_cast<uint32_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks;
         d += uint8_veclen, data += kIndexGroupSize * uint8_veclen) {
      uint32_t enc[veclen_int];
      raft::ldg(enc, reinterpret_cast<uint32_t const*>(data) + lane_id * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        uint32_t q = raft::shfl(queryReg, (d / 4) + k, raft::WarpSize);
        compute_dist<uint8_veclen, uint8_t, uint32_t>(dist, q, enc[k]);
        if constexpr (ComputeNorm) {
          norm_query = raft::dp4a(q, q, norm_query);
          norm_data  = raft::dp4a(enc[k], enc[k], norm_data);
        }
      }
    }
  }
};

// Keep this specialized uint8 Veclen = 4, because compiler is generating suboptimal code while
// using above common template of int2/int4
template <int kUnroll, bool ComputeNorm>
struct loadAndComputeDist<kUnroll, 4, uint8_t, uint32_t, ComputeNorm> {
  uint32_t& dist;
  uint32_t& norm_query;
  uint32_t& norm_data;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist,
                                                uint32_t& norm_query,
                                                uint32_t& norm_data)
    : dist(dist), norm_query(norm_query), norm_data(norm_data)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t encV      = reinterpret_cast<unsigned const*>(data)[loadIndex + j * kIndexGroupSize];
      uint32_t queryRegs = reinterpret_cast<unsigned const*>(query_shared + shmemIndex)[j];
      compute_dist<4, uint8_t, uint32_t>(dist, queryRegs, encV);
      if constexpr (ComputeNorm) {
        norm_query = raft::dp4a(queryRegs, queryRegs, norm_query);
        norm_data  = raft::dp4a(encV, encV, norm_data);
      }
    }
  }
  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    uint32_t queryReg =
      (lane_id < 8) ? reinterpret_cast<unsigned const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int veclen = 4;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < raft::WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        uint32_t encV = reinterpret_cast<unsigned const*>(data)[lane_id + j * kIndexGroupSize];
        uint32_t q    = raft::shfl(queryReg, i * kUnroll + j, raft::WarpSize);
        compute_dist<4, uint8_t, uint32_t>(dist, q, encV);
        if constexpr (ComputeNorm) {
          norm_query = raft::dp4a(q, q, norm_query);
          norm_data  = raft::dp4a(encV, encV, norm_data);
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int lane_id,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen = 4;
    const int loadDim    = dimBlocks + lane_id;
    uint32_t queryReg    = loadDim < dim ? reinterpret_cast<unsigned const*>(query)[loadDim] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      uint32_t enc = reinterpret_cast<unsigned const*>(data)[lane_id];
      uint32_t q   = raft::shfl(queryReg, d / veclen, raft::WarpSize);
      compute_dist<4, uint8_t, uint32_t>(dist, q, enc);
      if constexpr (ComputeNorm) {
        norm_query = raft::dp4a(q, q, norm_query);
        norm_data  = raft::dp4a(enc, enc, norm_data);
      }
    }
  }
};

template <int kUnroll, bool ComputeNorm>
struct loadAndComputeDist<kUnroll, 2, uint8_t, uint32_t, ComputeNorm> {
  uint32_t& dist;
  uint32_t& norm_query;
  uint32_t& norm_data;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist,
                                                uint32_t& norm_query,
                                                uint32_t& norm_data)
    : dist(dist), norm_query(norm_query), norm_data(norm_data)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t encV      = reinterpret_cast<uint16_t const*>(data)[loadIndex + j * kIndexGroupSize];
      uint32_t queryRegs = reinterpret_cast<uint16_t const*>(query_shared + shmemIndex)[j];
      compute_dist<2, uint8_t, uint32_t>(dist, queryRegs, encV);
      if constexpr (ComputeNorm) {
        norm_query = raft::dp4a(queryRegs, queryRegs, norm_query);
        norm_data  = raft::dp4a(encV, encV, norm_data);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    uint32_t queryReg =
      (lane_id < 16) ? reinterpret_cast<uint16_t const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int veclen = 2;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < raft::WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        uint32_t encV = reinterpret_cast<uint16_t const*>(data)[lane_id + j * kIndexGroupSize];
        uint32_t q    = raft::shfl(queryReg, i * kUnroll + j, raft::WarpSize);
        compute_dist<2, uint8_t, uint32_t>(dist, q, encV);
        if constexpr (ComputeNorm) {
          norm_query = raft::dp4a(q, q, norm_query);
          norm_data  = raft::dp4a(encV, encV, norm_data);
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int lane_id,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen = 2;
    int loadDim          = dimBlocks + lane_id * veclen;
    uint32_t queryReg = loadDim < dim ? reinterpret_cast<uint16_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      uint32_t enc = reinterpret_cast<uint16_t const*>(data)[lane_id];
      uint32_t q   = raft::shfl(queryReg, d / veclen, raft::WarpSize);
      compute_dist<2, uint8_t, uint32_t>(dist, q, enc);
      if constexpr (ComputeNorm) {
        norm_query = raft::dp4a(q, q, norm_query);
        norm_data  = raft::dp4a(enc, enc, norm_data);
      }
    }
  }
};

template <int kUnroll, bool ComputeNorm>
struct loadAndComputeDist<kUnroll, 1, uint8_t, uint32_t, ComputeNorm> {
  uint32_t& dist;
  uint32_t& norm_query;
  uint32_t& norm_data;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist,
                                                uint32_t& norm_query,
                                                uint32_t& norm_data)
    : dist(dist), norm_query(norm_query), norm_data(norm_data)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t encV      = data[loadIndex + j * kIndexGroupSize];
      uint32_t queryRegs = query_shared[shmemIndex + j];
      compute_dist<1, uint8_t, uint32_t>(dist, queryRegs, encV);
      if constexpr (ComputeNorm) {
        norm_query += queryRegs * queryRegs;
        norm_data += encV * encV;
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    uint32_t queryReg    = query[baseLoadIndex + lane_id];
    constexpr int veclen = 1;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < raft::WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        uint32_t encV = data[lane_id + j * kIndexGroupSize];
        uint32_t q    = raft::shfl(queryReg, i * kUnroll + j, raft::WarpSize);
        compute_dist<1, uint8_t, uint32_t>(dist, q, encV);
        if constexpr (ComputeNorm) {
          norm_query += q * q;
          norm_data += encV * encV;
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int lane_id,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen = 1;
    int loadDim          = dimBlocks + lane_id;
    uint32_t queryReg    = loadDim < dim ? query[loadDim] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      uint32_t enc = data[lane_id];
      uint32_t q   = raft::shfl(queryReg, d, raft::WarpSize);
      compute_dist<1, uint8_t, uint32_t>(dist, q, enc);
      if constexpr (ComputeNorm) {
        norm_query += q * q;
        norm_data += enc * enc;
      }
    }
  }
};

// This device function is for int8 veclens 4, 8 and 16
template <int kUnroll, int int8_veclen, bool ComputeNorm>
struct loadAndComputeDist<kUnroll, int8_veclen, int8_t, int32_t, ComputeNorm> {
  int32_t& dist;
  int32_t& norm_query;
  int32_t& norm_data;

  __device__ __forceinline__ loadAndComputeDist(int32_t& dist,
                                                int32_t& norm_query,
                                                int32_t& norm_data)
    : dist(dist), norm_query(norm_query), norm_data(norm_data)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const int8_t* const& data,
                                                      const int8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
    constexpr int veclen_int = int8_veclen / 4;  // converting int8_t veclens to int

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      int32_t encV[veclen_int];
      raft::ldg(
        encV,
        reinterpret_cast<int32_t const*>(data) + (loadIndex + j * kIndexGroupSize) * veclen_int);
      int32_t queryRegs[veclen_int];
      raft::lds(queryRegs,
                reinterpret_cast<int32_t const*>(query_shared + shmemIndex) + j * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        compute_dist<int8_veclen, int8_t, int32_t>(dist, queryRegs[k], encV[k]);
        if constexpr (ComputeNorm) {
          norm_query = raft::dp4a(queryRegs[k], queryRegs[k], norm_query);
          norm_data  = raft::dp4a(encV[k], encV[k], norm_data);
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const int8_t*& data,
                                                        const int8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    constexpr int veclen_int = int8_veclen / 4;  // converting int8_t veclens to int

    int32_t queryReg =
      (lane_id < 8) ? reinterpret_cast<int32_t const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int stride = kUnroll * int8_veclen;

#pragma unroll
    for (int i = 0; i < raft::WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        int32_t encV[veclen_int];
        raft::ldg(
          encV,
          reinterpret_cast<int32_t const*>(data) + (lane_id + j * kIndexGroupSize) * veclen_int);
        const int d = (i * kUnroll + j) * veclen_int;
#pragma unroll
        for (int k = 0; k < veclen_int; ++k) {
          int32_t q = raft::shfl(queryReg, d + k, raft::WarpSize);
          compute_dist<int8_veclen, int8_t, int32_t>(dist, q, encV[k]);
          if constexpr (ComputeNorm) {
            norm_query = raft::dp4a(q, q, norm_query);
            norm_data  = raft::dp4a(encV[k], encV[k], norm_data);
          }
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const int8_t*& data, const int8_t* query, const int lane_id, const int dim, const int dimBlocks)
  {
    constexpr int veclen_int = int8_veclen / 4;
    const int loadDim        = dimBlocks + lane_id * 4;  // Here 4 is for 1 - int;
    int32_t queryReg = loadDim < dim ? reinterpret_cast<int32_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks; d += int8_veclen, data += kIndexGroupSize * int8_veclen) {
      int32_t enc[veclen_int];
      raft::ldg(enc, reinterpret_cast<int32_t const*>(data) + lane_id * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        int32_t q = raft::shfl(queryReg, (d / 4) + k, raft::WarpSize);  // Here 4 is for 1 - int;
        compute_dist<int8_veclen, int8_t, int32_t>(dist, q, enc[k]);
        if constexpr (ComputeNorm) {
          norm_query = raft::dp4a(q, q, norm_query);
          norm_data  = raft::dp4a(enc[k], enc[k], norm_data);
        }
      }
    }
  }
};

template <int kUnroll, bool ComputeNorm>
struct loadAndComputeDist<kUnroll, 2, int8_t, int32_t, ComputeNorm> {
  int32_t& dist;
  int32_t& norm_query;
  int32_t& norm_data;
  __device__ __forceinline__ loadAndComputeDist(int32_t& dist,
                                                int32_t& norm_query,
                                                int32_t& norm_data)
    : dist(dist), norm_query(norm_query), norm_data(norm_data)
  {
  }
  __device__ __forceinline__ void runLoadShmemCompute(const int8_t* const& data,
                                                      const int8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      int32_t encV      = reinterpret_cast<uint16_t const*>(data)[loadIndex + j * kIndexGroupSize];
      int32_t queryRegs = reinterpret_cast<uint16_t const*>(query_shared + shmemIndex)[j];
      compute_dist<2, int8_t, int32_t>(dist, queryRegs, encV);
      if constexpr (ComputeNorm) {
        norm_query = raft::dp4a(queryRegs, queryRegs, norm_query);
        norm_data  = raft::dp4a(encV, encV, norm_data);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const int8_t*& data,
                                                        const int8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    int32_t queryReg =
      (lane_id < 16) ? reinterpret_cast<uint16_t const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int veclen = 2;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < raft::WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        int32_t encV = reinterpret_cast<uint16_t const*>(data)[lane_id + j * kIndexGroupSize];
        int32_t q    = raft::shfl(queryReg, i * kUnroll + j, raft::WarpSize);
        compute_dist<2, int8_t, int32_t>(dist, q, encV);
        if constexpr (ComputeNorm) {
          norm_query = raft::dp4a(q, q, norm_query);
          norm_data  = raft::dp4a(encV, encV, norm_data);
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const int8_t*& data, const int8_t* query, const int lane_id, const int dim, const int dimBlocks)
  {
    constexpr int veclen = 2;
    int loadDim          = dimBlocks + lane_id * veclen;
    int32_t queryReg = loadDim < dim ? reinterpret_cast<uint16_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      int32_t enc = reinterpret_cast<uint16_t const*>(data + lane_id * veclen)[0];
      int32_t q   = raft::shfl(queryReg, d / veclen, raft::WarpSize);
      compute_dist<2, int8_t, int32_t>(dist, q, enc);
      if constexpr (ComputeNorm) {
        norm_query = raft::dp4a(q, q, norm_query);
        norm_data  = raft::dp4a(enc, enc, norm_data);
      }
    }
  }
};

template <int kUnroll, bool ComputeNorm>
struct loadAndComputeDist<kUnroll, 1, int8_t, int32_t, ComputeNorm> {
  int32_t& dist;
  int32_t& norm_query;
  int32_t& norm_data;
  __device__ __forceinline__ loadAndComputeDist(int32_t& dist,
                                                int32_t& norm_query,
                                                int32_t& norm_data)
    : dist(dist), norm_query(norm_query), norm_data(norm_data)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const int8_t* const& data,
                                                      const int8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      compute_dist<1, int8_t, int32_t>(
        dist, query_shared[shmemIndex + j], data[loadIndex + j * kIndexGroupSize]);
      if constexpr (ComputeNorm) {
        norm_query += int32_t{query_shared[shmemIndex + j]} * int32_t{query_shared[shmemIndex + j]};
        norm_data += int32_t{data[loadIndex + j * kIndexGroupSize]} *
                     int32_t{data[loadIndex + j * kIndexGroupSize]};
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const int8_t*& data,
                                                        const int8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    constexpr int veclen = 1;
    constexpr int stride = kUnroll * veclen;
    int32_t queryReg     = query[baseLoadIndex + lane_id];

#pragma unroll
    for (int i = 0; i < raft::WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        int32_t q = raft::shfl(queryReg, i * kUnroll + j, raft::WarpSize);
        compute_dist<1, int8_t, int32_t>(dist, q, data[lane_id + j * kIndexGroupSize]);
        if constexpr (ComputeNorm) {
          norm_query += q * q;
          norm_data += data[lane_id + j * kIndexGroupSize] * data[lane_id + j * kIndexGroupSize];
        }
      }
    }
  }
  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const int8_t*& data, const int8_t* query, const int lane_id, const int dim, const int dimBlocks)
  {
    constexpr int veclen = 1;
    const int loadDim    = dimBlocks + lane_id;
    int32_t queryReg     = loadDim < dim ? query[loadDim] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      int32_t q = raft::shfl(queryReg, d, raft::WarpSize);
      compute_dist<1, int8_t, int32_t>(dist, q, data[lane_id]);
      if constexpr (ComputeNorm) {
        norm_query += q * q;
        norm_data += int32_t{data[lane_id]} * int32_t{data[lane_id]};
      }
    }
  }
};

// switch to dummy blocksort when Capacity is 0 this explicit dummy is chosen
// to support access to warpsort constants like ::queue_t::kDummy
template <int Capacity, bool Ascending, typename T, typename IdxT>
struct flat_block_sort {
  using type = raft::matrix::detail::select::warpsort::block_sort<
    raft::matrix::detail::select::warpsort::warp_sort_filtered,
    Capacity,
    Ascending,
    T,
    IdxT>;
};

template <typename T, bool Ascending, typename IdxT>
struct flat_block_sort<0, Ascending, T, IdxT>
  : ivf::detail::dummy_block_sort_t<T, IdxT, Ascending> {
  using type = ivf::detail::dummy_block_sort_t<T, IdxT, Ascending>;
};

template <int Capacity, bool Ascending, typename T, typename IdxT>
using block_sort_t = typename flat_block_sort<Capacity, Ascending, T, IdxT>::type;

/**
 * Scan clusters for nearest neighbors of the query vectors.
 * See `ivfflat_interleaved_scan` for more information.
 *
 * The clusters are stored in the interleaved index format described in ivf_flat_types.hpp.
 * For each query vector, a set of clusters is probed: the distance to each vector in the cluster is
 * calculated, and the top-k nearest neighbors are selected.
 *
 * @param compute_dist distance function
 * @param query_smem_elems number of dimensions of the query vector to fit in a shared memory of a
 * block; this number must be a multiple of `WarpSize * Veclen`.
 * @param[in] query a pointer to all queries in a row-major contiguous format [gridDim.y, dim]
 * @param[in] coarse_index a pointer to the cluster indices to search through [n_probes]
 * @param[in] list_indices index<T, IdxT>.indices
 * @param[in] list_data index<T, IdxT>.data
 * @param[in] list_sizes index<T, IdxT>.list_sizes
 * @param[in] list_offsets index<T, IdxT>.list_offsets
 * @param n_probes
 * @param k
 * @param dim
 * @param sample_filter
 * @param[out] neighbors
 * @param[out] distances
 */
template <int Capacity,
          int Veclen,
          bool Ascending,
          bool ComputeNorm,
          typename T,
          typename AccT,
          typename IdxT>
RAFT_KERNEL __launch_bounds__(kThreadsPerBlock)
  interleaved_scan_kernel(const uint32_t query_smem_elems,
                          const T* query,
                          const uint32_t* coarse_index,
                          const T* const* list_data_ptrs,
                          const uint32_t* list_sizes,
                          const uint32_t queries_offset,
                          const uint32_t n_probes,
                          const uint32_t k,
                          const uint32_t max_samples,
                          const uint32_t* chunk_indices,
                          const uint32_t dim,
                          IdxT* const* const inds_ptrs,
                          uint32_t* bitset_ptr,
                          IdxT bitset_len,
                          IdxT original_nbits,
                          uint32_t* neighbors,
                          float* distances)
{
  extern __shared__ __align__(256) uint8_t interleaved_scan_kernel_smem[];
  constexpr bool kManageLocalTopK = Capacity > 0;
  // Using shared memory for the (part of the) query;
  // This allows to save on global memory bandwidth when reading index and query
  // data at the same time.
  // Its size is `query_smem_elems`.
  T* query_shared = reinterpret_cast<T*>(interleaved_scan_kernel_smem);
  // Make the query input and output point to this block's shared query
  {
    const int query_id = blockIdx.y;
    query += query_id * dim;
    if constexpr (kManageLocalTopK) {
      neighbors += query_id * k * gridDim.x + blockIdx.x * k;
      distances += query_id * k * gridDim.x + blockIdx.x * k;
    } else {
      distances += query_id * uint64_t(max_samples);
    }
    chunk_indices += (n_probes * query_id);
    coarse_index += query_id * n_probes;
  }

  // Copy a part of the query into shared memory for faster processing
  copy_vectorized(query_shared, query, std::min(dim, query_smem_elems));
  __syncthreads();

  using local_topk_t = block_sort_t<Capacity, Ascending, float, uint32_t>;
  local_topk_t queue(k);
  {
    using align_warp  = raft::Pow2<raft::WarpSize>;
    const int lane_id = align_warp::mod(threadIdx.x);

    // How many full warps needed to compute the distance (without remainder)
    const uint32_t full_warps_along_dim = align_warp::roundDown(dim);

    const uint32_t shm_assisted_dim =
      (dim > query_smem_elems) ? query_smem_elems : full_warps_along_dim;

    // Every CUDA block scans one cluster at a time.
    for (int probe_id = blockIdx.x; probe_id < n_probes; probe_id += gridDim.x) {
      const uint32_t list_id = coarse_index[probe_id];  // The id of cluster(list)

      // The number of vectors in each cluster(list); [nlist]
      const uint32_t list_length = list_sizes[list_id];

      // The number of interleaved groups to be processed
      const uint32_t num_groups =
        align_warp::div(list_length + align_warp::Mask);  // ceildiv by power of 2

      uint32_t sample_offset = 0;
      if (probe_id > 0) { sample_offset = chunk_indices[probe_id - 1]; }
      assert(list_length == chunk_indices[probe_id] - sample_offset);
      assert(sample_offset + list_length <= max_samples);

      constexpr int kUnroll        = raft::WarpSize / Veclen;
      constexpr uint32_t kNumWarps = kThreadsPerBlock / raft::WarpSize;
      // Every warp reads WarpSize vectors and computes the distances to them.
      // Then, the distances and corresponding ids are distributed among the threads,
      // and each thread adds one (id, dist) pair to the filtering queue.
      for (uint32_t group_id = align_warp::div(threadIdx.x); group_id < num_groups;
           group_id += kNumWarps) {
        AccT dist         = 0;
        AccT norm_query   = 0;
        AccT norm_dataset = 0;
        // This is where this warp begins reading data (start position of an interleaved group)
        const T* data = list_data_ptrs[list_id] + (group_id * kIndexGroupSize) * dim;

        // This is the vector a given lane/thread handles
        const uint32_t vec_id = group_id * raft::WarpSize + lane_id;
        const bool valid      = vec_id < list_length && sample_filter(inds_ptrs,
                                                                 queries_offset + blockIdx.y,
                                                                 list_id,
                                                                 vec_id,
                                                                 bitset_ptr,
                                                                 bitset_len,
                                                                 original_nbits);

        if (valid) {
          // Process first shm_assisted_dim dimensions (always using shared memory)
          loadAndComputeDist<kUnroll, Veclen, T, AccT, ComputeNorm> lc(
            dist, norm_query, norm_dataset);
          for (int pos = 0; pos < shm_assisted_dim;
               pos += raft::WarpSize, data += kIndexGroupSize * raft::WarpSize) {
            lc.runLoadShmemCompute(data, query_shared, lane_id, pos);
          }

          if (dim > query_smem_elems) {
            // The default path - using shfl ops - for dimensions beyond query_smem_elems
            loadAndComputeDist<kUnroll, Veclen, T, AccT, ComputeNorm> lc(
              dist, norm_query, norm_dataset);
            for (int pos = shm_assisted_dim; pos < full_warps_along_dim; pos += raft::WarpSize) {
              lc.runLoadShflAndCompute(data, query, pos, lane_id);
            }
            lc.runLoadShflAndComputeRemainder(data, query, lane_id, dim, full_warps_along_dim);
          } else {
            // when  shm_assisted_dim == full_warps_along_dim < dim
            loadAndComputeDist<1, Veclen, T, AccT, ComputeNorm> lc(dist, norm_query, norm_dataset);
            for (int pos = full_warps_along_dim; pos < dim;
                 pos += Veclen, data += kIndexGroupSize * Veclen) {
              lc.runLoadShmemCompute(data, query_shared, lane_id, pos);
            }
          }
        }

        // Enqueue one element per thread
        float val = valid ? static_cast<float>(dist) : local_topk_t::queue_t::kDummy;

        if constexpr (ComputeNorm) {
          if (valid)
            val = val / (raft::sqrt(static_cast<float>(norm_query)) *
                         raft::sqrt(static_cast<float>(norm_dataset)));
        }
        if constexpr (kManageLocalTopK) {
          queue.add(val, sample_offset + vec_id);
        } else {
          if (vec_id < list_length) distances[sample_offset + vec_id] = val;
        }
      }

      // fill up unused slots for current query
      if constexpr (!kManageLocalTopK) {
        if (probe_id + 1 == n_probes) {
          for (uint32_t i = threadIdx.x + sample_offset + list_length; i < max_samples;
               i += blockDim.x) {
            distances[i] = local_topk_t::queue_t::kDummy;
          }
        }
      }
    }
  }

  // finalize and store selected neighbours
  if constexpr (kManageLocalTopK) {
    __syncthreads();
    queue.done(interleaved_scan_kernel_smem);
    queue.store(distances, neighbors, [](auto val) { return post_process(val); });
  }
}

}  // namespace cuvs::neighbors::ivf_flat::detail
