/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#include <cuvs/neighbors/ivf_rabitq/gpu_index/searcher_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/utils/memory.hpp>
#include <cuvs/neighbors/ivf_rabitq/utils/space_cuda.cuh>

#include <raft/core/resource/cuda_stream.hpp>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <iomanip>  // for std::setw
#include <numeric>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

struct StepStat {
  const char* name;
  double ms = 0;
};
//--------------------------------------------------------------------
// 1.  Helper: RAII wrapper for a pair of CUDA events
//--------------------------------------------------------------------
class GpuTimer {
 public:
  void start(cudaStream_t s)
  {
    RAFT_CUDA_TRY(cudaEventCreate(&beg_));
    RAFT_CUDA_TRY(cudaEventCreate(&end_));
    RAFT_CUDA_TRY(cudaEventRecord(beg_, s));
  }
  float stop(cudaStream_t s)
  {
    RAFT_CUDA_TRY(cudaEventRecord(end_, s));
    RAFT_CUDA_TRY(cudaEventSynchronize(end_));
    float ms = 0.0f;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, beg_, end_));
    RAFT_CUDA_TRY(cudaEventDestroy(beg_));
    RAFT_CUDA_TRY(cudaEventDestroy(end_));
    return ms;  // milliseconds on the GPU
  }

 private:
  cudaEvent_t beg_{}, end_{};
};

//--------------------------------------------------------------------
// 2.  Helper: CPU timer (steady clock)
//--------------------------------------------------------------------
class CpuTimer {
 public:
  void start() { t0_ = Clock::now(); }
  double stop() const
  {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
  }  // ms
 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point t0_;
};

/* new kernel function for topk*/
#ifndef TOPK_KM
#define TOPK_KM (100 * 10)  // compile-time constant  (K * M)
#endif
static_assert(TOPK_KM <= 1280, "KM too large for the register heap");

struct Candidate2 {
  float dist;
  int idx;
};

/// small register heap: unsorted set with linear replace
template <int KMAX>
struct TinyHeap {
  int size    = 0;
  float worst = -1.f;  // largest distance in the heap
  Candidate2 data[KMAX];

  __device__ void try_insert(float d, int id)
  {
    if (size < KMAX) {
      data[size++] = {d, id};
      if (d > worst) worst = d;
      return;
    }
    if (d >= worst) return;  // not better → ignore
    // replace the current worst (linear search)
    int pos = 0;
    for (int i = 1; i < size; ++i)
      if (data[i].dist > data[pos].dist) pos = i;
    data[pos] = {d, id};
    // recompute worst
    worst = data[0].dist;
    for (int i = 1; i < size; ++i)
      if (data[i].dist > worst) worst = data[i].dist;
  }
};

/// Kernel pass 1: each thread scans a strided chunk, keeps TinyHeap<TOPK_KM>
/// then each block cooperatively merges to a shared TinyHeap<TOPK_KM>
/// finally the block writes its KM candidates to a global scratch buffer.
__global__ void topk_collect_kernel(const float* __restrict__ d_dist,
                                    int N,
                                    int* __restrict__ scratch_count,       // 1 int
                                    Candidate2* __restrict__ scratch_buf)  // capacity: blocks*KM
{
  extern __shared__ Candidate2 sm[];  // KM * warpSize bytes

  TinyHeap<TOPK_KM> local;

  // strided scan
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
    float d = d_dist[i];
    local.try_insert(d, i);
  }

  // write local heap to shared memory
  for (int i = 0; i < local.size; ++i)
    sm[threadIdx.x * TOPK_KM + i] = local.data[i];
  __syncthreads();

  // first warp does a simple reduction over shared mem
  if (threadIdx.x == 0) {
    TinyHeap<TOPK_KM> blockh;
    int threads = blockDim.x;
    for (int t = 0; t < threads; ++t) {
      Candidate2* ptr = &sm[t * TOPK_KM];
      for (int j = 0; j < TOPK_KM; ++j)
        blockh.try_insert(ptr[j].dist, ptr[j].idx);
    }
    // write block KM into global scratch
    int offset = atomicAdd(scratch_count, blockh.size);
    for (int i = 0; i < blockh.size; ++i)
      scratch_buf[offset + i] = blockh.data[i];
  }
}
// ------------------------------------------------------------------
// add this near the top of the file (before the kernels)
// ------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void d_swap(T& a, T& b)
{
  T tmp = a;
  a     = b;
  b     = tmp;
}
/// Kernel pass 2: single block – reduce the scratch buffer to final KM
__global__ void topk_finalize_kernel(const Candidate2* __restrict__ scratch,
                                     int scratch_size,
                                     float* __restrict__ d_out_dist,
                                     int* __restrict__ d_out_idx)
{
  TinyHeap<TOPK_KM> heap;

  for (int i = threadIdx.x; i < scratch_size; i += blockDim.x)
    heap.try_insert(scratch[i].dist, scratch[i].idx);

  // merge to warp 0
  extern __shared__ Candidate2 sm[];
  for (int i = 0; i < heap.size; ++i)
    sm[threadIdx.x * TOPK_KM + i] = heap.data[i];
  __syncthreads();

  if (threadIdx.x == 0) {
    TinyHeap<TOPK_KM> finalh;
    int threads = blockDim.x;
    for (int t = 0; t < threads; ++t) {
      Candidate2* ptr = &sm[t * TOPK_KM];
      for (int j = 0; j < TOPK_KM; ++j)
        finalh.try_insert(ptr[j].dist, ptr[j].idx);
    }
    // simple insertion sort on KM (tiny)
    for (int i = 0; i < finalh.size; ++i)
      for (int j = i + 1; j < finalh.size; ++j)
        if (finalh.data[j].dist < finalh.data[i].dist) d_swap(finalh.data[i], finalh.data[j]);

    for (int i = 0; i < TOPK_KM; ++i) {
      d_out_dist[i] = finalh.data[i].dist;
      d_out_idx[i]  = finalh.data[i].idx;
    }
  }
}

/*********************************************************************
 * 1‑bit codes  ×  FP32 query  →  inner‑product + estimated distance
 *   – `quant_query_fp32` is now a float array of length num_dimensions
 *   – all other parameters are untouched
 *********************************************************************/
__global__ void compute_ip_kernel_fp32(const float* __restrict__ quant_query_fp32,  // <<< DIFFERENT
                                       const uint32_t* __restrict__ rabitq_codes_and_factors,
                                       size_t num_dimensions,
                                       size_t num_short_factors,
                                       size_t num_vector_cluster,
                                       float* __restrict__ ip_results,
                                       float* __restrict__ est_dis,
                                       float delta,
                                       float sumq,
                                       float qnorm,
                                       float one_over_sqrtD)
{
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane    = threadIdx.x & (WARP_SIZE - 1);

  if (warp_id >= (int)num_vector_cluster) return;

  // Each vector is stored as `num_words` 1‑bit blocks + `num_short_factors` floats
  const int num_words   = int((num_dimensions + 31) / 32);
  const uint32_t* block = rabitq_codes_and_factors + warp_id * (num_words + (int)num_short_factors);

  //----------------------------------------------------------------
  // 1) Per‑lane partial inner product
  //----------------------------------------------------------------
  float partial = 0.f;

  // Stride through dimensions so all 32 lanes cooperate
  for (int d = lane; d < (int)num_dimensions; d += WARP_SIZE) {
    const uint32_t word = block[d >> 5];  // d / 32
    const int bit       = (word >> (31 - (d & 31))) & 1;

    // If the bit is set, accumulate the FP32 query value
    if (bit) partial += __ldg(&quant_query_fp32[d]);  // read‑only cache
  }

  //    if (warp_id == 1) {
  //        printf("warp id: %d, partial before: %f\n", warp_id, partial);
  //    }

  //----------------------------------------------------------------
  // 2) Warp‑level reduction → lane 0 holds full dot product
  //----------------------------------------------------------------
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    partial += __shfl_down_sync(0xffffffff, partial, offset);

  //----------------------------------------------------------------
  // 3) Lane 0 writes results
  //----------------------------------------------------------------
  if (lane == 0) {
    const float res   = partial;  // ⟨q, b⟩
    const float onorm = (reinterpret_cast<const float*>(block))[num_words];

    const float ip_xb_prime = res;

    const float result =
      (ip_xb_prime - 0.5f * sumq + 0.58f) * (-5.0f * qnorm * one_over_sqrtD) * onorm +
      qnorm * qnorm + onorm * onorm;

    ip_results[warp_id] = ip_xb_prime;
    est_dis[warp_id]    = result;
#ifdef DEBUG_MULTIPLE_SEARCH
    if (warp_id > 270 && warp_id < 275)
      printf("warp id: %d, word: %x, ip_xb_prime: %f, result: %f, onorm: %f\n",
             warp_id,
             block[0],
             ip_xb_prime,
             result,
             onorm);
#endif
  }
}

/*********************************************************************
 * 1‑bit codes  ×  FP32 query  →  inner‑product + estimated distance
 *   – `quant_query_fp32` is now a float array of length num_dimensions
 *   – all other parameters are untouched
 *   - opt version: add filter to reduce possible output writes
 *********************************************************************/
__global__ void compute_ip_kernel_fp32_opt(
  const float* __restrict__ quant_query_fp32,  // <<< DIFFERENT
  const uint32_t* __restrict__ rabitq_codes_and_factors,
  size_t num_dimensions,
  size_t num_short_factors,
  size_t num_vector_cluster,
  float* __restrict__ ip_results,
  float* __restrict__ est_dis,
  float delta,
  float sumq,
  float qnorm,
  float one_over_sqrtD,
  float filter_dis)
{
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane    = threadIdx.x & (WARP_SIZE - 1);

  if (warp_id >= (int)num_vector_cluster) return;

  // Each vector is stored as `num_words` 1‑bit blocks + `num_short_factors` floats
  const int num_words   = int((num_dimensions + 31) / 32);
  const uint32_t* block = rabitq_codes_and_factors + warp_id * (num_words + (int)num_short_factors);

  //----------------------------------------------------------------
  // 1) Per‑lane partial inner product
  //----------------------------------------------------------------
  float partial = 0.f;

  // Stride through dimensions so all 32 lanes cooperate
  for (int d = lane; d < (int)num_dimensions; d += WARP_SIZE) {
    const uint32_t word = block[d >> 5];  // d / 32
    const int bit       = (word >> (31 - (d & 31))) & 1;

    // If the bit is set, accumulate the FP32 query value
    if (bit) partial += __ldg(&quant_query_fp32[d]);  // read‑only cache
  }

  //    if (warp_id == 1) {
  //        printf("warp id: %d, partial before: %f\n", warp_id, partial);
  //    }

  //----------------------------------------------------------------
  // 2) Warp‑level reduction → lane 0 holds full dot product
  //----------------------------------------------------------------
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    partial += __shfl_down_sync(0xffffffff, partial, offset);

  //----------------------------------------------------------------
  // 3) Lane 0 writes results
  //----------------------------------------------------------------
  if (lane == 0) {
    const float res   = partial;  // ⟨q, b⟩
    const float onorm = (reinterpret_cast<const float*>(block))[num_words];

    const float ip_xb_prime = res;

    const float result =
      (ip_xb_prime - 0.5f * sumq + 0.58f) * (-5.0f * qnorm * one_over_sqrtD) * onorm +
      qnorm * qnorm + onorm * onorm;

    if (result < filter_dis) {
      ip_results[warp_id] = ip_xb_prime;
      est_dis[warp_id]    = result;
    }
    //        if (warp_id == 1) {
    //            printf("warp id: %d, word: %x, ip_xb_prime: %f, result: %f, onorm: %f", warp_id,
    //            block[0], ip_xb_prime, result, onorm);
    //        }
  }
}

// new edit: return the TOPKth top least distance
float write_topk_to_pool(DeviceResultPool& d_pool,
                         float* d_top_dist,
                         uint32_t* d_top_pids,
                         int KM,
                         int TOPK,
                         rmm::cuda_stream_view stream)
{
  //    // 1) Wrap in thrust pointers
  //    auto par      = thrust::cuda::par.on(stream);
  //    auto dist_ptr = thrust::device_pointer_cast(d_top_dist);
  //    auto pid_ptr  = thrust::device_pointer_cast(d_top_pids);
  //
  //    // 2) In‐place sort the KM entries by ascending distance,
  //    //    carrying along the pids in d_top_pids.
  //    thrust::sort_by_key(par, dist_ptr, dist_ptr + KM, pid_ptr);

  // (1) ~ (2) cub sort
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_storage_bytes,
                                  d_top_dist,
                                  d_top_dist,
                                  d_top_pids,
                                  d_top_pids,
                                  KM,
                                  0,
                                  32,
                                  stream);
  RAFT_CUDA_TRY(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_storage_bytes,
                                  d_top_dist,
                                  d_top_dist,
                                  d_top_pids,
                                  d_top_pids,
                                  KM,
                                  0,
                                  32,
                                  stream);

  // 3) Copy the first TOPK distances & pids into the result pool’s arrays.
  raft::copy(d_pool.distances.data_handle(), d_top_dist, TOPK, stream);
  raft::copy(d_pool.ids.data_handle(), d_top_pids, TOPK, stream);

  float top_k_distance;
  raft::copy(&top_k_distance, d_pool.distances.data_handle() + (TOPK - 1), 1, stream);

  RAFT_CUDA_TRY(cudaFreeAsync(d_temp_storage, stream));

  return top_k_distance;

  //    // 4) Update the pool’s size field on the device
  //    set_pool_size_kernel<<<1,1>>>(d_pool, TOPK);
  //    RAFT_CUDA_TRY(cudaGetLastError());
  //    RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

float write_topk_to_pool_one_for_multi(DeviceResultPool& d_pool,
                                       float* d_top_dist,
                                       uint32_t* d_top_pids,
                                       int KM,
                                       int TOPK,
                                       float* d_filter_distk,
                                       rmm::cuda_stream_view stream)
{
  //    // 1) Wrap in thrust pointers
  //    auto par      = thrust::cuda::par.on(stream);
  //    auto dist_ptr = thrust::device_pointer_cast(d_top_dist);
  //    auto pid_ptr  = thrust::device_pointer_cast(d_top_pids);
  //
  //    // 2) In‐place sort the KM entries by ascending distance,
  //    //    carrying along the pids in d_top_pids.
  //    thrust::sort_by_key(par, dist_ptr, dist_ptr + KM, pid_ptr);

  // (1) ~ (2) cub sort
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_storage_bytes,
                                  d_top_dist,
                                  d_top_dist,
                                  d_top_pids,
                                  d_top_pids,
                                  KM,
                                  0,
                                  32,
                                  stream);
  RAFT_CUDA_TRY(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_storage_bytes,
                                  d_top_dist,
                                  d_top_dist,
                                  d_top_pids,
                                  d_top_pids,
                                  KM,
                                  0,
                                  32,
                                  stream);

  // 3) Copy the first TOPK distances & pids into the result pool’s arrays.
  raft::copy(d_pool.distances.data_handle(), d_top_dist, TOPK, stream);
  raft::copy(d_pool.ids.data_handle(), d_top_pids, TOPK, stream);
  raft::copy(d_filter_distk, d_pool.distances.data_handle() + (TOPK - 1), 1, stream);

  RAFT_CUDA_TRY(cudaFreeAsync(d_temp_storage, stream));

  return 0;

  //    // 4) Update the pool’s size field on the device
  //    set_pool_size_kernel<<<1,1>>>(d_pool, TOPK);
  //    RAFT_CUDA_TRY(cudaGetLastError());
  //    RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

/**
 * @brief Refine the top‑KM approximate distances using precomputed ip2 and onorm from short_data.
 *
 * @param d_top_dist          [in,out] device array of length KM: originally contains (res*delta)
 * @param d_ip2               [in]     device array of length KM: long‑code inner products
 * @param d_top_idx           [in]     device array of length KM: original vector IDs
 * @param sqr_y               [in]     squared centroid distance
 * @param FAC_RESCALE         [in]     as in your formula
 * @param sumq                [in]     sumq from query preparation
 * @param KM                  [in]     number of candidates (K×M)
 * @param d_short_data        [in]     device‐side buffer of short_data blocks
 * @param short_block_bytes   [in]     size in bytes of one short_data block (code+factors)
 * @param d_ex_factor         [in]     device array of ExFactor of length N
 * @param num_short_factors.  [in].    number of short factors.
 */
__global__ void refine_with_ip2_kernel(float* d_top_dist,
                                       const float* d_ip2,
                                       const int* d_top_idx,
                                       float sqr_y,
                                       int FAC_RESCALE,
                                       float sumq,
                                       size_t KM,
                                       const uint32_t* d_short_data,
                                       size_t short_block_bytes_in_uint32,
                                       const ExFactor* d_ex_factor,
                                       size_t num_short_factors)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (int)KM) return;

  // 1) which original vector?
  int vid = d_top_idx[tid];
  if (vid == -1) return;  // a fix for sentinels, leave the distances max

  // 2) load onorm from the short_data block:
  //    each block is short_block_bytes long, and the first factor (onorm)
  //    is stored immediately after the code section.  We assume that
  //    the code portion is exactly (short_block_bytes - sizeof(float)*num_short_factors).
  //    Since onorm is the very first float factor, it lives at byte-offset
  //      code_bytes = short_block_bytes - sizeof(float)*NUM_FACTORS
  //    We just index into the block by short_block_bytes * vid + code_bytes.
  const uint32_t* block = d_short_data + vid * short_block_bytes_in_uint32;
  // The first factor is a float at the very end of the code region:
  float onorm =
    *reinterpret_cast<const float*>(block + (short_block_bytes_in_uint32 - num_short_factors));

  // 3) load xipnorm:
  float xip = d_ex_factor[vid].xipnorm;

  // 4) read precomputed ip2:
  float ip2 = d_ip2[tid];

  // 5) original approximate distance (res*delta):
  float approx = d_top_dist[tid];

  // 6) apply refinement formula:
  float refined = onorm * onorm + sqr_y -
                  xip * (sqrt(sqr_y)) * (FAC_RESCALE * approx + ip2 - (FAC_RESCALE - 0.5f) * sumq);
#ifdef DEBUG_MULTIPLE_SEARCH
  printf(
    "onorm_ori_refine: %f, sqr_y_ori: %f, xip_ori: %f, approx_ori: %f, sumq_ori: %f, ip2_ori: %f, "
    "ip_ori: %f, idx in cluster and refined dis: %d %f\n",
    onorm,
    sqr_y,
    xip,
    approx,
    sumq,
    ip2,
    d_top_dist[tid],
    vid,
    refined);
#endif

  // 7) store back:
  d_top_dist[tid] = refined;
  //    printf("refined dis: %f\n", refined);
}

// Read an EX_BITS‐bit value for dimension d from a packed byte array.
//   codes: packed EX_BITS per dim, total bits = D*EX_BITS, stored big‐endian within each byte.
//   d:     which dimension [0..D)
//   EX_BITS:  bits per dimension (<=8).
__device__ inline uint32_t extract_code(const uint8_t* codes, size_t d, size_t EX_BITS)
{
  size_t bitPos    = d * EX_BITS;
  size_t byteIdx   = bitPos >> 3;  // bitPos/8
  size_t bitOffset = bitPos & 7;   // bitPos%8
  // grab enough bits across this byte (and maybe the next)
  uint32_t v = codes[byteIdx] << 8;
  if (bitOffset + EX_BITS > 8) { v |= codes[byteIdx + 1]; }
  // now top 16 bits of v hold at least (bitOffset+EX_BITS) bits—we shift:
  int shift = 16 - (bitOffset + EX_BITS);
  return (v >> shift) & ((1u << EX_BITS) - 1);
}

/**
 * @brief Select the K×M smallest distances from d_dist (length N), sorting d_dist
 *        in-place, and retrieve their corresponding pids (from d_ids) and original
 *        positions.  d_ids itself is never mutated.
 *
 * @param d_dist       Device pointer to distances array [N].  This **will** be reordered.
 * @param d_ids        Device pointer to pid array       [N].  This is untouched.
 * @param N            Total length of both above arrays.
 * @param K            Top‑K parameter.
 * @param M            Multiplier (so we select K*M elements).
 * @param d_out_dist   Device pointer to output distances [K*M].
 * @param d_out_pids   Device pointer to output pids      [K*M].
 * @param d_out_idx    Device pointer to output indices   [K*M].
 */
inline void select_topk_m_inplace(float* d_dist,
                                  const uint32_t* d_ids,
                                  float* d_ip_results,
                                  int N,
                                  int K,
                                  int M,
                                  float* d_out_dist,
                                  uint32_t* d_out_pids,
                                  int* d_out_idx,
                                  rmm::cuda_stream_view stream)
{
  int KM = K * M;
  assert(N > 0 && KM > 0);

  // 1) Build an index array [0,1,2,…,N-1] on the device.
  thrust::device_vector<int> idx(N);
  thrust::sequence(thrust::cuda::par.on(stream), idx.begin(), idx.end());

  // 2) Wrap raw pointer for the distances
  thrust::device_ptr<float> dist_ptr(d_dist);

  // 3) In‑place sort the first N distances ascending, permuting idx[] in the same way.
  thrust::sort_by_key(thrust::cuda::par.on(stream), dist_ptr, dist_ptr + N, idx.begin());

  // How many real entries we can copy
  int valid = (KM < N ? KM : N);

  // 4) Copy the top‑valid distances into your output buffer
  // new edit: copy ip results instead valid-distances to restore long codes;
  thrust::gather(thrust::cuda::par.on(stream),
                 idx.begin(),
                 idx.begin() + valid,
                 thrust::device_pointer_cast(d_ip_results),
                 thrust::device_pointer_cast(d_out_dist));

  // 5) Copy the top‑valid indices into d_out_idx
  thrust::copy(thrust::cuda::par.on(stream),
               idx.begin(),
               idx.begin() + valid,
               thrust::device_pointer_cast(d_out_idx));

  // 6) Gather the pids from the ORIGINAL d_ids array
  thrust::gather(thrust::cuda::par.on(stream),
                 idx.begin(),
                 idx.begin() + valid,
                 thrust::device_pointer_cast(const_cast<uint32_t*>(d_ids)),
                 thrust::device_pointer_cast(d_out_pids));

  // 7) Fill the remaining slots [valid..KM) with sentinels
  if (valid < KM) {
    auto dist_tail = thrust::device_pointer_cast(d_out_dist + valid);
    thrust::fill_n(
      thrust::cuda::par.on(stream), dist_tail, KM - valid, std::numeric_limits<float>::infinity());

    auto pid_tail = thrust::device_pointer_cast(d_out_pids + valid);
    thrust::fill_n(thrust::cuda::par.on(stream), pid_tail, KM - valid, UINT32_MAX);

    auto idx_tail = thrust::device_pointer_cast(d_out_idx + valid);
    thrust::fill_n(thrust::cuda::par.on(stream), idx_tail, KM - valid, -1);
  }
}

inline void select_topk_m_inplace_with_time(float* d_dist,
                                            const uint32_t* d_ids,
                                            float* d_ip_results,
                                            int N,
                                            int K,
                                            int M,
                                            float* d_out_dist,
                                            uint32_t* d_out_pids,
                                            int* d_out_idx,
                                            cudaStream_t stream)
{
  //------------------------------------------------------------------
  // Timing setup
  //------------------------------------------------------------------
  std::vector<StepStat> stats;
  stats.reserve(8);

  GpuTimer timer;

  int KM = K * M;
  assert(N > 0 && KM > 0);

  //------------------------------------------------------------------
  // 1) Build an index array [0,1,2,…,N-1] on the device.
  //------------------------------------------------------------------
  timer.start(stream);
  thrust::device_vector<int> idx(N);
  thrust::sequence(thrust::cuda::par.on(stream), idx.begin(), idx.end());
  stats.push_back({"sequence_idx", timer.stop(stream)});

  //------------------------------------------------------------------
  // 2) Wrap raw pointer for the distances
  //------------------------------------------------------------------
  thrust::device_ptr<float> dist_ptr(d_dist);

  //------------------------------------------------------------------
  // 3) In‑place sort the first N distances ascending, permuting idx[] in the same way.
  //    //------------------------------------------------------------------
  // original version
  thrust::sort_by_key(thrust::cuda::par.on(stream), dist_ptr, dist_ptr + N, idx.begin());

  // cublas version
  timer.start(stream);

  // We need an auxiliary buffer for keys (distances) and values (idx)
  float* d_dist_alt = nullptr;  // alternate keys
  int* d_idx_alt    = nullptr;  // alternate values
  RAFT_CUDA_TRY(cudaMallocAsync(&d_dist_alt, sizeof(float) * N, stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&d_idx_alt, sizeof(int) * N, stream));

  // Set up DoubleBuffers so CUB can ping-pong
  cub::DoubleBuffer<float> keys(d_dist, d_dist_alt);
  cub::DoubleBuffer<int> vals(idx.data().get(), d_idx_alt);

  // Temporary storage for the sort
  void* d_temp_storage = nullptr;
  size_t temp_bytes    = 0;

  // 1st pass: just query temp size
  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_bytes,
                                  keys,
                                  vals,  // key/value buffers
                                  N,     // number of items
                                  0,
                                  sizeof(float) * 8,  // sort all 32 bits (low→high)
                                  stream);

  RAFT_CUDA_TRY(cudaMallocAsync(&d_temp_storage, temp_bytes, stream));

  // 2nd pass: actual sort
  cub::DeviceRadixSort::SortPairs(
    d_temp_storage, temp_bytes, keys, vals, N, 0, sizeof(float) * 8, stream);

  // After the call, the sorted data are in keys.Current(), vals.Current()
  // but the code that follows expects them back in d_dist and idx[0]
  if (keys.Current() != d_dist) {
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_dist, keys.Current(), sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
  }
  if (vals.Current() != idx.data().get()) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      idx.data().get(), vals.Current(), sizeof(int) * N, cudaMemcpyDeviceToDevice, stream));
  }

  // clean up temp allocations (asynchronous, tied to stream)
  RAFT_CUDA_TRY(cudaFreeAsync(d_dist_alt, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_idx_alt, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_temp_storage, stream));

  stats.push_back({"cub_radix_sort", timer.stop(stream)});

  // self-kernel function version

  //    timer.start(stream);
  //
  //    int blocks = 256;                 // tune
  //    int threads= 256;
  //
  //    // scratch for pass-1 (allocate once per call)
  //    int* d_counter;            cudaMallocAsync(&d_counter, sizeof(int), stream);
  //    Candidate2* d_scratch;      cudaMallocAsync(&d_scratch,
  //                                               sizeof(Candidate) * blocks * TOPK_KM, stream);
  //    cudaMemsetAsync(d_counter, 0, sizeof(int), stream);
  //
  //    size_t shmem1 = threads * TOPK_KM * sizeof(Candidate);
  //    topk_collect_kernel<<<blocks, threads, shmem1, stream>>>(
  //            d_dist, N, d_counter, d_scratch);
  //
  //    // read scratch size
  //    int h_count;
  //    cudaMemcpyAsync(&h_count, d_counter, sizeof(int),
  //                    cudaMemcpyDeviceToHost, stream);
  //    cudaStreamSynchronize(stream);
  //
  //    size_t shmem2 = threads * TOPK_KM * sizeof(Candidate);
  //    topk_finalize_kernel<<<1, threads, shmem2, stream>>>(
  //            d_scratch, h_count, d_out_dist, d_out_idx);
  //
  //    cudaFreeAsync(d_counter , stream);
  //    cudaFreeAsync(d_scratch , stream);
  //
  //    stats.push_back({"topk_O(N)", timer.stop(stream)});

  //------------------------------------------------------------------
  // How many real entries we can copy
  //------------------------------------------------------------------
  int valid = (KM < N ? KM : N);

  //------------------------------------------------------------------
  // 4) Gather the top‑valid ip results
  //------------------------------------------------------------------
  timer.start(stream);
  thrust::gather(thrust::cuda::par.on(stream),
                 idx.begin(),
                 idx.begin() + valid,
                 thrust::device_pointer_cast(d_ip_results),
                 thrust::device_pointer_cast(d_out_dist));
  stats.push_back({"gather_ip", timer.stop(stream)});

  //------------------------------------------------------------------
  // 5) Copy the top‑valid indices into d_out_idx
  //------------------------------------------------------------------
  timer.start(stream);
  thrust::copy(thrust::cuda::par.on(stream),
               idx.begin(),
               idx.begin() + valid,
               thrust::device_pointer_cast(d_out_idx));
  stats.push_back({"copy_idx", timer.stop(stream)});

  //------------------------------------------------------------------
  // 6) Gather the pids from the ORIGINAL d_ids array
  //------------------------------------------------------------------
  timer.start(stream);
  thrust::gather(thrust::cuda::par.on(stream),
                 idx.begin(),
                 idx.begin() + valid,
                 thrust::device_pointer_cast(const_cast<uint32_t*>(d_ids)),
                 thrust::device_pointer_cast(d_out_pids));
  stats.push_back({"gather_pid", timer.stop(stream)});

  //------------------------------------------------------------------
  // 7) Fill the remaining slots [valid..KM) with sentinels (optional)
  //------------------------------------------------------------------
  if (valid < KM) {
    timer.start(stream);
    auto dist_tail = thrust::device_pointer_cast(d_out_dist + valid);
    thrust::fill_n(
      thrust::cuda::par.on(stream), dist_tail, KM - valid, std::numeric_limits<float>::infinity());
    stats.push_back({"filln_dist", timer.stop(stream)});

    timer.start(stream);
    auto pid_tail = thrust::device_pointer_cast(d_out_pids + valid);
    thrust::fill_n(thrust::cuda::par.on(stream), pid_tail, KM - valid, UINT32_MAX);
    stats.push_back({"filln_pid", timer.stop(stream)});

    timer.start(stream);
    auto idx_tail = thrust::device_pointer_cast(d_out_idx + valid);
    thrust::fill_n(thrust::cuda::par.on(stream), idx_tail, KM - valid, -1);
    stats.push_back({"filln_idx", timer.stop(stream)});
  }

  //------------------------------------------------------------------
  // Final: print timing statistics
  //------------------------------------------------------------------
  double total = 0.0;
  for (const auto& s : stats)
    total += s.ms;

  printf("\n----- Timing Breakdown: select_topk_m_inplace -----\n");
  for (const auto& s : stats) {
    printf("%-15s : %8.3f ms (%.1f%%)\n", s.name, s.ms, s.ms * 100.0 / total);
  }
  printf("TOTAL             : %8.3f ms\n", total);
}

//------------------------------------------------------------------------------------------------------------------
//  FusedFilterGather
//    • each thread looks at est_dist[idx]
//    • if it passes the threshold, it writes  (idx, ip[idx], pid[idx])
//      contiguously into   out_idx / out_ip / out_pid
//    • returns the total #written in *d_written
//------------------------------------------------------------------------------------------------------------------
template <int BLOCK_SZ = 256>
__global__ void FusedFilterGather(const float* __restrict__ est_dist,
                                  const float* __restrict__ ip,
                                  const uint32_t* __restrict__ pid,
                                  int N,
                                  float filter_k,
                                  /*out*/ int* __restrict__ d_written,  // single int in global mem
                                  /*out*/ int* __restrict__ out_idx,
                                  /*out*/ float* __restrict__ out_ip,
                                  /*out*/ uint32_t* __restrict__ out_pid)
{
  __shared__ int block_base;  // where this block will start writing

  const int tid       = threadIdx.x + blockIdx.x * blockDim.x;
  const int lane      = threadIdx.x & 31;
  const int warp      = threadIdx.x >> 5;
  const unsigned mask = __ballot_sync(0xFFFFFFFF, tid < N && est_dist[tid] < filter_k);

  //------------------------- warp-prefix --------------------------------
  const int warp_sel = __popc(mask);  // #selected in this warp
  int warp_offset    = 0;             // prefix inside block
  if (lane == 0) warp_offset = atomicAdd(&block_base, warp_sel);
  warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);

  const int lane_prefix = __popc(mask & ((1u << lane) - 1));
  const int out_pos = (tid < N && (mask >> lane) & 1) ? block_base + warp_offset + lane_prefix : -1;

  //------------------------- write survivors ----------------------------
  if (out_pos >= 0) {
    out_idx[out_pos] = tid;
    out_ip[out_pos]  = ip[tid];
    out_pid[out_pos] = pid[tid];
  }

  //------------------------- after finishing the block ------------------
  if (threadIdx.x == 0) {
    const int block_written = block_base;  // now holds total for this block
    if (block_written) {
      const int global_off = atomicAdd(d_written, block_written);
      // shift this block’s writes to the global position
      int* idx_dst      = out_idx + global_off;
      float* ip_dst     = out_ip + global_off;
      uint32_t* pid_dst = out_pid + global_off;

      for (int i = 0; i < block_written; ++i) {
        idx_dst[i] = out_idx[i];  // copy from shared area
        ip_dst[i]  = out_ip[i];
        pid_dst[i] = out_pid[i];
      }
    }
    block_base = 0;  // reset for next launch (optional)
  }
}

// Disabled, sth wrong with it.
inline int select_and_keep_topKM_by_distance_timed(const float* est_dist,
                                                   const uint32_t* ids,
                                                   const float* ip_results,
                                                   std::size_t N,
                                                   int KM,
                                                   float filter_dist_k,
                                                   float* d_top_ip,       // OUT
                                                   uint32_t* d_top_pids,  // OUT
                                                   int* d_top_idx,        // OUT
                                                   rmm::cuda_stream_view stream)
{
  CpuTimer host_timer;
  std::vector<StepStat> s(5);  // 0..3 GPU, 4 host

  host_timer.start();  // start host timing (tiny)

  //    if (N == 0 || KM == 0) {      // trivial case – nothing to do
  //        s[4] = {"host-overhead", host_timer.stop()};
  //        goto print_stats;
  //    }

  //------------------------------------------------------------------
  // (1)  filter — copy_if
  //------------------------------------------------------------------
  GpuTimer gt;
  gt.start(stream);
  thrust::device_vector<int> selected_idx(N);
  const int BLOCK = 256;
  const int GRID  = (N + BLOCK - 1) / BLOCK;

  int* d_written;
  RAFT_CUDA_TRY(cudaMalloc(&d_written, sizeof(int)));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_written, 0, sizeof(int), stream));

  FusedFilterGather<BLOCK><<<GRID, BLOCK, 0, stream>>>(
    est_dist, ip_results, ids, N, filter_dist_k, d_written, d_top_idx, d_top_ip, d_top_pids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  int h_written;
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(&h_written, d_written, sizeof(int), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaFree(d_written));

  s[0] = {"gather 3× arrays", gt.stop(stream)};

  //------------------------------------------------------------------
  // (4) final device→device copies to the user buffers
  //------------------------------------------------------------------
  gt.start(stream);
  if (h_written <= KM) return h_written;  // done – no extra work

  //---------------- when we overflow, just sort the compacted piece -----
  thrust::sort_by_key(
    thrust::cuda::par.on(stream),
    d_top_ip,
    d_top_ip + h_written,  // key = dist
    thrust::make_zip_iterator(thrust::make_tuple(d_top_idx,
                                                 d_top_ip,  // <-- we kept inner-prod here
                                                 d_top_pids)));
  s[1] = {"device→device copies out", gt.stop(stream)};

  s[2] = {"host-overhead", host_timer.stop()};

print_stats:
  //------------------------------------------------------------------
  // pretty print
  //------------------------------------------------------------------
  const double gpu_total = std::accumulate(
    s.begin(), s.begin() + 4, 0.0, [](double acc, const StepStat& st) { return acc + st.ms; });

  printf("\n╔════════════════════════════╦═══════════╦══════════╗\n");
  printf("║  step                      ║   ms      ║ percent  ║\n");
  printf("╠════════════════════════════╬═══════════╬══════════╣\n");
  for (int i = 0; i < 2; ++i)
    printf("║ %-26s ║ %8.2f ║ %7.1f %% ║\n",
           s[i].name,
           s[i].ms,
           gpu_total ? (s[i].ms / gpu_total * 100.0) : 0.0);
  printf("╠════════════════════════════╬═══════════╬══════════╣\n");
  printf("║ TOTAL GPU time             ║ %8.2f ║ 100 %%    ║\n", gpu_total);
  printf("╚════════════════════════════╩═══════════╩══════════╝\n");
  printf("Host (CPU) overhead: %.2f ms\n\n", s[2].ms);

  return h_written < KM ? h_written : KM;
}

/// Each warp handles one of the KM candidates.  We launch at least KM*32 threads.
/// Inputs:
///   unit_q            – float[D] query vector (device)
///   d_top_idx         – int[KM] indices into your long‐code table
///   KM                – number of candidates
///   D                 – dimension
///   EX_BITS           – bits per dimension in the long code
///   d_long_code       – uint8_t[N * long_code_bytes] packed codes
///   long_code_bytes   – bytes per vector (=(D*EX_BITS+7)/8)
/// Output:
///   d_ip2             – float[KM] inner products
__global__ void compute_long_ip_kernel(const float* unit_q,
                                       const int* d_top_idx,
                                       size_t KM,
                                       size_t D,
                                       size_t EX_BITS,
                                       const uint8_t* d_long_code,
                                       size_t long_code_bytes,
                                       float* d_ip2)
{
  const int WARP = 32;
  int tid        = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId     = tid / WARP;
  int lane       = threadIdx.x & (WARP - 1);
  if (warpId >= (int)KM) return;

  // Which vector:
  int vid = d_top_idx[warpId];
  // fix for sentinel
  if (vid == -1) return;

  const uint8_t* codes = d_long_code + vid * long_code_bytes;

  // Partial sum for this lane:
  float partial = 0.f;
  for (int d = lane; d < (int)D; d += WARP) {
    uint32_t c = extract_code(codes, d, EX_BITS);
    partial += float(c) * unit_q[d];
  }

  // Warp‐wide sum:
#pragma unroll
  for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffff, partial, offset);
  }

  // Write result:
  if (lane == 0) {
    d_ip2[warpId] = partial;
#ifdef DEBUG_MULTIPLE_SEARCH
//        if (vid == 925) {
//            printf("first four codes of uint8_t: %d, %d, %d, %d\n", codes[0], codes[1], codes[2],
//            codes[3]);
//        }
#endif
  }
}

// In this implementation, we directly compute the inner product between 1-bit rabitq codes and
// quantized queries
void SearcherGPU::SearchCluster(const IVFGPU& cur_ivf,
                                const IVFGPU::GPUClusterMeta& cur_cluster,
                                float sqr_y,
                                DeviceResultPool& KNNs,
                                float* centroid_data)
{
  // Get related device pointers in kernel function using cid
  uint32_t* rabitq_codes_and_factor = cur_cluster.first_block(cur_ivf);
  PID* ids                          = cur_cluster.ids(cur_ivf);

  // debug
  //    uint32_t code;
  //    RAFT_CUDA_TRY(cudaMemcpy(&code, rabitq_codes_and_factor, sizeof(uint32_t),
  //    cudaMemcpyDeviceToHost)); printf("Cur cluster start idx: %d, first rabitq code in uint32: %x
  //    \n", cur_cluster.start_index, code);

  // Related information
  size_t num_vector_cluster = cur_cluster.num;
  size_t num_dimensions     = cur_ivf.get_num_padded_dim();

  float y = std::sqrt(sqr_y);

  // Option 1: using quantized query
  // preparation for query (on CPU (may not be needed))
  this->shift = 0;
  this->sumq  = normalize_query16_scalar(unit_q, query, centroid_data, y, D);
  //    high_acc_quantize16_scalar(quant_query, unit_q, delta, D);

  // copy unit_q to gpu:
  float* unit_q_gpu = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&unit_q_gpu, sizeof(float) * num_dimensions, stream_));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    unit_q_gpu, unit_q, sizeof(float) * num_dimensions, cudaMemcpyHostToDevice, stream_));

  //    float distk = KNNs.worst_distance();
  // tmp = ((res[i] + shift) * delta - (0.5*sumq - 0.58)) * (-5*qnorm/sqrtD) * onorm + qnorm² +
  // onorm²

  // first step: IP kernel function

  // Option 1: using quantized query

  // copy quantized query vector to GPU
  //    int16_t* quant_query_gpu = nullptr;
  //    cudaMallocAsync(&quant_query_gpu, sizeof(int16_t) * D, stream);
  //    cudaMemcpyAsync(quant_query_gpu, quant_query,
  //                    sizeof(int16_t) * D,
  //                    cudaMemcpyHostToDevice, stream);

  // Option 2: using float
  //    float* rotated_query_gpu;
  //    RAFT_CUDA_TRY(cudaMallocAsync((void**)& rotated_query_gpu, sizeof(float) * D, stream));
  //    RAFT_CUDA_TRY(cudaMemcpyAsync( rotated_query_gpu, unit_q, sizeof(float) * D,
  //    cudaMemcpyHostToDevice, stream));
  // get cluster data pointer

  // allocate results space for results;
  float* ip_results = nullptr;
  float* est_dis    = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&ip_results, sizeof(float) * num_vector_cluster, stream_));
  RAFT_CUDA_TRY(cudaMallocAsync(&est_dis, sizeof(float) * num_vector_cluster, stream_));

  // execute kernel function && restore estimated distances
  int threadsPerBlock = 256;  // must be a multiple of 32
  int warpsPerBlock   = threadsPerBlock / 32;
  int grid            = (num_vector_cluster + warpsPerBlock - 1) / warpsPerBlock;
  //

  //    compute_ip_kernel<<<grid, threadsPerBlock, 0, stream>>>(
  //            quant_query_gpu,
  //            rabitq_codes_and_factor,
  //            num_dimensions,
  //            cur_ivf.quantizer().num_short_factors(),
  //            num_vector_cluster,
  //            ip_results,
  //            est_dis,
  //            delta,
  //            sumq,
  //            y,
  //            one_over_sqrtD
  //    );
  compute_ip_kernel_fp32<<<grid, threadsPerBlock, 0, stream_>>>(
    unit_q_gpu,
    rabitq_codes_and_factor,
    num_dimensions,
    cur_ivf.quantizer().num_short_factors(),
    num_vector_cluster,
    ip_results,
    est_dis,
    delta,
    sumq,
    y,
    one_over_sqrtD);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  //    RAFT_CUDA_TRY(cudaDeviceSynchronize());

  // Select certain distances and restore them

  // First we get TOPK * M results, then get their ex_codes;
  size_t M  = 10;  // multiples of 10 temporally
  size_t KM = M * KNNs.capacity;

  // 2) Allocate output buffers on device:
  float* d_top_dist;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_top_dist, sizeof(float) * KM, stream_));
  uint32_t* d_top_pids;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_top_pids, sizeof(uint32_t) * KM, stream_));
  int* d_top_idx;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_top_idx, sizeof(int) * KM, stream_));

  select_topk_m_inplace(  // unchanged API, but it calls kernels ↦ must take stream too
    est_dis,
    ids,
    ip_results,
    num_vector_cluster,
    KNNs.capacity,
    M,
    d_top_dist,
    d_top_pids,
    d_top_idx,
    stream_  // <-- add parameter inside that function
  );

  // restore distances using the excodes

  // first compute the ip between long codes and unit_q
  // malloc temp space
  float* d_ip2;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_ip2, sizeof(float) * KM, stream_));

  // debug - access llong code
  //    uint8_t temp;
  //    RAFT_CUDA_TRY(cudaMemcpy(&temp, cur_cluster.long_code(cur_ivf, 0,
  //    cur_ivf.quantizer().long_code_length()), sizeof(uint8_t), cudaMemcpyDeviceToHost));
  //    printf("Cur cluster start idx: %d, first exrabitq code block in uint8: %x \n",
  //    cur_cluster.start_index, temp);
  // access the whole block
  //    for (int i = 0; i < cur_cluster.num; i++) {
  //        RAFT_CUDA_TRY(cudaMemcpy(&temp, cur_cluster.long_code(cur_ivf, i,
  //        cur_ivf.quantizer().long_code_length()), sizeof(uint8_t), cudaMemcpyDeviceToHost));
  //    }

  grid = (KM + warpsPerBlock - 1) / warpsPerBlock;
  compute_long_ip_kernel<<<grid, threadsPerBlock, 0, stream_>>>(
    unit_q_gpu,
    d_top_idx,
    KM,
    D,
    cur_ivf.get_ex_bits(),
    cur_cluster.long_code(cur_ivf, 0, cur_ivf.quantizer().long_code_length()),
    cur_ivf.quantizer().long_code_length(),
    d_ip2);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  //    RAFT_CUDA_TRY(cudaDeviceSynchronize());

  // restore distances;
  // Launch one thread per candidate (you can choose any block size you like).
  int threads = 256;
  int blocks  = (KM + threads - 1) / threads;
  // now d_top_dist is top ip instead
  refine_with_ip2_kernel<<<blocks, threads, 0, stream_>>>(
    d_top_dist,
    d_ip2,
    d_top_idx,
    sqr_y,
    FAC_RESCALE,
    sumq,
    KM,
    cur_cluster.first_block(cur_ivf),
    cur_ivf.quantizer().short_code_length() + cur_ivf.quantizer().num_short_factors(),
    cur_cluster.ex_factor(cur_ivf, 0),
    cur_ivf.quantizer().num_short_factors());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  //    RAFT_CUDA_TRY(cudaDeviceSynchronize());

  // sort finally and write back results
  auto cluster_topk_distance =
    write_topk_to_pool(KNNs, d_top_dist, d_top_pids, KM, KNNs.capacity, stream_);
  KNNs.size = KNNs.capacity;

  // set top-k for further computing (only for the first cluster)
  if (cluster_topk_distance < h_filter_distk) { h_filter_distk = cluster_topk_distance; }

  //    cudaStreamSynchronize(stream);

  // release space
  //    free (centroid_data);
  //    RAFT_CUDA_TRY(cudaFree(quant_query_gpu));
  //    RAFT_CUDA_TRY(cudaFree(rotated_query_gpu));
  //    cudaFreeAsync(rotated_query_gpu, stream);
  //    cudaFreeAsync(quant_query_gpu, stream);
  RAFT_CUDA_TRY(cudaFreeAsync(ip_results, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(est_dis, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_top_dist, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_top_pids, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_top_idx, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(unit_q_gpu, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_ip2, stream_));
}

void SearcherGPU::SearchClustershowingTime(const IVFGPU& cur_ivf,
                                           const IVFGPU::GPUClusterMeta& cur_cluster,
                                           float sqr_y,
                                           DeviceResultPool& KNNs,
                                           float* centroid_data)
{
  // to store results
  struct StepStat {
    std::string name;
    double ms = 0;
  };
  std::vector<StepStat> stats;  // push in call order
  size_t num_vector_cluster         = cur_cluster.num;
  size_t num_dimensions             = cur_ivf.get_num_padded_dim();
  uint32_t* rabitq_codes_and_factor = cur_cluster.first_block(cur_ivf);
  PID* ids                          = cur_cluster.ids(cur_ivf);

  //--------------------------------------------------------
  // A) host-side scalar preparation
  //--------------------------------------------------------
  CpuTimer cpu;
  cpu.start();
  float y     = std::sqrt(sqr_y);
  this->shift = 0;
  this->sumq  = normalize_query16_scalar(unit_q, query, centroid_data, y, D);
  //    high_acc_quantize16_scalar(quant_query, unit_q, delta, D);

  stats.push_back({"host-prep", cpu.stop()});  // ms

  //--------------------------------------------------------
  // B) H2D copies of query + buffers (GPU time)
  //--------------------------------------------------------
  GpuTimer gt;
  gt.start(stream_);
  float* unit_q_gpu = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&unit_q_gpu, sizeof(float) * num_dimensions, stream_));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    unit_q_gpu, unit_q, sizeof(float) * num_dimensions, cudaMemcpyHostToDevice, stream_));
  //    cudaMallocAsync(&quant_query_gpu, sizeof(int16_t) * D, stream);
  //    cudaMemcpyAsync(quant_query_gpu, quant_query,
  //                    sizeof(int16_t) * D, cudaMemcpyHostToDevice, stream);
  float* ip_results = nullptr;
  float* est_dis    = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&ip_results, sizeof(float) * num_vector_cluster, stream_));
  RAFT_CUDA_TRY(cudaMallocAsync(&est_dis, sizeof(float) * num_vector_cluster, stream_));

  stats.push_back({"copy-query-H2D", gt.stop(stream_)});

  //--------------------------------------------------------
  // C) IP kernel
  //--------------------------------------------------------
  gt.start(stream_);

  int threadsPerBlock = 256;  // must be a multiple of 32
  int warpsPerBlock   = threadsPerBlock / 32;
  int grid            = (num_vector_cluster + warpsPerBlock - 1) / warpsPerBlock;
  compute_ip_kernel_fp32<<<grid, threadsPerBlock, 0, stream_>>>(
    unit_q_gpu,
    rabitq_codes_and_factor,
    num_dimensions,
    cur_ivf.quantizer().num_short_factors(),
    num_vector_cluster,
    ip_results,
    est_dis,
    delta,
    sumq,
    y,
    one_over_sqrtD);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  stats.push_back({"kernel-ip", gt.stop(stream_)});

  //--------------------------------------------------------
  // D) select_topk_m_inplace   (it launches several kernels inside)
  //--------------------------------------------------------
  gt.start(stream_);
  size_t M  = 10;  // multiples of 10 temporally
  size_t KM = M * KNNs.capacity;

  // 2) Allocate output buffers on device:
  float* d_top_dist;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_top_dist, sizeof(float) * KM, stream_));
  uint32_t* d_top_pids;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_top_pids, sizeof(uint32_t) * KM, stream_));
  int* d_top_idx;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_top_idx, sizeof(int) * KM, stream_));

  select_topk_m_inplace(  // unchanged API, but it calls kernels ↦ must take stream too
    est_dis,
    ids,
    ip_results,
    num_vector_cluster,
    KNNs.capacity,
    M,
    d_top_dist,
    d_top_pids,
    d_top_idx,
    stream_  // <-- add parameter inside that function
  );
  stats.push_back({"kernel-select_topk*m", gt.stop(stream_)});

  //--------------------------------------------------------
  // E) long-code IP kernel
  //--------------------------------------------------------
  gt.start(stream_);
  float* d_ip2;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_ip2, sizeof(float) * KM, stream_));

  grid = (KM + warpsPerBlock - 1) / warpsPerBlock;
  compute_long_ip_kernel<<<grid, threadsPerBlock, 0, stream_>>>(
    unit_q_gpu,
    d_top_idx,
    KM,
    D,
    cur_ivf.get_ex_bits(),
    cur_cluster.long_code(cur_ivf, 0, cur_ivf.quantizer().long_code_length()),
    cur_ivf.quantizer().long_code_length(),
    d_ip2);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  stats.push_back({"kernel-long_ip", gt.stop(stream_)});

  //--------------------------------------------------------
  // F) refine kernel
  //--------------------------------------------------------
  gt.start(stream_);
  int threads = 256;
  int blocks  = (KM + threads - 1) / threads;
  // now d_top_dist is top ip instead
  refine_with_ip2_kernel<<<blocks, threads, 0, stream_>>>(
    d_top_dist,
    d_ip2,
    d_top_idx,
    sqr_y,
    FAC_RESCALE,
    sumq,
    KM,
    cur_cluster.first_block(cur_ivf),
    cur_ivf.quantizer().short_code_length() + cur_ivf.quantizer().num_short_factors(),
    cur_cluster.ex_factor(cur_ivf, 0),
    cur_ivf.quantizer().num_short_factors());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  stats.push_back({"kernel-refine", gt.stop(stream_)});

  //--------------------------------------------------------
  // G) D2H copy + write_topk_to_pool  (host work)
  //--------------------------------------------------------
  gt.start(stream_);

  cpu.start();
  auto cluster_topk_distance =
    write_topk_to_pool(KNNs, d_top_dist, d_top_pids, KM, KNNs.capacity, stream_);
  KNNs.size = KNNs.capacity;

  // set top-k for further computing (only for the first cluster)
  if (cluster_topk_distance < h_filter_distk) { h_filter_distk = cluster_topk_distance; }
  double write_ms = cpu.stop();

  stats.push_back({"select_final_topk_write_back", write_ms});

  //--------------------------------------------------------
  // H)  final report (once per probe or outside the OMP loop)
  //--------------------------------------------------------
  double total_ms = 0.0;
  for (auto& s : stats)
    total_ms += s.ms;

  std::cout << "\n---- SearchCluster timing ----\n";
  std::cout << std::left << std::setw(22) << "Step" << std::right << std::setw(12) << "ms"
            << std::setw(10) << "%" << '\n';

  for (auto& s : stats) {
    std::cout << std::left << std::setw(22) << s.name << std::right << std::setw(12) << std::fixed
              << std::setprecision(3) << s.ms << std::setw(9) << std::fixed << std::setprecision(1)
              << (s.ms * 100.0 / total_ms) << '\n';
  }
  std::cout << std::left << std::setw(22) << "TOTAL" << std::right << std::setw(12) << std::fixed
            << std::setprecision(3) << total_ms << '\n';
}

SearcherGPU::SearcherGPU(raft::resources const& handle,
                         const float* q,
                         size_t d,
                         size_t ex_bits,
                         std::string mode,
                         bool rabitq_quantize_flag)
  : D(d),
    query(q),
    one_over_sqrtD(1.0 / std::sqrt((float)D)),
    FAC_RESCALE(1 << ex_bits),
    rabitq_quantize_flag(rabitq_quantize_flag),
    mode(mode),
    handle_(handle),
    stream_(raft::resource::get_cuda_stream(handle_))
{
  unit_q      = memory::align_mm<64, float>(D * sizeof(float));
  quant_query = memory::align_mm<64, int16_t>(D * sizeof(int16_t));
  // set d_filter_distk (may be unused)
  float temp = INFINITY;
  if (mode == "quant4") {
    best_rescaling_factor = DataQuantizerGPU::get_const_scaling_factors(
      handle, d, 3);  // suppose that always quantize query to 4 bits (1 + 3) per dim
  } else if (mode == "quant8") {
    best_rescaling_factor = DataQuantizerGPU::get_const_scaling_factors(handle, d, 7);
  }
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_filter_distk, sizeof(float), stream_));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_filter_distk, &temp, sizeof(float), cudaMemcpyHostToDevice, stream_));
  raft::resource::sync_stream(handle);
}

// Need to malloc and free accordingly
SearcherGPU* SearcherGPU::CreateNewSearcherforStream(size_t d,
                                                     size_t ex_bits,
                                                     size_t num_clusters,
                                                     size_t num_vectors,
                                                     size_t k,
                                                     rmm::cuda_stream_view stream)
{
  // Basic settings
  SearcherGPU* new_searcher    = (SearcherGPU*)malloc(sizeof(SearcherGPU));
  new_searcher->D              = d;
  new_searcher->query          = nullptr;
  new_searcher->one_over_sqrtD = 1.0 / std::sqrt((float)(d));
  new_searcher->FAC_RESCALE    = 1 << ex_bits;
  new_searcher->unit_q         = memory::align_mm<64, float>(d * sizeof(float));
  new_searcher->quant_query    = memory::align_mm<64, int16_t>(d * sizeof(int16_t));
  float temp                   = INFINITY;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&(new_searcher->d_filter_distk), sizeof(float), stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    new_searcher->d_filter_distk, &temp, sizeof(float), cudaMemcpyHostToDevice, stream));

  // Additional settings for multicluster searcher
  RAFT_CUDA_TRY(
    cudaMallocAsync(&(new_searcher->d_unit_q_gpu), sizeof(float) * d * num_clusters, stream));
  RAFT_CUDA_TRY(
    cudaMallocAsync(&(new_searcher->d_ip_results), sizeof(float) * num_vectors, stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&(new_searcher->d_est_dis), sizeof(float) * num_vectors, stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&(new_searcher->d_buf), sizeof(Candidate3) * num_vectors, stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&(new_searcher->d_candidate_buffer),
                                sizeof(Candidate4) * num_vectors,
                                stream));  // very conservative estimation

  int M  = 10;
  int KM = k * M;
  RAFT_CUDA_TRY(
    cudaMallocAsync(&(new_searcher->d_top_ip), sizeof(float) * KM * num_clusters, stream));
  RAFT_CUDA_TRY(
    cudaMallocAsync(&(new_searcher->d_top_pids), sizeof(PID) * KM * num_clusters, stream));
  RAFT_CUDA_TRY(
    cudaMallocAsync(&(new_searcher->d_top_idx), sizeof(int) * KM * num_clusters, stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&(new_searcher->d_ip2), sizeof(float) * KM * num_clusters, stream));
  RAFT_CUDA_TRY(
    cudaMallocAsync(&(new_searcher->d_sum_norm), sizeof(SumNorm) * num_clusters, stream));

  return new_searcher;
}

// ------------------------------------------------------------------
//  Packed record used *only* in the overflow path
// ------------------------------------------------------------------
struct __align__(16) Packed {
  float l2;      // key
  float ip;      // payload
  PID pid;       // payload
  uint32_t idx;  // payload
};
static_assert(sizeof(Packed) == 16, "Packed must stay 16 bytes");

// ==================================================================
//  Kernel helpers
// ==================================================================
__global__ void makeFlagsKernel(const float* __restrict__ d_l2,
                                uint8_t* d_flags,
                                float thr,
                                std::size_t N)
{
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) d_flags[i] = d_l2[i] < thr;
}

__global__ void fillPackedKernel(const float* __restrict__ d_l2,
                                 const float* __restrict__ d_ip,
                                 const PID* __restrict__ d_pid,
                                 Packed* d_out,
                                 std::size_t N)
{
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) { d_out[i] = {d_l2[i], d_ip[i], d_pid[i], static_cast<uint32_t>(i)}; }
}

__global__ void scatterCheapKernel(const float* __restrict__ d_l2,
                                   const float* __restrict__ d_ip,
                                   const PID* __restrict__ d_pid,
                                   const uint32_t* d_prefix,
                                   const uint8_t* d_flags,
                                   std::size_t KM,
                                   float* d_top_ip,
                                   PID* d_top_pid,
                                   int* d_top_idx)
{
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!d_flags[i]) return;  // not selected
  std::size_t pos = d_prefix[i];
  if (pos >= KM) return;  // overflow ignored
  d_top_ip[pos]  = d_ip[i];
  d_top_pid[pos] = d_pid[i];
  d_top_idx[pos] = static_cast<int>(i);
}

__global__ void scatterPackedKernel(const Packed* __restrict__ d_src,
                                    std::size_t written,
                                    float* d_top_ip,
                                    PID* d_top_pid,
                                    int* d_top_idx)
{
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < written) {
    const Packed p = d_src[i];
    d_top_ip[i]    = p.ip;
    d_top_pid[i]   = p.pid;
    d_top_idx[i]   = static_cast<int>(p.idx);
  }
}

__global__ void copyKeysKernel(const Packed* __restrict__ src, float* dst, std::size_t n)
{
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i].l2;
}

// ==================================================================
//  Public API
// ==================================================================
/**
 * @return number of elements written into (d_top_ip, d_top_pid, d_top_idx)
 */
inline std::size_t selectTopKM(const float* d_est_dist,
                               const float* d_ip_results,
                               const PID* d_ids,
                               std::size_t num_vector_cluster,
                               std::size_t KM,
                               float filter_dist_k,
                               /*out*/ float* d_top_ip,
                               /*out*/ PID* d_top_pid,
                               /*out*/ int* d_top_idx,
                               rmm::cuda_stream_view stream)
{
  const std::size_t N = num_vector_cluster;
  constexpr int BLOCK = 256;
  const int GRID      = static_cast<int>((N + BLOCK - 1) / BLOCK);

  // ------------------------------------------------------------------
  //  1. Build flag array ― d_flags[i]=1 if d_est_dist[i] < filter
  // ------------------------------------------------------------------
  uint8_t* d_flags;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_flags, N, stream));
  makeFlagsKernel<<<GRID, BLOCK, 0, stream>>>(d_est_dist, d_flags, filter_dist_k, N);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // ------------------------------------------------------------------
  //  2. Count how many passed the filter  (DeviceReduce::Sum)
  // ------------------------------------------------------------------
  uint32_t* d_selected;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_selected, sizeof(uint32_t), stream));
  void* temp             = nullptr;
  std::size_t temp_bytes = 0;
  cub::DeviceReduce::Sum(temp, temp_bytes, d_flags, d_selected, N, stream);
  RAFT_CUDA_TRY(cudaMallocAsync(&temp, temp_bytes, stream));
  cub::DeviceReduce::Sum(temp, temp_bytes, d_flags, d_selected, N, stream);

  uint32_t h_selected = 0;
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(&h_selected, d_selected, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  // ------------------------------------------------------------------
  //  Decide path
  // ------------------------------------------------------------------
  if (h_selected == 0) {
    // nothing to write
    RAFT_CUDA_TRY(cudaFreeAsync(d_flags, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_selected, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(temp, stream));
    return 0;
  }

  if (h_selected < KM) {
    /**********************  CHEAP PATH  ***************************/
    // prefix sum on flags → d_prefix
    uint32_t* d_prefix;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_prefix, N * sizeof(uint32_t), stream));
    // reuse temp buffer
    temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_flags, d_prefix, N, stream);
    RAFT_CUDA_TRY(cudaFreeAsync(temp, stream));
    RAFT_CUDA_TRY(cudaMallocAsync(&temp, temp_bytes, stream));
    cub::DeviceScan::ExclusiveSum(temp, temp_bytes, d_flags, d_prefix, N, stream);

    // scatter directly
    scatterCheapKernel<<<GRID, BLOCK, 0, stream>>>(
      d_est_dist, d_ip_results, d_ids, d_prefix, d_flags, KM, d_top_ip, d_top_pid, d_top_idx);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // cleanup
    RAFT_CUDA_TRY(cudaFreeAsync(d_prefix, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_flags, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_selected, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(temp, stream));
    return static_cast<std::size_t>(h_selected);
  } else {
    // --- 3.a  compact AoS  ------------------------------------------------
    Packed* d_in;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_in, N * sizeof(Packed), stream));
    fillPackedKernel<<<GRID, BLOCK, 0, stream>>>(d_est_dist, d_ip_results, d_ids, d_in, N);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // --- 3.b  flag–select  ------------------------------------------------
    Packed* d_sel_val;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_sel_val, N * sizeof(Packed), stream));
    uint8_t* d_sel_key;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_sel_key, N, stream));  // reuse flags as keys

    std::size_t temp_sel = 0;
    cub::DeviceSelect::Flagged(nullptr,
                               temp_sel,
                               d_in,
                               d_flags,
                               d_sel_val,
                               d_selected,  // counter
                               N,
                               stream);
    RAFT_CUDA_TRY(cudaFreeAsync(temp, stream));
    RAFT_CUDA_TRY(cudaMallocAsync(&temp, temp_sel, stream));
    cub::DeviceSelect::Flagged(temp, temp_sel, d_in, d_flags, d_sel_val, d_selected, N, stream);

    uint32_t sel = 0;
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(&sel, d_selected, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    // --- 3.c  build dense key array (L2) ----------------------------------
    float* d_key_in;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_key_in, sel * sizeof(float), stream));
    float* d_key_out;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_key_out, sel * sizeof(float), stream));
    Packed* d_val_out;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_val_out, sel * sizeof(Packed), stream));

    // kernel: copy l2 into dense array
    int gridK = (sel + BLOCK - 1) / BLOCK;
    copyKeysKernel<<<gridK, BLOCK, 0, stream>>>(d_sel_val, d_key_in, sel);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // --- 3.d  radix-sort (key, payload) -----------------------------------
    void* tmp_sort             = nullptr;
    std::size_t tmp_sort_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
      tmp_sort, tmp_sort_bytes, d_key_in, d_key_out, d_sel_val, d_val_out, sel, 0, 32, stream);
    RAFT_CUDA_TRY(cudaMallocAsync(&tmp_sort, tmp_sort_bytes, stream));
    cub::DeviceRadixSort::SortPairs(
      tmp_sort, tmp_sort_bytes, d_key_in, d_key_out, d_sel_val, d_val_out, sel, 0, 32, stream);

    // --- 3.e  scatter first KM to user buffers ----------------------------
    scatterPackedKernel<<<(KM + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
      d_val_out, KM, d_top_ip, d_top_pid, d_top_idx);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // --- cleanup ----------------------------------------------------------
    RAFT_CUDA_TRY(cudaFreeAsync(d_in, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_sel_val, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_key_in, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_key_out, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_val_out, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(tmp_sort, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_flags, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_selected, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(temp, stream));
    return KM;
  }
}

/* --------------------------------  API  -------------------------------------------------- */
/// \param  est_dist      [N]  – estimated L2 distances
/// \param  ids           [N]  – PIDs that belong to each distance
/// \param  ip_results    [N]  – inner products that belong to each distance
/// \param  N                    number of vectors in the three arrays above
/// \param  KM                   TOPK * M  (maximum number of results to keep)
/// \param  filter_dist_k        keep only distances  <  filter_dist_k
/// \param  d_top_ip      [KM] –  OUT  inner products that survive
/// \param  d_top_pids    [KM] –  OUT  PIDs            that survive
/// \param  d_top_idx     [KM] –  OUT  original indices that survive
/// \param  stream               CUDA stream to run everything on
/// \return                        number of elements written to each OUT array ( ≤ KM )
///
inline int select_and_keep_topKM_by_distance(const float* est_dist,
                                             const uint32_t* ids,
                                             const float* ip_results,
                                             std::size_t N,
                                             int KM,
                                             float filter_dist_k,
                                             float* d_top_ip,       // OUT
                                             uint32_t* d_top_pids,  // OUT
                                             int* d_top_idx,        // OUT
                                             rmm::cuda_stream_view stream)
{
  if (N == 0 || KM == 0) return 0;  // trivial corner case

  /* ------------------------------------------------------------------ *
   *  1)  Copy all indices i∈[0,N) whose est_dist[i] < filter_dist_k    *
   * ------------------------------------------------------------------ */
  thrust::device_vector<int> selected_idx(N);  // temporary
  auto first = thrust::make_counting_iterator<std::size_t>(0);

  auto sel_end = thrust::copy_if(  // indices → selected_idx
    thrust::cuda::par.on(stream),
    first,
    first + N,  // input  (0,1,2,…)
    est_dist,   // stencil
    selected_idx.begin(),
    [filter_dist_k] __device__(float d) { return d < filter_dist_k; });

  const int sel_cnt = static_cast<int>(sel_end - selected_idx.begin());

  if (sel_cnt == 0) return 0;  // nothing survived, we’re done

  /* ------------------------------------------------------------------ *
   *  2)  If sel_cnt ≤ KM  ➜ just gather                                *
   *      else          ➜ sort selected candidates by distance          *
   * ------------------------------------------------------------------ */
  const int out_cnt = (sel_cnt <= KM) ? sel_cnt : KM;  // how many we will finally write

  /* --- gather (distance, pid, ip) of the still-alive indices -------------------------- */
  thrust::device_vector<float> dist_tmp(sel_cnt);
  thrust::device_vector<float> ip_tmp(sel_cnt);
  thrust::device_vector<uint32_t> pid_tmp(sel_cnt);

  thrust::gather(
    thrust::cuda::par.on(stream), selected_idx.begin(), sel_end, est_dist, dist_tmp.begin());
  thrust::gather(
    thrust::cuda::par.on(stream), selected_idx.begin(), sel_end, ip_results, ip_tmp.begin());
  thrust::gather(thrust::cuda::par.on(stream), selected_idx.begin(), sel_end, ids, pid_tmp.begin());

  /* --- if we have more than KM survivors, partial-sort by distance -------------------- */
  if (sel_cnt > KM) {
    thrust::sort_by_key(
      thrust::cuda::par.on(stream),
      dist_tmp.begin(),
      dist_tmp.end(),             // key  = distance
      thrust::make_zip_iterator(  // value = (idx, ip, pid)
        thrust::make_tuple(selected_idx.begin(), ip_tmp.begin(), pid_tmp.begin())));
    /* after sort, the first KM entries are the ones we keep */
  }

  /* ------------------------------------------------------------------ *
   *  3)  Scatter the first out_cnt items into the user’s output arrays *
   * ------------------------------------------------------------------ */
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_top_idx,
                                thrust::raw_pointer_cast(selected_idx.data()),
                                sizeof(int) * out_cnt,
                                cudaMemcpyDeviceToDevice,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_top_ip,
                                thrust::raw_pointer_cast(ip_tmp.data()),
                                sizeof(float) * out_cnt,
                                cudaMemcpyDeviceToDevice,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_top_pids,
                                thrust::raw_pointer_cast(pid_tmp.data()),
                                sizeof(uint32_t) * out_cnt,
                                cudaMemcpyDeviceToDevice,
                                stream));

  return out_cnt;  // ← how many items were written
}

__global__ void splitKernel(
  const Candidate3* in, float* out_ip, uint32_t* out_pid, int* out_idx, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out_ip[i]  = in[i].ip;
    out_pid[i] = in[i].pid;
    out_idx[i] = in[i].idx;
  }
}

//------------------------------------------------------------------------------
// pass‑1 kernel: filter + gather in a single sweep
//------------------------------------------------------------------------------
__global__ void filterGatherKernel(const float* dist,
                                   const float* ip,
                                   const uint32_t* pid,
                                   int N,
                                   float threshold,
                                   Candidate3* out,
                                   int* d_counter)
{
  using BlockScan = cub::BlockScan<int, 256>;
  __shared__ BlockScan::TempStorage scanTmp;
  __shared__ int blockBase;

  int tid  = blockIdx.x * blockDim.x + threadIdx.x;
  int flag = 0;
  float vdist, vip;
  uint32_t vpid;

  if (tid < N) {
    vdist = dist[tid];
    flag  = (vdist < threshold);
    if (flag) {
      vip  = ip[tid];
      vpid = pid[tid];
    }
  }

  int offset, blockTotal;
  BlockScan(scanTmp).ExclusiveSum(flag, offset, blockTotal);

  if (threadIdx.x == 0) blockBase = atomicAdd(d_counter, blockTotal);
  __syncthreads();
  //    printf("d_counter: %d\n", *d_counter);

  if (flag) {
    int pos = blockBase + offset;
    //        printf("Block base: %d\n", blockBase);
    //        printf("Offset: %d\n", offset);
    //        printf("d_counter: %d\n", *d_counter);
    //        if (blockIdx.x == 0 && flag && offset < 4)
    //            printf("tid %d  offset %d  pos %d  dist %.3f\n",
    //                   tid, offset, pos, vdist);
    out[pos] = {vdist, vip, vpid, tid};
  }

  //    if (blockIdx.x < 4 && threadIdx.x == 0)          // first 4 blocks only
  //        printf("block %d: blockTotal = %d  blockBase = %d\n",
  //               blockIdx.x, blockTotal, blockBase);
}

//------------------------------------------------------------------------------
// pass‑1 kernel: filter + gather in a single sweep V2: for Memopt V2
//------------------------------------------------------------------------------
__global__ void filterGatherKernelV2(const float* dist,
                                     const float* ip,
                                     const uint32_t* pid,
                                     int N,
                                     float threshold,
                                     Candidate3* out,
                                     int* d_counter)
{
  using BlockScan = cub::BlockScan<int, 256>;
  __shared__ BlockScan::TempStorage scanTmp;
  __shared__ int blockBase;

  int tid  = blockIdx.x * blockDim.x + threadIdx.x;
  int flag = 0;
  float vdist, vip;
  uint32_t vpid;

  if (tid < N) {
    vdist = dist[tid];
    flag  = (vdist != 0.0f);
    if (flag) {
      vip  = ip[tid];
      vpid = pid[tid];
    }
  }

  int offset, blockTotal;
  BlockScan(scanTmp).ExclusiveSum(flag, offset, blockTotal);

  if (threadIdx.x == 0) blockBase = atomicAdd(d_counter, blockTotal);
  __syncthreads();
  //    printf("d_counter: %d\n", *d_counter);

  if (flag) {
    int pos  = blockBase + offset;
    out[pos] = {vdist, vip, vpid, tid};
  }
}

__global__ void filterGatherKernel_SoA(const float* __restrict__ dist,
                                       const float* __restrict__ ip,
                                       const uint32_t* __restrict__ pid,
                                       int N,
                                       float threshold,
                                       float* out_ip,
                                       uint32_t* out_pid,
                                       int* out_idx,
                                       int* d_counter,  // atomic counter ≤ KM
                                       int KM)
{
  using BlockScan = cub::BlockScan<int, 256>;
  __shared__ BlockScan::TempStorage scanTmp;
  __shared__ int blockBase;

  int tid  = blockIdx.x * blockDim.x + threadIdx.x;
  int flag = 0;
  float vip;
  uint32_t vpid;

  if (tid < N) {
    float vdist = dist[tid];
    flag        = (vdist < threshold);
    if (flag) {
      vip  = ip[tid];
      vpid = pid[tid];
    }
  }

  int offset, blockTotal;
  BlockScan(scanTmp).ExclusiveSum(flag, offset, blockTotal);

  // reserve space only up to KM
  if (threadIdx.x == 0) blockBase = atomicAdd(d_counter, blockTotal);
  __syncthreads();

  // discard excess survivors beyond KM
  int pos = blockBase + offset;
  if (flag && pos < KM) {
    out_ip[pos]  = vip;
    out_pid[pos] = vpid;
    out_idx[pos] = tid;
  }
}

__global__ void copyDistKernel(const Candidate3* __restrict__ src, float* __restrict__ dst, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i].dist;
}

__global__ void copyDistKernel4(const Candidate4* __restrict__ src, float* __restrict__ dst, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i].dist;
}

//------------------------------------------------------------------------------
// host wrapper
//------------------------------------------------------------------------------
int fast_select_and_keep(const float* d_dist,
                         const float* d_ip,
                         const uint32_t* d_pid,
                         int N,
                         int KM,
                         float threshold,
                         float* d_top_ip,
                         uint32_t* d_top_pid,
                         int* d_top_idx,
                         rmm::cuda_stream_view stream)
{
  // 1) allocate worst‑case survivor buffer  (in practise you can reuse one)
  Candidate3* d_buf;
  RAFT_CUDA_TRY(cudaMallocAsync(
    &d_buf, sizeof(Candidate3) * N, stream));  // oversize ok (N for storage, N for sorting)
  int* d_counter;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_counter, sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counter, 0, sizeof(int), stream));

  int BLK = 256, GRD = (N + BLK - 1) / BLK;
  int sel_cnt;

  // 2) launch pass‑1
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counter, 0, sizeof(int), stream));
  filterGatherKernel<<<GRD, BLK, 0, stream>>>(d_dist, d_ip, d_pid, N, threshold, d_buf, d_counter);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // 3) fetch survivor count
  RAFT_CUDA_TRY(cudaMemcpyAsync(&sel_cnt, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  if (sel_cnt == 0) {
    RAFT_CUDA_TRY(cudaFreeAsync(d_buf, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_counter, stream));
    return 0;
  }

  if (sel_cnt > KM) {
    //------------------------------------------------------------------
    // 4‑A)  build a contiguous array of distance keys
    //------------------------------------------------------------------
    float* d_keys;  // [sel_cnt] float
    RAFT_CUDA_TRY(cudaMallocAsync(&d_keys, sel_cnt * sizeof(float), stream));

    // one kernel copy is faster than Thrust here
    int BLK = 256, GRD = (sel_cnt + BLK - 1) / BLK;
    copyDistKernel<<<GRD, BLK, 0, stream>>>(d_buf, d_keys, sel_cnt);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    //------------------------------------------------------------------
    // 4‑B)  radix‑sort ( key = d_keys , value = d_buf )
    //------------------------------------------------------------------
    size_t tmpBytes = 0;
    cub::DeviceRadixSort::SortPairs(
      /*temp=*/nullptr,
      tmpBytes,
      d_keys,
      d_keys,  // in‑place: keys_in == keys_out
      d_buf,
      d_buf,  // in‑place: values_in == values_out
      sel_cnt,
      0,
      32,
      stream);  // 0‑31 bits are enough for floats

    void* d_tmp;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_tmp, tmpBytes, stream));
    cub::DeviceRadixSort::SortPairs(
      d_tmp, tmpBytes, d_keys, d_keys, d_buf, d_buf, sel_cnt, 0, 32, stream);
    RAFT_CUDA_TRY(cudaFreeAsync(d_tmp, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_keys, stream));
    //  d_buf is now sorted by ascending dist
  }

  // 5) copy first out_cnt elements to user buffers
  int out_cnt = min(sel_cnt, KM);

  GRD = (out_cnt + BLK - 1) / BLK;
  splitKernel<<<GRD, BLK, 0, stream>>>(d_buf, d_top_ip, d_top_pid, d_top_idx, out_cnt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaFreeAsync(d_buf, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_counter, stream));
  return out_cnt;
}

// v2: for memoptv2
int fast_select_and_keepv2(const float* d_dist,
                           const float* d_ip,
                           const uint32_t* d_pid,
                           int N,
                           int KM,
                           float threshold,
                           float* d_top_ip,
                           uint32_t* d_top_pid,
                           int* d_top_idx,
                           rmm::cuda_stream_view stream)
{
  // 1) allocate worst‑case survivor buffer  (in practise you can reuse one)
  Candidate3* d_buf;
  RAFT_CUDA_TRY(cudaMallocAsync(
    &d_buf, sizeof(Candidate3) * N, stream));  // oversize ok (N for storage, N for sorting)
  int* d_counter;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_counter, sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counter, 0, sizeof(int), stream));

  int BLK = 256, GRD = (N + BLK - 1) / BLK;
  int sel_cnt;

  // 2) launch pass‑1
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counter, 0, sizeof(int), stream));
  filterGatherKernelV2<<<GRD, BLK, 0, stream>>>(
    d_dist, d_ip, d_pid, N, threshold, d_buf, d_counter);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // 3) fetch survivor count
  RAFT_CUDA_TRY(cudaMemcpyAsync(&sel_cnt, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  if (sel_cnt == 0) {
    RAFT_CUDA_TRY(cudaFreeAsync(d_buf, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_counter, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    return 0;
  }

  if (sel_cnt > KM) {
    //------------------------------------------------------------------
    // 4‑A)  build a contiguous array of distance keys
    //------------------------------------------------------------------
    float* d_keys;  // [sel_cnt] float
    RAFT_CUDA_TRY(cudaMallocAsync(&d_keys, sel_cnt * sizeof(float), stream));

    // one kernel copy is faster than Thrust here
    int BLK = 256, GRD = (sel_cnt + BLK - 1) / BLK;
    copyDistKernel<<<GRD, BLK, 0, stream>>>(d_buf, d_keys, sel_cnt);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    //------------------------------------------------------------------
    // 4‑B)  radix‑sort ( key = d_keys , value = d_buf )
    //------------------------------------------------------------------
    size_t tmpBytes = 0;
    cub::DeviceRadixSort::SortPairs(
      /*temp=*/nullptr,
      tmpBytes,
      d_keys,
      d_keys,  // in‑place: keys_in == keys_out
      d_buf,
      d_buf,  // in‑place: values_in == values_out
      sel_cnt,
      0,
      32,
      stream);  // 0‑31 bits are enough for floats

    void* d_tmp;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_tmp, tmpBytes, stream));
    cub::DeviceRadixSort::SortPairs(
      d_tmp, tmpBytes, d_keys, d_keys, d_buf, d_buf, sel_cnt, 0, 32, stream);
    RAFT_CUDA_TRY(cudaFreeAsync(d_tmp, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_keys, stream));
    //  d_buf is now sorted by ascending dist
  }

  // 5) copy first out_cnt elements to user buffers
  int out_cnt = min(sel_cnt, KM);

  GRD = (out_cnt + BLK - 1) / BLK;
  splitKernel<<<GRD, BLK, 0, stream>>>(d_buf, d_top_ip, d_top_pid, d_top_idx, out_cnt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaFreeAsync(d_buf, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_counter, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return out_cnt;
}

inline int fast_select_and_keep_timed(const float* d_dist,
                                      const float* d_ip,
                                      const uint32_t* d_pid,
                                      int N,
                                      int KM,
                                      float threshold,
                                      float* d_top_ip,
                                      uint32_t* d_top_pid,
                                      int* d_top_idx,
                                      rmm::cuda_stream_view stream)
{
  std::vector<StepStat> s(6);  // 0..4 GPU, 5 host
  CpuTimer host;
  host.start();

  // ───────────────────────── scratch ──────────────────────────
  GpuTimer gt;
  gt.start(stream);
  Candidate3* d_buf;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_buf, sizeof(Candidate3) * N, stream));
  int* d_cnt;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_cnt, sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_cnt, 0, sizeof(int), stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  s[0] = {"scratch alloc + memset", gt.stop(stream)};

  // ───────────────────────── pass‑1 filter+gather ─────────────
  const int BLK = 256, GRD = (N + BLK - 1) / BLK;
  gt.start(stream);
  filterGatherKernel<<<GRD, BLK, 0, stream>>>(d_dist, d_ip, d_pid, N, threshold, d_buf, d_cnt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  s[1] = {"filter+gather kernel", gt.stop(stream)};

  // ───────────────────────── fetch sel_cnt ────────────────────
  int sel_cnt;
  RAFT_CUDA_TRY(cudaMemcpyAsync(&sel_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));  // wait so sel_cnt is valid

  //    if (sel_cnt == 0) {
  //        cudaFree(d_buf);  cudaFree(d_cnt);
  //        s[5] = {"host overhead", host.stop()};
  //        goto print_stats;
  //    }

  // ───────────────────────── optional sort path ───────────────
  if (sel_cnt > KM) {
    gt.start(stream);
    float* d_keys;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_keys, sel_cnt * sizeof(float), stream));
    int GRD2 = (sel_cnt + BLK - 1) / BLK;
    copyDistKernel<<<GRD2, BLK, 0, stream>>>(d_buf, d_keys, sel_cnt);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    size_t tmpBytes = 0;
    cub::DeviceRadixSort::SortPairs(
      nullptr, tmpBytes, d_keys, d_keys, d_buf, d_buf, sel_cnt, 0, 32, stream);
    void* d_tmp;
    RAFT_CUDA_TRY(cudaMalloc(&d_tmp, tmpBytes));
    cub::DeviceRadixSort::SortPairs(
      d_tmp, tmpBytes, d_keys, d_keys, d_buf, d_buf, sel_cnt, 0, 32, stream);
    RAFT_CUDA_TRY(cudaFreeAsync(d_tmp, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_keys, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    s[2] = {"copyDist + radix‑sort", gt.stop(stream)};
  } else {
    s[2] = {"copyDist + radix‑sort (skipped)", 0.0};
  }

  // ───────────────────────── split to SoA ─────────────────────
  const int out_cnt = std::min(sel_cnt, KM);
  gt.start(stream);
  int GRD3 = (out_cnt + BLK - 1) / BLK;
  splitKernel<<<GRD3, BLK, 0, stream>>>(d_buf, d_top_ip, d_top_pid, d_top_idx, out_cnt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));  // ensure kernel done
  s[3] = {"splitKernel (AoS→SoA)", gt.stop(stream)};

  // ───────────────────────── cleanup ──────────────────────────
  gt.start(stream);
  RAFT_CUDA_TRY(cudaFreeAsync(d_buf, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_cnt, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  s[4] = {"free scratch", gt.stop(stream)};

  s[5] = {"host overhead", host.stop()};

print_stats:
  // ───────────────────────── pretty print ─────────────────────
  double gpu_total = std::accumulate(
    s.begin(), s.begin() + 5, 0.0, [](double a, const StepStat& st) { return a + st.ms; });

  printf("\n╔══════════════════════════════════╦═══════════╦══════════╗\n");
  printf("║  step                           ║   ms      ║ percent  ║\n");
  printf("╠══════════════════════════════════╬═══════════╬══════════╣\n");
  for (int i = 0; i < 5; ++i)
    printf("║ %-30s ║ %8.2f ║ %7.1f %% ║\n",
           s[i].name,
           s[i].ms,
           gpu_total ? (s[i].ms / gpu_total * 100.0) : 0.0);
  printf("╠══════════════════════════════════╬═══════════╬══════════╣\n");
  printf("║ TOTAL GPU time                  ║ %8.2f ║ 100 %%    ║\n", gpu_total);
  printf("╚══════════════════════════════════╩═══════════╩══════════╝\n");
  printf("Host overhead: %.2f ms\n\n", s[5].ms);

  return (sel_cnt > KM ? KM : sel_cnt);
}

void SearcherGPU::SearchClusterWithFilter(const IVFGPU& cur_ivf,
                                          const IVFGPU::GPUClusterMeta& cur_cluster,
                                          float sqr_y,
                                          DeviceResultPool& KNNs,
                                          float* centroid_data)
{
  // related information
#ifdef COUNT_CLUSTER_TIME
  struct StepStat {
    std::string name;
    double ms = 0;
  };
  std::vector<StepStat> stats;  // push in call order
#endif
  size_t num_vector_cluster         = cur_cluster.num;
  size_t num_dimensions             = cur_ivf.get_num_padded_dim();
  uint32_t* rabitq_codes_and_factor = cur_cluster.first_block(cur_ivf);
  PID* ids                          = cur_cluster.ids(cur_ivf);

  //--------------------------------------------------------
// A) host-side scalar preparation
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  CpuTimer cpu;
  cpu.start();
#endif
  float y     = std::sqrt(sqr_y);
  this->shift = 0;
  this->sumq  = normalize_query16_scalar(unit_q, query, centroid_data, y, D);
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"host-prep", cpu.stop()});
#endif
//--------------------------------------------------------
// B) H2D copies of query + buffers (GPU time)
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  GpuTimer gt;
  gt.start(stream_);
#endif
  float* unit_q_gpu = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&unit_q_gpu, sizeof(float) * num_dimensions, stream_));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    unit_q_gpu, unit_q, sizeof(float) * num_dimensions, cudaMemcpyHostToDevice, stream_));
  //    cudaMallocAsync(&quant_query_gpu, sizeof(int16_t) * D, stream);
  //    cudaMemcpyAsync(quant_query_gpu, quant_query,
  //                    sizeof(int16_t) * D, cudaMemcpyHostToDevice, stream);
  float* ip_results = nullptr;
  float* est_dis    = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&ip_results, sizeof(float) * num_vector_cluster, stream_));
  RAFT_CUDA_TRY(cudaMallocAsync(&est_dis, sizeof(float) * num_vector_cluster, stream_));
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"copy-query-H2D", gt.stop(stream_)});
#endif
  //--------------------------------------------------------
  // C) IP kernel
  //--------------------------------------------------------

#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  int threadsPerBlock = 256;  // must be a multiple of 32
  int warpsPerBlock   = threadsPerBlock / 32;
  int grid            = (num_vector_cluster + warpsPerBlock - 1) / warpsPerBlock;
  compute_ip_kernel_fp32<<<grid, threadsPerBlock, 0, stream_>>>(
    unit_q_gpu,
    rabitq_codes_and_factor,
    num_dimensions,
    cur_ivf.quantizer().num_short_factors(),
    num_vector_cluster,
    ip_results,
    est_dis,
    delta,
    sumq,
    y,
    one_over_sqrtD);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-ip", gt.stop(stream_)});
#endif

  // (D) apply filters and restore K
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  //    printf("trying to filter and gather...");
  size_t M  = 10;  // multiples of 10 temporally
  size_t KM = M * KNNs.capacity;
  float* d_top_ip;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_top_ip, sizeof(float) * KM, stream_));
  uint32_t* d_top_pids;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_top_pids, sizeof(uint32_t) * KM, stream_));
  int* d_top_idx;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_top_idx, sizeof(int) * KM, stream_));
  //    int written = select_and_keep_topKM_by_distance(
  //            est_dis, ids, ip_results,
  //            num_vector_cluster,              // num_vector_cluster
  //            KM,             // TOPK * M
  //            h_filter_distk,
  //            d_top_ip,     // same pointer names you used
  //            d_top_pids,
  //            d_top_idx,
  //            stream);     // or 0 for the default stream
  int written = fast_select_and_keep(est_dis,
                                     ip_results,
                                     ids,
                                     num_vector_cluster,
                                     KM,
                                     h_filter_distk,
                                     d_top_ip,
                                     d_top_pids,
                                     d_top_idx,
                                     stream_);
//    int written = fused_filter_gather(est_dis, ip_results, ids, num_vector_cluster, KM,
//    h_filter_distk, d_top_ip, d_top_pids, d_top_idx, stream);
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-filter-select_topk*m", gt.stop(stream_)});
#endif

  //    int written = (int)selectTopKM(est_dis, ip_results, ids,
  //            num_vector_cluster,
  //            KM,                      // TOPK * M
  //            h_filter_distk,
  //            d_top_ip, d_top_pids, d_top_idx,
  //            stream);

  // original correct version
//    select_topk_m_inplace(            // unchanged API, but it calls kernels ↦ must take stream
//    too
//            est_dis, ids, ip_results,
//            num_vector_cluster,
//            KNNs->capacity, M,
//            d_top_ip, d_top_pids, d_top_idx,
//            stream                           // <-- add parameter inside that function
//    );
//    written = KM;
//    printf("written: %d\n", written);
//--------------------------------------------------------
// E) long-code IP kernel
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  float* d_ip2;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_ip2, sizeof(float) * KM, stream_));

  grid = (written + warpsPerBlock - 1) / warpsPerBlock;
  compute_long_ip_kernel<<<grid, threadsPerBlock, 0, stream_>>>(
    unit_q_gpu,
    d_top_idx,
    written,
    D,
    cur_ivf.get_ex_bits(),
    cur_cluster.long_code(cur_ivf, 0, cur_ivf.quantizer().long_code_length()),
    cur_ivf.quantizer().long_code_length(),
    d_ip2);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-long_ip", gt.stop(stream_)});
#endif
//--------------------------------------------------------
// F) refine kernel
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  int threads = 256;
  int blocks  = (written + threads - 1) / threads;
  // now d_top_dist is top ip instead
  refine_with_ip2_kernel<<<blocks, threads, 0, stream_>>>(
    d_top_ip,
    d_ip2,
    d_top_idx,
    sqr_y,
    FAC_RESCALE,
    sumq,
    written,
    cur_cluster.first_block(cur_ivf),
    cur_ivf.quantizer().short_code_length() + cur_ivf.quantizer().num_short_factors(),
    cur_cluster.ex_factor(cur_ivf, 0),
    cur_ivf.quantizer().num_short_factors());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-refine", gt.stop(stream_)});
#endif

//--------------------------------------------------------
// G) D2H copy + write_topk_to_pool  (host work)
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  //    cpu.start();
  //    printf("writing back results... \n");
  if (written > KNNs.capacity) {
    //        sort_num += 1;
    //        printf("writing topk to pool, written: %zu \n", written);
    auto cluster_topk_distance =
      write_topk_to_pool(KNNs, d_top_ip, d_top_pids, written, KNNs.capacity, stream_);
    KNNs.size = KNNs.capacity;
    //        printf("Before assignment:\n");
    //        printf("  h_filter_distk        = %.6f\n", h_filter_distk);
    //
    if (h_filter_distk > cluster_topk_distance) { h_filter_distk = cluster_topk_distance; }

    //        printf("After assignment:\n");
    //        printf("  h_filter_distk        = %.6f\n", h_filter_distk);
  } else {
    // copy directly without sorting
    //        direct_num += 1;
    //        printf("copying directly \n");
    raft::copy(KNNs.distances.data_handle(), d_top_ip, written, stream_);
    raft::copy(KNNs.ids.data_handle(), d_top_pids, written, stream_);

    KNNs.size = written;
  }
  //    printf("writing finished \n");

  // TODO: update h_filter
//   f write_ms = gt.stop();
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"select_final_topk_write_back", gt.stop()});

  //--------------------------------------------------------
  // H)  final report (once per probe or outside the OMP loop)
  //--------------------------------------------------------
  double total_ms = 0.0;
  for (auto& s : stats)
    total_ms += s.ms;

  std::cout << "\n---- SearchCluster timing ----\n";
  std::cout << std::left << std::setw(22) << "Step" << std::right << std::setw(12) << "ms"
            << std::setw(10) << "%" << '\n';

  for (auto& s : stats) {
    std::cout << std::left << std::setw(22) << s.name << std::right << std::setw(12) << std::fixed
              << std::setprecision(3) << s.ms << std::setw(9) << std::fixed << std::setprecision(1)
              << (s.ms * 100.0 / total_ms) << '\n';
  }
  std::cout << std::left << std::setw(22) << "TOTAL" << std::right << std::setw(12) << std::fixed
            << std::setprecision(3) << total_ms << '\n';
#endif

  //    RAFT_CUDA_TRY(cudaFree(quant_query_gpu));
  //    RAFT_CUDA_TRY(cudaFree(rotated_query_gpu));
  //    cudaFreeAsync(rotated_query_gpu, stream);
  //    cudaFreeAsync(quant_query_gpu, stream);
  RAFT_CUDA_TRY(cudaFreeAsync(ip_results, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(est_dis, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_top_ip, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_top_pids, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_top_idx, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(unit_q_gpu, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_ip2, stream_));
}

inline uint32_t extract_code_cpu(const uint8_t* codes, size_t d, size_t EX_BITS)
{
  size_t bitPos    = d * EX_BITS;
  size_t byteIdx   = bitPos >> 3;  // bitPos/8
  size_t bitOffset = bitPos & 7;   // bitPos%8
  // grab enough bits across this byte (and maybe the next)
  uint32_t v = codes[byteIdx] << 8;
  if (bitOffset + EX_BITS > 8) { v |= codes[byteIdx + 1]; }
  // now top 16 bits of v hold at least (bitOffset+EX_BITS) bits—we shift:
  int shift = 16 - (bitOffset + EX_BITS);
  return (v >> shift) & ((1u << EX_BITS) - 1);
}

void SearcherGPU::SearchClusterWithFilterMemOptOffload(const IVFGPU& cur_ivf,
                                                       const IVFGPU::GPUClusterMeta& cur_cluster,
                                                       float sqr_y,
                                                       BoundedKNN* KNNs,
                                                       float* centroid_data)
{
  // related information
#ifdef COUNT_CLUSTER_TIME
  struct StepStat {
    std::string name;
    double ms = 0;
  };
  std::vector<StepStat> stats;  // push in call order
#endif
  size_t num_vector_cluster         = cur_cluster.num;
  size_t num_dimensions             = cur_ivf.get_num_padded_dim();
  uint32_t* rabitq_codes_and_factor = cur_cluster.first_block(cur_ivf);
  PID* ids                          = cur_cluster.ids(cur_ivf);

  //--------------------------------------------------------
// A) host-side scalar preparation
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  CpuTimer cpu;
  cpu.start();
#endif
  float y     = std::sqrt(sqr_y);
  this->shift = 0;
  this->sumq  = normalize_query16_scalar(unit_q, query, centroid_data, y, D);
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"host-prep", cpu.stop()});
#endif
//--------------------------------------------------------
// B) H2D copies of query + buffers (GPU time)
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  GpuTimer gt;
  gt.start(stream_);
#endif
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    d_unit_q_gpu, unit_q, sizeof(float) * num_dimensions, cudaMemcpyHostToDevice, stream_));
  //    cudaMallocAsync(&quant_query_gpu, sizeof(int16_t) * D, stream);
  //    cudaMemcpyAsync(quant_query_gpu, quant_query,
  //                    sizeof(int16_t) * D, cudaMemcpyHostToDevice, stream);

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"copy-query-H2D", gt.stop(stream_)});
#endif
  //--------------------------------------------------------
  // C) IP kernel
  //--------------------------------------------------------

#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  int threadsPerBlock = 256;  // must be a multiple of 32
  int warpsPerBlock   = threadsPerBlock / 32;
  int grid            = (num_vector_cluster + warpsPerBlock - 1) / warpsPerBlock;
  compute_ip_kernel_fp32<<<grid, threadsPerBlock, 0, stream_>>>(
    d_unit_q_gpu,
    rabitq_codes_and_factor,
    num_dimensions,
    cur_ivf.quantizer().num_short_factors(),
    num_vector_cluster,
    d_ip_results,
    d_est_dis,
    delta,
    sumq,
    y,
    one_over_sqrtD);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-ip", gt.stop(stream_)});
#endif

#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  // copy data offload to CPU;
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_ip_results,
                                d_ip_results,
                                sizeof(float) * num_vector_cluster,
                                cudaMemcpyDeviceToHost,
                                stream_));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    h_est_dis, d_est_dis, sizeof(float) * num_vector_cluster, cudaMemcpyDeviceToHost, stream_));
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  // (D) apply filters and restore K
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"copy back data from GPU to CPU:", gt.stop(stream_)});
#endif

#ifdef COUNT_CLUSTER_TIME
  cpu.start();
  int count = 0;
#endif
  for (size_t i = 0; i < num_vector_cluster; ++i) {
    if (h_est_dis[i] < h_filter_distk) {
      // get long codes
      auto long_code_bytes = cur_ivf.quantizer().long_code_length();
      const uint8_t* codes =
        cur_cluster.long_code_host(cur_ivf, 0, cur_ivf.quantizer().long_code_length()) +
        i * long_code_bytes;
      float ip2 = 0.f;
      for (size_t j = 0; j < num_dimensions; ++j) {
        uint32_t c = extract_code_cpu(codes, j, cur_ivf.get_ex_bits());
        ip2 += float(c) * unit_q[j];
      }
#ifdef COUNT_CLUSTER_TIME
      count++;
#endif
      // refine the final distances
      size_t short_block_bytes_in_uint32 =
        cur_ivf.quantizer().short_code_length() + cur_ivf.quantizer().num_short_factors();
      uint32_t* block = cur_cluster.first_block_host(cur_ivf) + i * short_block_bytes_in_uint32;
      float onorm     = *reinterpret_cast<const float*>(
        block + (short_block_bytes_in_uint32 - cur_ivf.quantizer().num_short_factors()));
      float x_ip   = (cur_cluster.ex_factor_host(cur_ivf, 0))[i].xipnorm;
      float approx = h_ip_results[i];
      float refined =
        onorm * onorm + sqr_y -
        x_ip * (sqrt(sqr_y)) * (FAC_RESCALE * approx + ip2 - (FAC_RESCALE - 0.5f) * sumq);
      //            printf("refined: %f\n", refined);
      KNNs->insert({refined, cur_cluster.ids_host(cur_ivf)[i]});
    }
  }

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"CPU-side filter and get dis with ex bits", cpu.stop()});
#endif

  // Note: Merge results outside this function

#ifdef COUNT_CLUSTER_TIME
  //    stats.push_back({"select_final_topk_write_back",  gt.stop()});

  //--------------------------------------------------------
  // H)  final report (once per probe or outside the OMP loop)
  //--------------------------------------------------------
  double total_ms = 0.0;
  for (auto& s : stats)
    total_ms += s.ms;

  std::cout << "\n---- SearchCluster timing ----\n";
  printf("num of vectors after filtering: %d\n", count);
  std::cout << std::left << std::setw(22) << "Step" << std::right << std::setw(12) << "ms"
            << std::setw(10) << "%" << '\n';

  for (auto& s : stats) {
    std::cout << std::left << std::setw(22) << s.name << std::right << std::setw(12) << std::fixed
              << std::setprecision(3) << s.ms << std::setw(9) << std::fixed << std::setprecision(1)
              << (s.ms * 100.0 / total_ms) << '\n';
  }
  std::cout << std::left << std::setw(22) << "TOTAL" << std::right << std::setw(12) << std::fixed
            << std::setprecision(3) << total_ms << '\n';
#endif
  //    RAFT_CUDA_TRY(cudaGetLastError());
  //    RAFT_CUDA_TRY(cudaFree(quant_query_gpu));
  //    RAFT_CUDA_TRY(cudaFree(rotated_query_gpu));
  //    cudaFreeAsync(rotated_query_gpu, stream);
  //    cudaFreeAsync(quant_query_gpu, stream);
}

void SearcherGPU::SearchClusterWithFilterMemOpt(const IVFGPU& cur_ivf,
                                                const IVFGPU::GPUClusterMeta& cur_cluster,
                                                float sqr_y,
                                                DeviceResultPool& KNNs,
                                                float* centroid_data)
{
  // related information
#ifdef COUNT_CLUSTER_TIME
  struct StepStat {
    std::string name;
    double ms = 0;
  };
  std::vector<StepStat> stats;  // push in call order
#endif
  size_t num_vector_cluster         = cur_cluster.num;
  size_t num_dimensions             = cur_ivf.get_num_padded_dim();
  uint32_t* rabitq_codes_and_factor = cur_cluster.first_block(cur_ivf);
  PID* ids                          = cur_cluster.ids(cur_ivf);

  //--------------------------------------------------------
// A) host-side scalar preparation
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  CpuTimer cpu;
  cpu.start();
#endif
  float y     = std::sqrt(sqr_y);
  this->shift = 0;
  this->sumq  = normalize_query16_scalar(unit_q, query, centroid_data, y, D);
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"host-prep", cpu.stop()});
#endif
//--------------------------------------------------------
// B) H2D copies of query + buffers (GPU time)
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  GpuTimer gt;
  gt.start(stream_);
#endif
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    d_unit_q_gpu, unit_q, sizeof(float) * num_dimensions, cudaMemcpyHostToDevice, stream_));
  //    cudaMallocAsync(&quant_query_gpu, sizeof(int16_t) * D, stream);
  //    cudaMemcpyAsync(quant_query_gpu, quant_query,
  //                    sizeof(int16_t) * D, cudaMemcpyHostToDevice, stream);

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"copy-query-H2D", gt.stop(stream_)});
#endif
  //--------------------------------------------------------
  // C) IP kernel
  //--------------------------------------------------------

#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  int threadsPerBlock = 256;  // must be a multiple of 32
  int warpsPerBlock   = threadsPerBlock / 32;
  int grid            = (num_vector_cluster + warpsPerBlock - 1) / warpsPerBlock;
  compute_ip_kernel_fp32<<<grid, threadsPerBlock, 0, stream_>>>(
    d_unit_q_gpu,
    rabitq_codes_and_factor,
    num_dimensions,
    cur_ivf.quantizer().num_short_factors(),
    num_vector_cluster,
    d_ip_results,
    d_est_dis,
    delta,
    sumq,
    y,
    one_over_sqrtD);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-ip", gt.stop(stream_)});
#endif

  // (D) apply filters and restore K
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  //    printf("trying to filter and gather...");
  size_t M  = 10;  // multiples of 10 temporally
  size_t KM = M * KNNs.capacity;
  ;
  //    int written = select_and_keep_topKM_by_distance(
  //            est_dis, ids, ip_results,
  //            num_vector_cluster,              // num_vector_cluster
  //            KM,             // TOPK * M
  //            h_filter_distk,
  //            d_top_ip,     // same pointer names you used
  //            d_top_pids,
  //            d_top_idx,
  //            stream);     // or 0 for the default stream
  int written = fast_select_and_keep(d_est_dis,
                                     d_ip_results,
                                     ids,
                                     num_vector_cluster,
                                     KM,
                                     h_filter_distk,
                                     d_top_ip,
                                     d_top_pids,
                                     d_top_idx,
                                     stream_);
  //    int written = fused_filter_gather(est_dis, ip_results, ids, num_vector_cluster, KM,
  //    h_filter_distk, d_top_ip, d_top_pids, d_top_idx, stream); RAFT_CUDA_TRY(cudaGetLastError());
  //    printf("written: %d\n", written);
  if (written == 0) {
    KNNs.size = 0;
    return;
  }

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-filter-select_topk*m", gt.stop(stream_)});
#endif

  //    int written = (int)selectTopKM(est_dis, ip_results, ids,
  //            num_vector_cluster,
  //            KM,                      // TOPK * M
  //            h_filter_distk,
  //            d_top_ip, d_top_pids, d_top_idx,
  //            stream);

  // original correct version
//    select_topk_m_inplace(            // unchanged API, but it calls kernels ↦ must take stream
//    too
//            est_dis, ids, ip_results,
//            num_vector_cluster,
//            KNNs->capacity, M,
//            d_top_ip, d_top_pids, d_top_idx,
//            stream                           // <-- add parameter inside that function
//    );
//    written = KM;
//    printf("written: %d\n", written);
//--------------------------------------------------------
// E) long-code IP kernel
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif

  grid = (written + warpsPerBlock - 1) / warpsPerBlock;
  compute_long_ip_kernel<<<grid, threadsPerBlock, 0, stream_>>>(
    d_unit_q_gpu,
    d_top_idx,
    written,
    D,
    cur_ivf.get_ex_bits(),
    cur_cluster.long_code(cur_ivf, 0, cur_ivf.quantizer().long_code_length()),
    cur_ivf.quantizer().long_code_length(),
    d_ip2);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-long_ip", gt.stop(stream_)});
#endif
//--------------------------------------------------------
// F) refine kernel
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  int threads = 256;
  int blocks  = (written + threads - 1) / threads;
  // now d_top_dist is top ip instead
  refine_with_ip2_kernel<<<blocks, threads, 0, stream_>>>(
    d_top_ip,
    d_ip2,
    d_top_idx,
    sqr_y,
    FAC_RESCALE,
    sumq,
    written,
    cur_cluster.first_block(cur_ivf),
    cur_ivf.quantizer().short_code_length() + cur_ivf.quantizer().num_short_factors(),
    cur_cluster.ex_factor(cur_ivf, 0),
    cur_ivf.quantizer().num_short_factors());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-refine", gt.stop(stream_)});
#endif

//--------------------------------------------------------
// G) D2H copy + write_topk_to_pool  (host work)
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  //    cpu.start();
  //    printf("writing back results... \n");
  if (written > KNNs.capacity) {
    //        sort_num += 1;
    //        printf("writing topk to pool, written: %zu \n", written);
    auto cluster_topk_distance =
      write_topk_to_pool(KNNs, d_top_ip, d_top_pids, written, KNNs.capacity, stream_);
    KNNs.size = KNNs.capacity;
    //        printf("Before assignment:\n");
    //        printf("  h_filter_distk        = %.6f\n", h_filter_distk);
    //
    if (h_filter_distk > cluster_topk_distance) { h_filter_distk = cluster_topk_distance; }

    //        printf("After assignment:\n");
    //        printf("  h_filter_distk        = %.6f\n", h_filter_distk);
  } else {
    // copy directly without sorting
    //        direct_num += 1;
    //        printf("copying directly \n");
    raft::copy(KNNs.distances.data_handle(), d_top_ip, written, stream_);
    raft::copy(KNNs.ids.data_handle(), d_top_pids, written, stream_);

    KNNs.size = written;
  }

#ifdef DEBUG_MULTIPLE_SEARCH
  // copy back KNNs from a device and print data inside
  DeviceResultPool host_knns;
  host_knns.capacity  = KNNs[0].capacity;
  host_knns.distances = (float*)malloc(sizeof(float) * KNNs[0].capacity);
  host_knns.ids       = (uint32_t*)malloc(sizeof(uint32_t) * KNNs[0].capacity);
  RAFT_CUDA_TRY(cudaMemcpy(host_knns.distances,
                           KNNs[0].distances,
                           sizeof(float) * KNNs[0].capacity,
                           cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(
    host_knns.ids, KNNs[0].ids, sizeof(uint32_t) * KNNs[0].capacity, cudaMemcpyDeviceToHost));
  for (int i = 0; i < KNNs[0].size; i++) {
    printf("dist-ori: %f\n", host_knns.distances[i]);
    printf("id-ori: %d\n", host_knns.ids[i]);
  }
  free(host_knns.distances);
  free(host_knns.ids);
#endif
  //    printf("writing finished \n");

  // TODO: update h_filter
//   f write_ms = gt.stop();
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"select_final_topk_write_back", gt.stop()});

  //--------------------------------------------------------
  // H)  final report (once per probe or outside the OMP loop)
  //--------------------------------------------------------
  double total_ms = 0.0;
  for (auto& s : stats)
    total_ms += s.ms;

  std::cout << "\n---- SearchCluster timing ----\n";
  std::cout << std::left << std::setw(22) << "Step" << std::right << std::setw(12) << "ms"
            << std::setw(10) << "%" << '\n';

  for (auto& s : stats) {
    std::cout << std::left << std::setw(22) << s.name << std::right << std::setw(12) << std::fixed
              << std::setprecision(3) << s.ms << std::setw(9) << std::fixed << std::setprecision(1)
              << (s.ms * 100.0 / total_ms) << '\n';
  }
  std::cout << std::left << std::setw(22) << "TOTAL" << std::right << std::setw(12) << std::fixed
            << std::setprecision(3) << total_ms << '\n';
#endif
  //    RAFT_CUDA_TRY(cudaGetLastError());
  //    RAFT_CUDA_TRY(cudaFree(quant_query_gpu));
  //    RAFT_CUDA_TRY(cudaFree(rotated_query_gpu));
  //    cudaFreeAsync(rotated_query_gpu, stream);
  //    cudaFreeAsync(quant_query_gpu, stream);
}

// always sort for the first cluster, first cluster, thershold is an infinity.
int fast_select_and_keep_one_for_multi(const float* d_dist,
                                       const float* d_ip,
                                       const uint32_t* d_pid,
                                       int N,
                                       int KM,
                                       float threshold,
                                       float* d_top_ip,
                                       uint32_t* d_top_pid,
                                       int* d_top_idx,
                                       rmm::cuda_stream_view stream)
{
  // 1) allocate worst‑case survivor buffer  (in practise you can reuse one)
  Candidate3* d_buf;
  RAFT_CUDA_TRY(cudaMallocAsync(
    &d_buf, sizeof(Candidate3) * N, stream));  // oversize ok (N for storage, N for sorting)
  int* d_counter;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_counter, sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counter, 0, sizeof(int), stream));

  int BLK = 256, GRD = (N + BLK - 1) / BLK;

  // 2) launch pass‑1
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counter, 0, sizeof(int), stream));
  filterGatherKernel<<<GRD, BLK, 0, stream>>>(d_dist, d_ip, d_pid, N, threshold, d_buf, d_counter);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // 3) fetch survivor count

  //    if (sel_cnt == 0) { cudaFreeAsync(d_buf, stream); cudaFreeAsync(d_counter, stream); return
  //    0; }

  //    if (sel_cnt > KM)
  //    printf("N: %d, KM: %d\n", N, KM);
  {
    //------------------------------------------------------------------
    // 4‑A)  build a contiguous array of distance keys
    //------------------------------------------------------------------
    float* d_keys;  // [sel_cnt] float
    RAFT_CUDA_TRY(cudaMallocAsync(&d_keys, N * sizeof(float), stream));

    // one kernel copy is faster than Thrust here
    int BLK = 256, GRD = (N + BLK - 1) / BLK;
    copyDistKernel<<<GRD, BLK, 0, stream>>>(d_buf, d_keys, N);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    //------------------------------------------------------------------
    // 4‑B)  radix‑sort ( key = d_keys , value = d_buf )
    //------------------------------------------------------------------
    size_t tmpBytes = 0;
    cub::DeviceRadixSort::SortPairs(
      /*temp=*/nullptr,
      tmpBytes,
      d_keys,
      d_keys,  // in‑place: keys_in == keys_out
      d_buf,
      d_buf,  // in‑place: values_in == values_out
      N,
      0,
      32,
      stream);  // 0‑31 bits are enough for floats

    void* d_tmp;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_tmp, tmpBytes, stream));
    cub::DeviceRadixSort::SortPairs(
      d_tmp, tmpBytes, d_keys, d_keys, d_buf, d_buf, N, 0, 32, stream);
    RAFT_CUDA_TRY(cudaFreeAsync(d_tmp, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_keys, stream));
    //  d_buf is now sorted by ascending dist
  }

  // 5) copy first out_cnt elements to user buffers
  int out_cnt = min(N, KM);

  GRD = (out_cnt + BLK - 1) / BLK;
  splitKernel<<<GRD, BLK, 0, stream>>>(d_buf, d_top_ip, d_top_pid, d_top_idx, out_cnt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaFreeAsync(d_buf, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_counter, stream));
  return out_cnt;
}

void SearcherGPU::SearchClusterWithFilterMemOptOneforMulti(
  const IVFGPU& cur_ivf,
  const IVFGPU::GPUClusterMeta& cur_cluster,
  float sqr_y,
  DeviceResultPool& KNNs,
  float* centroid_data)
{
  // related information
#ifdef COUNT_CLUSTER_TIME
  struct StepStat {
    std::string name;
    double ms = 0;
  };
  std::vector<StepStat> stats;  // push in call order
#endif
  size_t num_vector_cluster         = cur_cluster.num;
  size_t num_dimensions             = cur_ivf.get_num_padded_dim();
  uint32_t* rabitq_codes_and_factor = cur_cluster.first_block(cur_ivf);
  PID* ids                          = cur_cluster.ids(cur_ivf);

  //--------------------------------------------------------
// A) host-side scalar preparation
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  CpuTimer cpu;
  cpu.start();
#endif
  float y     = std::sqrt(sqr_y);
  this->shift = 0;
  this->sumq  = normalize_query16_scalar(unit_q, query, centroid_data, y, D);
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"host-prep", cpu.stop()});
#endif
//--------------------------------------------------------
// B) H2D copies of query + buffers (GPU time)
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  GpuTimer gt;
  gt.start(stream_);
#endif
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    d_unit_q_gpu, unit_q, sizeof(float) * num_dimensions, cudaMemcpyHostToDevice, stream_));
  //    cudaMallocAsync(&quant_query_gpu, sizeof(int16_t) * D, stream);
  //    cudaMemcpyAsync(quant_query_gpu, quant_query,
  //                    sizeof(int16_t) * D, cudaMemcpyHostToDevice, stream);

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"copy-query-H2D", gt.stop(stream_)});
#endif
  //--------------------------------------------------------
  // C) IP kernel
  //--------------------------------------------------------

#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  int threadsPerBlock = 256;  // must be a multiple of 32
  int warpsPerBlock   = threadsPerBlock / 32;
  int grid            = (num_vector_cluster + warpsPerBlock - 1) / warpsPerBlock;
  compute_ip_kernel_fp32<<<grid, threadsPerBlock, 0, stream_>>>(
    d_unit_q_gpu,
    rabitq_codes_and_factor,
    num_dimensions,
    cur_ivf.quantizer().num_short_factors(),
    num_vector_cluster,
    d_ip_results,
    d_est_dis,
    delta,
    sumq,
    y,
    one_over_sqrtD);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-ip", gt.stop(stream_)});
#endif

  // (D) apply filters and restore K
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  //    printf("trying to filter and gather...");
  size_t M  = 10;  // multiples of 10 temporally
  size_t KM = M * KNNs.capacity;
  ;
  //    int written = select_and_keep_topKM_by_distance(
  //            est_dis, ids, ip_results,
  //            num_vector_cluster,              // num_vector_cluster
  //            KM,             // TOPK * M
  //            h_filter_distk,
  //            d_top_ip,     // same pointer names you used
  //            d_top_pids,
  //            d_top_idx,
  //            stream);     // or 0 for the default stream
  int written = fast_select_and_keep_one_for_multi(d_est_dis,
                                                   d_ip_results,
                                                   ids,
                                                   num_vector_cluster,
                                                   KM,
                                                   h_filter_distk,
                                                   d_top_ip,
                                                   d_top_pids,
                                                   d_top_idx,
                                                   stream_);
  //    int written = fused_filter_gather(est_dis, ip_results, ids, num_vector_cluster, KM,
  //    h_filter_distk, d_top_ip, d_top_pids, d_top_idx, stream); RAFT_CUDA_TRY(cudaGetLastError());
  //    printf("written: %d\n", written);
  if (written == 0) {
    KNNs.size = 0;
    return;
  }

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-filter-select_topk*m", gt.stop(stream_)});
#endif

  //    int written = (int)selectTopKM(est_dis, ip_results, ids,
  //            num_vector_cluster,
  //            KM,                      // TOPK * M
  //            h_filter_distk,
  //            d_top_ip, d_top_pids, d_top_idx,
  //            stream);

  // original correct version
//    select_topk_m_inplace(            // unchanged API, but it calls kernels ↦ must take stream
//    too
//            est_dis, ids, ip_results,
//            num_vector_cluster,
//            KNNs->capacity, M,
//            d_top_ip, d_top_pids, d_top_idx,
//            stream                           // <-- add parameter inside that function
//    );
//    written = KM;
//    printf("written: %d\n", written);
//--------------------------------------------------------
// E) long-code IP kernel
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif

  grid = (written + warpsPerBlock - 1) / warpsPerBlock;
  compute_long_ip_kernel<<<grid, threadsPerBlock, 0, stream_>>>(
    d_unit_q_gpu,
    d_top_idx,
    written,
    D,
    cur_ivf.get_ex_bits(),
    cur_cluster.long_code(cur_ivf, 0, cur_ivf.quantizer().long_code_length()),
    cur_ivf.quantizer().long_code_length(),
    d_ip2);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-long_ip", gt.stop(stream_)});
#endif
//--------------------------------------------------------
// F) refine kernel
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  int threads = 256;
  int blocks  = (written + threads - 1) / threads;
  // now d_top_dist is top ip instead
  refine_with_ip2_kernel<<<blocks, threads, 0, stream_>>>(
    d_top_ip,
    d_ip2,
    d_top_idx,
    sqr_y,
    FAC_RESCALE,
    sumq,
    written,
    cur_cluster.first_block(cur_ivf),
    cur_ivf.quantizer().short_code_length() + cur_ivf.quantizer().num_short_factors(),
    cur_cluster.ex_factor(cur_ivf, 0),
    cur_ivf.quantizer().num_short_factors());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-refine", gt.stop(stream_)});
#endif

//--------------------------------------------------------
// G) D2H copy + write_topk_to_pool  (host work)
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  //    cpu.start();
  //    printf("writing back results... \n");
  if (written > KNNs.capacity) {
    // edit for multicluster search: directly set d_filter_distk
    write_topk_to_pool_one_for_multi(
      KNNs, d_top_ip, d_top_pids, written, KNNs.capacity, d_filter_distk, stream_);
    KNNs.size = KNNs.capacity;
  } else {
    raft::copy(KNNs.distances.data_handle(), d_top_ip, written, stream_);
    raft::copy(KNNs.ids.data_handle(), d_top_pids, written, stream_);

    // pick a random dist for to filter
    // TODO: more accurate distk
    raft::copy(d_filter_distk, KNNs.distances.data_handle() + (written - 1), 1, stream_);

    KNNs.size = written;
  }

#ifdef DEBUG_MULTIPLE_SEARCH
  // copy back KNNs from a device and print data inside
  DeviceResultPool host_knns;
  host_knns.capacity  = KNNs[0].capacity;
  host_knns.distances = (float*)malloc(sizeof(float) * KNNs[0].capacity);
  host_knns.ids       = (uint32_t*)malloc(sizeof(uint32_t) * KNNs[0].capacity);
  RAFT_CUDA_TRY(cudaMemcpy(host_knns.distances,
                           KNNs[0].distances,
                           sizeof(float) * KNNs[0].capacity,
                           cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(
    host_knns.ids, KNNs[0].ids, sizeof(uint32_t) * KNNs[0].capacity, cudaMemcpyDeviceToHost));
  for (int i = 0; i < KNNs[0].size; i++) {
    printf("dist-ori: %f\n", host_knns.distances[i]);
    printf("id-ori: %d\n", host_knns.ids[i]);
  }
  free(host_knns.distances);
  free(host_knns.ids);
#endif
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"select_final_topk_write_back", gt.stop()});

  //--------------------------------------------------------
  // H)  final report (once per probe or outside the OMP loop)
  //--------------------------------------------------------
  double total_ms = 0.0;
  for (auto& s : stats)
    total_ms += s.ms;

  std::cout << "\n---- SearchCluster timing ----\n";
  std::cout << std::left << std::setw(22) << "Step" << std::right << std::setw(12) << "ms"
            << std::setw(10) << "%" << '\n';

  for (auto& s : stats) {
    std::cout << std::left << std::setw(22) << s.name << std::right << std::setw(12) << std::fixed
              << std::setprecision(3) << s.ms << std::setw(9) << std::fixed << std::setprecision(1)
              << (s.ms * 100.0 / total_ms) << '\n';
  }
  std::cout << std::left << std::setw(22) << "TOTAL" << std::right << std::setw(12) << std::fixed
            << std::setprecision(3) << total_ms << '\n';
#endif
}

void SearcherGPU::SearchClusterWithFilterMemOptV2(const IVFGPU& cur_ivf,
                                                  const IVFGPU::GPUClusterMeta& cur_cluster,
                                                  float sqr_y,
                                                  DeviceResultPool& KNNs,
                                                  float* centroid_data)
{
  // related information
#ifdef COUNT_CLUSTER_TIME
  struct StepStat {
    std::string name;
    double ms = 0;
  };
  std::vector<StepStat> stats;  // push in call order
#endif
  size_t num_vector_cluster         = cur_cluster.num;
  size_t num_dimensions             = cur_ivf.get_num_padded_dim();
  uint32_t* rabitq_codes_and_factor = cur_cluster.first_block(cur_ivf);
  PID* ids                          = cur_cluster.ids(cur_ivf);

  //--------------------------------------------------------
// A) host-side scalar preparation
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  CpuTimer cpu;
  cpu.start();
#endif
  float y     = std::sqrt(sqr_y);
  this->shift = 0;
  this->sumq  = normalize_query16_scalar(unit_q, query, centroid_data, y, D);
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"host-prep", cpu.stop()});
#endif
//--------------------------------------------------------
// B) H2D copies of query + buffers (GPU time)
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  GpuTimer gt;
  gt.start(stream_);
#endif
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    d_unit_q_gpu, unit_q, sizeof(float) * num_dimensions, cudaMemcpyHostToDevice, stream_));

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"copy-query-H2D", gt.stop(stream_)});
#endif
  //--------------------------------------------------------
  // C) IP kernel
  //--------------------------------------------------------

#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  int threadsPerBlock = 256;  // must be a multiple of 32
  int warpsPerBlock   = threadsPerBlock / 32;
  int grid            = (num_vector_cluster + warpsPerBlock - 1) / warpsPerBlock;
  compute_ip_kernel_fp32_opt<<<grid, threadsPerBlock, 0, stream_>>>(
    d_unit_q_gpu,
    rabitq_codes_and_factor,
    num_dimensions,
    cur_ivf.quantizer().num_short_factors(),
    num_vector_cluster,
    d_ip_results,
    d_est_dis,
    delta,
    sumq,
    y,
    one_over_sqrtD,
    h_filter_distk);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-ip", gt.stop(stream_)});
#endif

  // (D) apply filters and restore K
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  //    printf("trying to filter and gather...");
  size_t M  = 10;  // multiples of 10 temporally
  size_t KM = M * KNNs.capacity;
  ;

  int written = fast_select_and_keepv2(d_est_dis,
                                       d_ip_results,
                                       ids,
                                       num_vector_cluster,
                                       KM,
                                       h_filter_distk,
                                       d_top_ip,
                                       d_top_pids,
                                       d_top_idx,
                                       stream_);

  if (written == 0) {
    KNNs.size = 0;
    return;
  }

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-filter-select_topk*m", gt.stop(stream_)});
#endif

  //--------------------------------------------------------
// E) long-code IP kernel
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif

  grid = (written + warpsPerBlock - 1) / warpsPerBlock;
  compute_long_ip_kernel<<<grid, threadsPerBlock, 0, stream_>>>(
    d_unit_q_gpu,
    d_top_idx,
    written,
    D,
    cur_ivf.get_ex_bits(),
    cur_cluster.long_code(cur_ivf, 0, cur_ivf.quantizer().long_code_length()),
    cur_ivf.quantizer().long_code_length(),
    d_ip2);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-long_ip", gt.stop(stream_)});
#endif
//--------------------------------------------------------
// F) refine kernel
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  int threads = 256;
  int blocks  = (written + threads - 1) / threads;
  // now d_top_dist is top ip instead
  refine_with_ip2_kernel<<<blocks, threads, 0, stream_>>>(
    d_top_ip,
    d_ip2,
    d_top_idx,
    sqr_y,
    FAC_RESCALE,
    sumq,
    written,
    cur_cluster.first_block(cur_ivf),
    cur_ivf.quantizer().short_code_length() + cur_ivf.quantizer().num_short_factors(),
    cur_cluster.ex_factor(cur_ivf, 0),
    cur_ivf.quantizer().num_short_factors());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"kernel-refine", gt.stop(stream_)});
#endif

//--------------------------------------------------------
// G) D2H copy + write_topk_to_pool  (host work)
//--------------------------------------------------------
#ifdef COUNT_CLUSTER_TIME
  gt.start(stream_);
#endif
  //    cpu.start();
  //    printf("writing back results... \n");
  if (written > KNNs.capacity) {
    auto cluster_topk_distance =
      write_topk_to_pool(KNNs, d_top_ip, d_top_pids, written, KNNs.capacity, stream_);
    KNNs.size = KNNs.capacity;
    if (h_filter_distk > cluster_topk_distance) { h_filter_distk = cluster_topk_distance; }
  } else {
    raft::copy(KNNs.distances.data_handle(), d_top_ip, written, stream_);
    raft::copy(KNNs.ids.data_handle(), d_top_pids, written, stream_);

    KNNs.size = written;
  }
  //    printf("writing finished \n");

  // TODO: update h_filter
//   f write_ms = gt.stop();
#ifdef COUNT_CLUSTER_TIME
  stats.push_back({"select_final_topk_write_back", gt.stop()});

  //--------------------------------------------------------
  // H)  final report (once per probe or outside the OMP loop)
  //--------------------------------------------------------
  double total_ms = 0.0;
  for (auto& s : stats)
    total_ms += s.ms;

  std::cout << "\n---- SearchCluster timing ----\n";
  std::cout << std::left << std::setw(22) << "Step" << std::right << std::setw(12) << "ms"
            << std::setw(10) << "%" << '\n';

  for (auto& s : stats) {
    std::cout << std::left << std::setw(22) << s.name << std::right << std::setw(12) << std::fixed
              << std::setprecision(3) << s.ms << std::setw(9) << std::fixed << std::setprecision(1)
              << (s.ms * 100.0 / total_ms) << '\n';
  }
  std::cout << std::left << std::setw(22) << "TOTAL" << std::right << std::setw(12) << std::fixed
            << std::setprecision(3) << total_ms << '\n';
#endif
}

// following for searching multiple cluster

//--------------------------------------------------------------
// warp32 kernel – single warp per centroid
//--------------------------------------------------------------
__global__ void normalize_warp32_kernel(const Candidate* __restrict__ dc,
                                        const float* __restrict__ dcent,
                                        float* __restrict__ d_vecs,
                                        SumNorm* __restrict__ d_sum_norm,
                                        float* __restrict__ c_query,
                                        int D)
{
  constexpr float eps  = 1e-5f;
  const int p          = blockIdx.x;
  const Candidate cand = dc[p];
  const float* c       = dcent + cand.id * D;
  float* u             = d_vecs + p * D;

  float norm = sqrtf(cand.distance);
  bool good  = norm > eps;
  float inv  = good ? 1.f / norm : rsqrtf((float)D);
  float fill = good ? 0.f : inv;

  float local = 0.f;
  for (int d = threadIdx.x * 4; d < D; d += 32 * 4) {
    float4 cv = *reinterpret_cast<const float4*>(c + d);
    float q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[i] = c_query[d + i];
      //            if (d+i > D) {
      //                printf("d+i > D\n");
      //            }
    }
    float* cp = reinterpret_cast<float*>(&cv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v  = good ? (q[i] - cp[i]) * inv : fill;
      u[d + i] = v;
      local += v;
    }
  }
  for (int off = 16; off; off >>= 1)
    local += __shfl_down_sync(0xffffffff, local, off);
  if (threadIdx.x == 0) { d_sum_norm[p] = {local, norm}; }
}

// returns probe s.t.  d_starts[probe] ≤ idx < d_starts[probe+1]
__device__ __forceinline__ int locate_probe(int idx,
                                            const int* __restrict__ d_starts,  // length nprobe+1
                                            int nprobe)
{
  int lo = 0, hi = nprobe;  // invariant: idx < d_starts[hi]

  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    int s   = __ldg(d_starts + mid);  // = d_starts[mid]

    // keep the half that still contains idx
    (idx < s) ? hi = mid   // idx is in the left half
              : lo = mid;  // idx is in [mid, hi)
  }
  return lo;  // d_starts[lo] ≤ idx < d_starts[lo+1]
}

/**
 * Nwarp = total #vectors across all selected clusters
 * grid   = ceil(Nwarp * 32 / blockDim.x)
 * block  = e.g. 128, 256, 512  (must be a multiple of 32)
 */
__global__ void compute_ip_kernel_fp32_multi(
  const float* __restrict__ d_unit_queries,  // nprobe × D  (row-major)
  const SumNorm* __restrict__ d_sum_norm,
  IVFGPU::GPUClusterMeta* __restrict__ d_meta,  // nprobe
  const size_t D,
  uint32_t* d_short_data,
  const size_t num_short_factors,
  const size_t short_code_length,
  int* d_starts,  // prefix_sum
  int nprobe,
  const float one_over_sqrtD,
  const float filter_dis,
  float* __restrict__ d_ip_results,  // ← per-vector
  float* __restrict__ d_est_dis)     // ← per-vector
{
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_id = tid >> 5;  // one warp → one data vector
  const int lane    = threadIdx.x & 31;

  // ------------------------------------------------------------
  // 1) Map warp-id → (cluster-id, index-in-cluster)
  // ------------------------------------------------------------
  // Our cluster list is small ( ≤ 10k ), so a linear search is OK.
  // All 32 lanes in the warp do the same search; no divergence.

  int p = locate_probe(warp_id, d_starts, nprobe);
  //    first_idx = d_meta[p].start_index;
  //    last_idx  = d_meta[p].start_index + d_meta[p].num;

  const IVFGPU::GPUClusterMeta* meta = &d_meta[p];
  const int idx_in_cluster           = int(warp_id - d_starts[p]);

  if (idx_in_cluster >= meta->num) return;  // safety for over-sized grid

  // ------------------------------------------------------------
  // 2) Gather per-cluster pointers / scalars
  // ------------------------------------------------------------
  const float* q    = d_unit_queries + p * D;  // row   D
  const float sumq  = d_sum_norm[p].sum;
  const float qnorm = d_sum_norm[p].norm;

  //    const int num_words = int((D + 31) >> 5);
  const uint32_t* block = meta->first_block_gpu(d_short_data, short_code_length, num_short_factors);
  block += idx_in_cluster * (short_code_length + num_short_factors);

  // ------------------------------------------------------------
  // 3) Per-lane partial inner product (same as before)
  // ------------------------------------------------------------
  float partial = 0.f;

  for (int d = lane; d < int(D); d += 32) {
    const uint32_t word = block[d >> 5];
    const int bit       = (word >> (31 - (d & 31))) & 1;
    if (bit) partial += __ldg(&q[d]);  // query lives in global RO cache
  }

  // Warp reduction
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xFFFFFFFF, partial, off);

  // ------------------------------------------------------------
  // 4) Lane-0 finalises and stores
  // ------------------------------------------------------------
  if (lane == 0) {
    const float onorm = reinterpret_cast<const float*>(block)[short_code_length];

    const float ip_xb_prime = partial;  // ⟨q, b⟩

    const float result =
      (ip_xb_prime - 0.5f * sumq + 0.58f) * (-5.0f * qnorm * one_over_sqrtD) * onorm +
      qnorm * qnorm + onorm * onorm;

#ifdef DEBUG_MULTIPLE_SEARCH
    if (warp_id > 270 && warp_id < 275)
      printf("warp id: %d, cluster id: %d, index in cluster: %d, result: %f\n",
             warp_id,
             p,
             idx_in_cluster,
             result);
#endif

    //        if (result < filter_dis) {
    const int gidx     = warp_id;  // absolute vector id
    d_ip_results[gidx] = ip_xb_prime;
    d_est_dis[gidx]    = result;
    //        }
  }
}

/*****************************************************************
 *  d_meta     – GPUClusterMeta*              (device)  nprobe
 *  d_starts   – int*                         (device)  nprobe+1
 *  tmp_storage, bytes  – for CUB scan (can be reused elsewhere)
 *****************************************************************/
__global__ void extract_counts(const IVFGPU::GPUClusterMeta* meta, int* counts, int nprobe)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nprobe) counts[i] = static_cast<int>(meta[i].num);
}

void build_prefix_device(const IVFGPU::GPUClusterMeta* d_meta,
                         int nprobe,
                         int* d_starts,
                         cudaStream_t stream = 0)
{
  // 1) materialise the per-cluster sizes
  int* d_counts = d_starts;  // reuse first nprobe slots
  int BLK = 128, GRD = (nprobe + BLK - 1) / BLK;
  extract_counts<<<GRD, BLK, 0, stream>>>(d_meta, d_counts, nprobe);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // 2) exclusive scan → prefix-sum
  //    (requires nprobe+1 output slots, so use d_starts as keys_out)
  void* tmp_storage = nullptr;
  size_t tmp_bytes  = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, (int*)nullptr, (int*)nullptr, 1, stream);
  RAFT_CUDA_TRY(cudaMallocAsync(&tmp_storage, tmp_bytes, stream));

  cub::DeviceScan::ExclusiveSum(tmp_storage, tmp_bytes, d_counts, d_starts, nprobe + 1, stream);
}

// -----------------------------------------------------------------------------
//  Pass-1 :  filter + gather   (multi-cluster)
//. pid should be global start of the pid
// -----------------------------------------------------------------------------
template <int BLK = 256>
__global__ void filterGatherKernel_multi(const float* __restrict__ dist,
                                         const float* __restrict__ ip,
                                         const uint32_t* __restrict__ pid,
                                         const int* __restrict__ d_starts,  // prefix sum
                                         int N,
                                         int nprobe,
                                         float* threshold,
                                         IVFGPU::GPUClusterMeta* d_meta,
                                         Candidate4* __restrict__ out,
                                         int* d_counter)
{
  using BlockScan = cub::BlockScan<int, BLK>;
  __shared__ typename BlockScan::TempStorage scanTmp;
  __shared__ int blockBase;
  __shared__ float shared_threshold;  // Add this

  // Load threshold once per block
  if (threadIdx.x == 0) { shared_threshold = *threshold; }
  __syncthreads();

  const int tid = blockIdx.x * BLK + threadIdx.x;

  // ------------------------------------------------------------
  // 1) predicate + local registers
  // ------------------------------------------------------------
  int flag = 0;
  float vdist, vip;
  uint32_t vpid;
  int vidx, vprobe;

  if (tid < N) {
    vdist = dist[tid];
    flag  = (vdist != 0.f && vdist < shared_threshold);
    //        flag  = (vdist != 0.f);

    if (flag) {
      vip = ip[tid];
      //            vpid  = pid[tid];
      vidx = tid;  // global vector id

      // map global id → (probe , idx-in-cluster)
      vprobe = locate_probe(vidx, d_starts, nprobe);
      vidx -= __ldg(d_starts + vprobe);  // local idx
      // get actually pid from by probe
      vpid = __ldg(pid + d_meta[vprobe].start_index + vidx);
#ifdef DEBUG_MULTIPLE_SEARCH
//            for (int i = 0; i < nprobe; ++i) {
//                printf("d_starts[i] = %d\n", d_starts[i]);
//            }
//            printf("vdist = %f, vip = %f, vpid = %d, vidx = %d, vprobe = %d, tid = %d\n",
//                   vdist, vip, vpid, vidx, vprobe, tid);
#endif
    }
  }

  // ------------------------------------------------------------
  // 2) cooperative compaction into `out[]`
  // ------------------------------------------------------------
  int offset, blockTotal;
  BlockScan(scanTmp).ExclusiveSum(flag, offset, blockTotal);

  if (threadIdx.x == 0) blockBase = atomicAdd(d_counter, blockTotal);
  __syncthreads();

  if (flag) {
    int pos = blockBase + offset;
#ifdef DEBUG_MULTIPLE_SEARCH
//        printf("pos = %d, blockBase = %d, offset = %d, blockTotal = %d\n", pos, blockBase, offset,
//        blockTotal);
#endif
    out[pos] = {vdist, vip, vpid, vidx, vprobe};
  }
}

// -----------------------------------------------------------------------------
//  Split kernel  → scattered user buffers (same as before + probe)
// -----------------------------------------------------------------------------
//__global__ void splitKernel4(const Candidate4*  in,
//                             float*             out_ip,
//                             uint32_t*          out_pid,
//                             int*               out_idx,
//                             int*               out_probe,
//                             int n)
//{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i < n) {
//        out_ip   [i] = in[i].ip;
//        out_pid  [i] = in[i].pid;
//        out_idx  [i] = in[i].idx;
//        out_probe[i] = in[i].probe;
//    }
//}

// -----------------------------------------------------------------------------
//  Host helper  –  identical call signature plus      d_starts , nprobe
// -----------------------------------------------------------------------------
int fast_select_and_keep_multi(
  const float* d_dist,
  const float* d_ip,
  const uint32_t* d_pid,
  const int* d_starts,  // prefix sum (length nprobe+1)
  int N,
  int nprobe,
  int KM,
  float* d_filter_distk,
  IVFGPU::GPUClusterMeta* d_meta,
  // user buffers
  //        float*    d_top_ip,
  //        uint32_t* d_top_pid,
  //        int*      d_top_idx,
  //        int*      d_top_probe,
  Candidate4* d_top_candidates,  // Note: this is used as sorting buffer as well
  rmm::cuda_stream_view stream)
{
  //------------------------------------------------------------------
  // 0)  scratch
  //------------------------------------------------------------------
  Candidate4* d_buf = d_top_candidates;
  int* d_counter;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_counter, sizeof(int), stream));

  const int BLK = 256;
  const int GRD = (N + BLK - 1) / BLK;
#ifdef DEBUG_MULTIPLE_SEARCH
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaGetLastError());
#endif
  //------------------------------------------------------------------
  // 1) pass-1  (filter + gather)
  //------------------------------------------------------------------
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counter, 0, sizeof(int), stream));
  filterGatherKernel_multi<<<GRD, BLK, 0, stream>>>(
    d_dist, d_ip, d_pid, d_starts, N, nprobe, d_filter_distk, d_meta, d_buf, d_counter);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
#ifdef DEBUG_MULTIPLE_SEARCH
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
//    RAFT_CUDA_TRY(cudaGetLastError());
#endif
  //------------------------------------------------------------------
  // 2) fetch survivor count
  //------------------------------------------------------------------
  //    int* h_sel_cnt = nullptr;
  //    cudaMallocHost(&h_sel_cnt, sizeof(int));
  int h_sel_cnt;
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(&h_sel_cnt, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream));

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  int sel_cnt = h_sel_cnt;

  if (sel_cnt == 0) {
    RAFT_CUDA_TRY(cudaFreeAsync(d_counter, stream));
    return 0;
  }

  //------------------------------------------------------------------
  // 3) if too many, full radix-sort on  (key = dist)
  //------------------------------------------------------------------
  //    if (sel_cnt > KM)
  if (0) {
    float* d_keys;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_keys, sel_cnt * sizeof(float), stream));

    // build contiguous key array
    int GRD2 = (sel_cnt + BLK - 1) / BLK;
    copyDistKernel4<<<GRD2, BLK, 0, stream>>>(d_buf, d_keys, sel_cnt);  // reuse old kernel
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // radix-sort keys + values in-place
    size_t tmpBytes = 0;
    cub::DeviceRadixSort::SortPairs(
      nullptr, tmpBytes, d_keys, d_keys, d_buf, d_buf, sel_cnt, 0, 32, stream);

    void* d_tmp;
    RAFT_CUDA_TRY(cudaMallocAsync(&d_tmp, tmpBytes, stream));
    cub::DeviceRadixSort::SortPairs(
      d_tmp, tmpBytes, d_keys, d_keys, d_buf, d_buf, sel_cnt, 0, 32, stream);
    RAFT_CUDA_TRY(cudaFreeAsync(d_tmp, stream));
    RAFT_CUDA_TRY(cudaFreeAsync(d_keys, stream));
  }

  //------------------------------------------------------------------
  // 4) output first  KM  winners
  //------------------------------------------------------------------
  //    int out_cnt = std::min(sel_cnt, KM);
  int out_cnt = sel_cnt;
  // just skip because the result is already in the buffer
  //    int GRD3    = (out_cnt + BLK - 1) / BLK;
  //
  //    splitKernel4<<<GRD3, BLK, 0, stream>>>(
  //            d_buf, d_top_ip, d_top_pid, d_top_idx, d_top_probe, out_cnt);

  //------------------------------------------------------------------
  // 5)  clean up
  //------------------------------------------------------------------
  //    RAFT_CUDA_TRY(cudaFree(d_buf));
  RAFT_CUDA_TRY(cudaFreeAsync(d_counter, stream));
  //    cudaFreeHost(h_sel_cnt);
  return out_cnt;
}

// ---------------------------------------------
// kernel
// ---------------------------------------------
__global__ void refine_multi_kernel(
  /* winners --------------------------------------------------- */
  //        float*           d_top_ip,         // [KM]   in/out  (approx → refined)
  //        const int*       d_top_idx,          // [KM]   global vector ids (−1 = sentinel)
  Candidate4* d_top_candidates,
  int KM,
  /* per-cluster data ----------------------------------------- */
  const float* d_unit_q,  // [nprobe × D]   row-major
                          //        const float*     d_sumq,             // [nprobe]
                          //        const float*     d_sqr_y,            // [nprobe]  (= qnorm²)
  SumNorm* d_sum_norm,
  const int* d_starts,  // [nprobe+1] prefix sums
  int nprobe,
  int D,
  IVFGPU::GPUClusterMeta* d_meta,
  /* global tables -------------------------------------------- */
  const uint8_t* d_long_code,
  size_t long_code_bytes,
  const uint32_t* d_short_data,
  size_t short_block_uint32,
  const ExFactor* d_ex_factor,
  int num_short_factors,
  /* constants ------------------------------------------------- */
  int FAC_RESCALE,
  int BLOCK,
  int EX_BITS)
{
  constexpr int WARP = 32;
  int tid            = blockIdx.x * BLOCK + threadIdx.x;
  int warpId         = tid >> 5;
  int lane           = threadIdx.x & 31;
  if (warpId >= KM) return;

  /* ---------- fetch winner id & guard against sentinel --------- */
  Candidate4* candidate = d_top_candidates + warpId;
  //    int vid = candidate->idx;
  //    if (vid == -1) return;

  /* ---------- map vid → (probe, local idx) --------------------- */
  int probe          = candidate->nprobe;
  float approx       = candidate->ip;
  int cluster_offset = d_meta[probe].start_index;
  //    int idx_in_cluster = vid - cluster_offset;
  int idx_in_cluster = candidate->idx;

  /* ---------- per-cluster scalars & query ptr ------------------ */
  const float* q = d_unit_q + (size_t)probe * D;
  float sumq     = d_sum_norm[probe].sum;
  float qnorm    = d_sum_norm[probe].norm;
  //    float sumq  = __ldg(d_sumq  + probe);
  //    float sqr_y = __ldg(d_sqr_y + probe);
  float sqr_y = qnorm * qnorm;

  /* ---------- long-code pointer -------------------------------- */
  const uint8_t* codes = d_long_code + (size_t)(idx_in_cluster + cluster_offset) * long_code_bytes;

  /* ---------- part-1: compute ip2  ----------------------------- */
  float partial = 0.f;
  for (int d = lane; d < D; d += WARP) {
    uint32_t c = extract_code(codes, d, EX_BITS);
    partial += float(c) * __ldg(q + d);
  }
#ifdef DEBUG_MULTIPLE_SEARCH
  if (idx_in_cluster == 925) {
    printf("first four codes of uint8_t: %d, %d, %d, %d\n", codes[0], codes[1], codes[2], codes[3]);
  }
#endif
  // warp reduce
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffff, partial, off);
  float ip2 = __shfl_sync(0xffffffff, partial, 0);  // broadcast

  /* ---------- part-2: load onorm & xip -------------------------- */
  const uint32_t* block_short = d_short_data + (size_t)(idx_in_cluster + cluster_offset) *
                                                 (short_block_uint32 + num_short_factors);
  // onorm is first float after the code section:
  float onorm = __ldg(reinterpret_cast<const float*>(block_short + short_block_uint32));

  float xip = __ldg(&d_ex_factor[idx_in_cluster + cluster_offset].xipnorm);

  /* ---------- part-3: refine distance -------------------------- */
  if (lane == 0) {
    float refined = onorm * onorm + sqr_y -
                    xip * qnorm * (FAC_RESCALE * approx + ip2 - (FAC_RESCALE - 0.5f) * sumq);

#ifdef DEBUG_MULTIPLE_SEARCH
    //        printf("onorm: %f\n", onorm);
    //        printf("sqr_y: %f\n", sqr_y);
    //        printf("xip: %f\n", xip);
    //        printf("approx: %f\n", approx);
    //        printf("sumq: %f\n", sumq);
    //        printf("ip2: %f\n", ip2);
    //        printf("1bit dis: %f\n", candidate->dist);
    //        printf("refined dist: %f\n", refined);
    //        printf("id: %d\n", candidate->pid);
    //        printf("warp id: %d, cluster id: %d, index in cluster: %d, refined: %f\n", warpId,
    //        probe, idx_in_cluster, refined); if (refined < candidate->dist) {
    printf(
      "onorm: %f, sqr_y: %f, xip: %f, approx: %f, sumq: %f, ip2: %f, 1bit dis: %f, warp id: %d, "
      "cluster id: %d, index in cluster: %d, refined: %f\n",
      onorm,
      sqr_y,
      xip,
      approx,
      sumq,
      ip2,
      candidate->dist,
      warpId,
      probe,
      idx_in_cluster,
      refined);
//        }
#endif
    candidate->dist = refined;
  }
}

__global__ void copyTopKKernel(const Candidate4* __restrict__ src,
                               float* __restrict__ dst_dist,
                               uint32_t* __restrict__ dst_pid,
                               int K)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < K) {
    dst_dist[i] = src[i].dist;
    dst_pid[i]  = src[i].pid;
  }
}

float write_topk_to_pool_from_candidates(DeviceResultPool& d_pool,  // lives on device
                                         Candidate4* d_candidates,  // KM elements
                                         int KM,
                                         int TOPK,  // == d_pool->capacity  (caller guarantees)
                                         rmm::cuda_stream_view stream)
{
  if (KM == 0) {  // nothing to copy
    return INFINITY;
  }

  const int Kwrite = min(KM, TOPK);  // how many we will store
  dim3 blk(256), grd((Kwrite + blk.x - 1) / blk.x);
  auto par = thrust::cuda::par.on(stream);

  //------------------------------------------------------------------
  // CASE 1 :  KM > TOPK  → full sort (need the best TOPK only)
  //------------------------------------------------------------------
  if (KM > TOPK) {
    thrust::sort(
      par,
      d_candidates,
      d_candidates + KM,
      [] __device__(const Candidate4& a, const Candidate4& b) { return a.dist < b.dist; });

    copyTopKKernel<<<grd, blk, 0, stream>>>(
      d_candidates, d_pool.distances.data_handle(), d_pool.ids.data_handle(), TOPK);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // disable it for multiple search
    //        float kth;
    //        cudaMemcpyAsync(&kth,
    //                                   d_pool->distances + (TOPK - 1),
    //                                   sizeof(float),
    //                                   cudaMemcpyDeviceToHost, stream);
    //        cudaStreamSynchronize(stream);
    return 1;  // TOP-K-th distance
  }

  //------------------------------------------------------------------
  // CASE 2 :  KM ≤ TOPK  → no sort, copy all KM winners as-is
  //------------------------------------------------------------------
  copyTopKKernel<<<grd, blk, 0, stream>>>(
    d_candidates, d_pool.distances.data_handle(), d_pool.ids.data_handle(), KM);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // compute max(dist) of the KM elements to return as “K-th”
  //    float kth = thrust::reduce(par,
  //                               thrust::device_pointer_cast(d_pool->distances),
  //                               thrust::device_pointer_cast(d_pool->distances + KM),
  //                               -FLT_MAX,
  //                               thrust::maximum<float>());

  // (optional) update pool size on device
  // set_pool_size_kernel<<<1,1,0,stream>>>(d_pool, KM);

  return -0.1;  // just a flag
                // actually the KM-th (largest) distance
}
static constexpr int MAX_D = 2048;  // vector dimensionality (multiple of 4)
__constant__ float c_query[MAX_D];  // broadcast query (<= 64 kB)
// when calling, nprobe should be actual_nprobe - 1;
// GPUClusterMeta is supposed to be selected Meta of nprobes
void SearcherGPU::SearchMultipleClusters(const IVFGPU& cur_ivf,
                                         IVFGPU::GPUClusterMeta* d_cluster_meta,
                                         Candidate* d_centroid_candidates,
                                         DeviceResultPool& KNNs,
                                         float* d_centroid,
                                         const float* h_query,
                                         size_t nprobe)
{
  // (1) Normalize query to different clusters
  // first copy query to constant memory
  //    cudaMemcpyToSymbolAsync(c_query,h_query,sizeof(float)*D);
  // copy to global memory
  float* c_query = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&c_query, sizeof(float) * D, stream_));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(c_query, h_query, sizeof(float) * D, cudaMemcpyHostToDevice, stream_));

#ifdef DEBUG_MULTIPLE_SEARCH
  raft::resource::sync_stream(handle_);
  RAFT_CUDA_TRY(cudaGetLastError());
  printf("copy kernel done\n");
#endif

  // normalize query to multiple centroids
  dim3 np(nprobe);
  normalize_warp32_kernel<<<np, 32, 0, stream_>>>(
    d_centroid_candidates, d_centroid, d_unit_q_gpu, d_sum_norm, c_query, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
#ifdef DEBUG_MULTIPLE_SEARCH
  raft::resource::sync_stream(handle_);
  printf("normalize_warp32_kernel done\n");
#endif

  // (1.5) build prefix sum array for nprobe for the following computation (maybe better on CPU?)
  int* d_starts = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_starts, sizeof(int) * (nprobe + 1), stream_));
  build_prefix_device(d_cluster_meta, nprobe, d_starts, stream_);
#ifdef DEBUG_MULTIPLE_SEARCH
  printf("build_prefix_device done\n");
  printf("h_filter_distk: %f\n", h_filter_distk);
  raft::resource::sync_stream(handle_);
  RAFT_CUDA_TRY(cudaGetLastError());
#endif

  // (2) Compute IP for all clusters
  int BLOCK = 256;
  //    int* h_vecs_to_compute = nullptr;                 // 1. allocate pinned host memory
  //    cudaMallocHost(&h_vecs_to_compute, sizeof(int));  //    (returns page-locked memory)
  int h_vecs_to_compute;
  RAFT_CUDA_TRY(cudaMemcpyAsync(  // 2. async copy device → pinned host
    &h_vecs_to_compute,           // dest (pinned)
    &d_starts[nprobe],            // src  (device)
    sizeof(int),
    cudaMemcpyDeviceToHost,
    stream_));
  raft::resource::sync_stream(handle_);     // or cudaEvent, or wait on the stream elsewhere
  int vecs_to_compute = h_vecs_to_compute;  // now it’s safe to use
#ifdef DEBUG_MULTIPLE_SEARCH
  printf("vecs_to_compute: %d\n", vecs_to_compute);
#endif
  int warps = vecs_to_compute;  //
  int grids = (warps * 32 + BLOCK - 1) / BLOCK;
  dim3 grid(grids), block(BLOCK);
  compute_ip_kernel_fp32_multi<<<grids, BLOCK, 0, stream_>>>(
    d_unit_q_gpu,
    d_sum_norm,
    d_cluster_meta,
    D,
    cur_ivf.get_short_data_device(),
    cur_ivf.quantizer().num_short_factors(),
    cur_ivf.quantizer().short_code_length(),
    d_starts,
    nprobe,
    one_over_sqrtD,
    h_filter_distk,
    d_ip_results,
    d_est_dis);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef DEBUG_MULTIPLE_SEARCH
  printf("compute_ip_kernel_fp32_multi done\n");
  raft::resource::sync_stream(handle_);
//    RAFT_CUDA_TRY(cudaGetLastError());
#endif

  // (3) Gather Filtered Distances
  int KM      = KNNs.capacity * 10;
  int written = fast_select_and_keep_multi(d_est_dis,
                                           d_ip_results,
                                           cur_ivf.get_ids_device(),
                                           d_starts,
                                           vecs_to_compute,
                                           nprobe,
                                           KM,
                                           d_filter_distk,
                                           d_cluster_meta,
                                           d_candidate_buffer,
                                           stream_);
#ifdef DEBUG_MULTIPLE_SEARCH
  printf("fast_select_and_keep_multi done\n");
#endif

  // (4) access and compute long codes **and restore distances*** based on the candidate buffer
  int nWarps = written;  // 1 warp / candidate
  grid       = (nWarps * 32 + BLOCK - 1) / BLOCK;
  refine_multi_kernel<<<grid, BLOCK, 0, stream_>>>(d_candidate_buffer,
                                                   written,
                                                   d_unit_q_gpu,
                                                   d_sum_norm,
                                                   d_starts,
                                                   nprobe,
                                                   D,
                                                   d_cluster_meta,
                                                   cur_ivf.get_long_code_device(),
                                                   cur_ivf.quantizer().long_code_length(),
                                                   cur_ivf.get_short_data_device(),
                                                   cur_ivf.quantizer().short_code_length(),
                                                   cur_ivf.get_ex_factor_device(),
                                                   cur_ivf.quantizer().num_short_factors(),
                                                   FAC_RESCALE,
                                                   BLOCK,
                                                   cur_ivf.get_ex_bits());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
#ifdef DEBUG_MULTIPLE_SEARCH
  printf("refine_multi_kernel done\n");
#endif

  // (5) Finally sort the buffers
  //    float kth =
  write_topk_to_pool_from_candidates(KNNs, d_candidate_buffer, written, KNNs.capacity, stream_);
//    printf("written: %d, kth: %f\n", written, kth);
#ifdef DEBUG_MULTIPLE_SEARCH
  printf("write_topk_to_pool_from_candidates done\n");
#endif

  // Update filter dist if possible
  //    if (kth > 0) {
  //        if (kth <h_filter_distk) {
  //            h_filter_distk = kth;
  //        }
  //    }
  KNNs.size = written < KNNs.capacity ? written : KNNs.capacity;

#ifdef DEBUG_MULTIPLE_SEARCH
  // copy back KNNs from a device and print data inside
  printf("written: %d\n", written);
  printf("Knns capacity: %d\n", KNNs.capacity);
  DeviceResultPool host_knns;
  host_knns.capacity  = KNNs.capacity;
  host_knns.distances = (float*)malloc(sizeof(float) * KNNs.capacity);
  host_knns.ids       = (uint32_t*)malloc(sizeof(uint32_t) * KNNs.capacity);
  RAFT_CUDA_TRY(cudaMemcpy(
    host_knns.distances, KNNs.distances, sizeof(float) * KNNs.capacity, cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(
    cudaMemcpy(host_knns.ids, KNNs.ids, sizeof(uint32_t) * KNNs.capacity, cudaMemcpyDeviceToHost));
  for (int i = 0; i < KNNs.size; i++) {
    printf("dist: %f\n", host_knns.distances[i]);
    printf("id: %d\n", host_knns.ids[i]);
  }
  free(host_knns.distances);
  free(host_knns.ids);
#endif

  // (
  RAFT_CUDA_TRY(cudaFreeAsync(d_starts, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(c_query, stream_));
  //    cudaFreeHost(h_vecs_to_compute);
}

void SearcherGPU::destroy() noexcept
{
  /* —— GPU buffers —— */
  safeCudaFreeAsync(d_filter_distk);
  safeCudaFreeAsync(d_unit_q_gpu);
  safeCudaFreeAsync(d_ip_results);
  safeCudaFreeAsync(d_est_dis);
  safeCudaFreeAsync(d_top_ip);
  safeCudaFreeAsync(d_top_pids);
  safeCudaFreeAsync(d_top_idx);
  safeCudaFreeAsync(d_ip2);
  safeCudaFreeAsync(d_buf);
  safeCudaFreeAsync(d_topk_threshold);
  safeCudaFreeAsync(d_centroid_candidates);
  // jamxia edit: following 3 lines
  safeCudaFreeAsync(d_centroid_distances);
  safeCudaFreeAsync(d_c_norms);
  safeCudaFreeAsync(d_q_norms);
  //    safeCudaFreeAsync(d_topk_results);

  /* —— host‑side buffers we *own* —— */
  safeHostFree(quant_query);  // int16_t*
  safeHostFree(unit_q);       // float*

  /* Everything else (e.g.  query) is not owned → NO free */
}

void SearcherGPU::AllocateSearcherSpace(const IVFGPU& cur_ivf,
                                        size_t num_queries,
                                        size_t k,
                                        size_t max_nprobes,
                                        size_t max_cluster_length)
{
  //    cudaMallocAsync(&d_top_ip, sizeof(float) * max_cluster_length * num_queries, s);
  //    cudaMallocAsync(&d_est_dis, sizeof(float) * max_cluster_length * num_queries, s);
  //    cudaMallocAsync(&d_topk_threshold, sizeof(float) * num_queries, s);
  //    topk_results.resize(num_queries);
  //    for (int i = 0; i < num_queries; i++) {
  //        topk_results[i] = createDeviceResultPool(k, s);
  //    }
  //    cudaMallocAsync(&d_top_idx, sizeof(uint32_t)  * max_cluster_length * num_queries, s);
  //    cudaMallocAsync(&d_top_pids, sizeof(PID)  * max_cluster_length * num_queries, s);
  //    cudaMallocAsync(&d_ip2, sizeof(float) * max_cluster_length * num_queries, s);
  //    cudaMallocAsync(&d_centroid_candidates, sizeof(Candidate) * max_nprobes * num_queries, s);
  RAFT_CUDA_TRY(cudaMallocAsync(
    &d_centroid_distances, sizeof(float) * num_queries * cur_ivf.get_num_centroids(), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync(&d_c_norms, sizeof(float) * cur_ivf.get_num_centroids(), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync(&d_q_norms, sizeof(float) * num_queries, stream_));
  raft::resource::sync_stream(handle_);
};

__global__ void compute_ip_kernel(const int16_t* quant_query_gpu,
                                  const uint32_t* rabitq_codes_and_factors,
                                  size_t num_dimensions,
                                  size_t num_short_factors,
                                  size_t num_vector_cluster,
                                  float* ip_results,
                                  float* est_dis,
                                  float delta,
                                  float sumq,
                                  float qnorm,
                                  float one_over_sqrtD)
{
  // Global thread index
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Which warp this thread belongs to
  int warp_id = tid / WARP_SIZE;
  // Lane within the warp [0..31]
  int lane = threadIdx.x & (WARP_SIZE - 1);

  if (warp_id >= (int)num_vector_cluster) return;

  // Number of 32-bit words storing the 1‑bit vector
  const int num_words = (int)((num_dimensions + 31) / 32);
  // Pointer to the start of this vector’s block
  const uint32_t* block = rabitq_codes_and_factors + warp_id * (num_words + (int)num_short_factors);

  // 1) Compute partial inner product for this lane
  float partial = 0.0f;
  for (int d = lane; d < (int)num_dimensions; d += WARP_SIZE) {
    // locate the bit for dimension d
    uint32_t word = block[d >> 5];                  // d/32
    int bit       = (word >> (31 - (d & 31))) & 1;  // extract bit
    if (bit) {
      partial += float(quant_query_gpu[d]);
      //            printf("quanted query [%d]: %d\n " , d, quant_query_gpu[d]);
    }
  }

  // 2) Warp‐level reduction of partial sums → full res in lane 0
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffff, partial, offset);
  }

  if (lane == 0) {
    float res = partial;
    // read the first factor (onorm), which lives immediately after the num_words uint32s
    float onorm = (reinterpret_cast<const float*>(block))[num_words];
    // apply your final est distance
    //        float result = (res * delta
    //                       - (0.5f * sumq - 0.58f)) * (-5.0f * qnorm * one_over_sqrtD) * onorm
    //                       + qnorm * qnorm
    //                       + onorm * onorm;
    // only use ip_xb_prime;
    float ip_xb_prime = res * delta;
    float result =
      (ip_xb_prime - (0.5f * sumq - 0.58f)) * (-5.0f * qnorm * one_over_sqrtD) * onorm +
      qnorm * qnorm + onorm * onorm;
    ip_results[warp_id] = ip_xb_prime;
    est_dis[warp_id]    = result;
    //        if (warp_id == 1) {
    //            printf("warp id: %d, word: %x, ip_xb_prime: %f, result: %f, onorm: %f\n", warp_id,
    //            block[0], ip_xb_prime, result, onorm);
    //        }
  }
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
