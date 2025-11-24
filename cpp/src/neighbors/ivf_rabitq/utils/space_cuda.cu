/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_rabitq/utils/space_cuda.cuh>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <algorithm>  // std::max, std::min, std::clamp
#include <cmath>      // std::abs
#include <cstdint>    // int16_t, int32_t
#include <limits>     // std::numeric_limits
#include <numeric>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

float L2SqrThrust(const float* h_x, const float* h_y, size_t N)
{
  // Copy host data to device vectors
  thrust::device_vector<float> d_x(h_x, h_x + N);
  thrust::device_vector<float> d_y(h_y, h_y + N);

  //    printf("I'm in!");
  // Create zip iterators
  auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(), d_y.begin()));
  auto end   = thrust::make_zip_iterator(thrust::make_tuple(d_x.end(), d_y.end()));

  // Compute the L2 squared distance using transform_reduce
  float result = thrust::transform_reduce(begin,
                                          end,                   // Input range
                                          L2Functor(),           // Unary operation
                                          0.0f,                  // Initial value
                                          thrust::plus<float>()  // Summation operation
  );

  return result;
}

float L2SqrCPU_STL(const float* h_x, const float* h_y, size_t N)
{
  // Using STL algorithms (C++11 or later)
  return std::inner_product(
    h_x,
    h_x + N,                // First range
    h_y,                    // Second range begin
    0.0f,                   // Initial value
    std::plus<float>(),     // Sum operation
    [](float a, float b) {  // Product operation (replaced with squared difference)
      float diff = a - b;
      return diff * diff;
    });
}

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 每个线程块处理256个元素，可根据GPU架构调整
constexpr int BLOCK_SIZE = 256;

// 计算剩余尾部元素的核函数
__global__ void l2sqr_tail_kernel(const float* __restrict__ x,
                                  const float* __restrict__ y,
                                  float* __restrict__ output,
                                  size_t start_idx,
                                  size_t L)
{
  const int tid    = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;

  float sum = 0.0f;
  for (size_t i = start_idx + tid; i < L; i += stride) {
    float diff = x[i] - y[i];
    sum += diff * diff;
  }

  // 块内归约
  __shared__ float smem[BLOCK_SIZE];
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
    __syncthreads();
  }

  if (threadIdx.x == 0) { atomicAdd(output, smem[0]); }
}

// 主计算核函数（批量处理16的倍数部分）
__global__ void l2sqr_main_kernel(const float* __restrict__ x,
                                  const float* __restrict__ y,
                                  float* __restrict__ output,
                                  size_t L)
{
  const int tid    = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;

  float sum = 0.0f;

  // 每个线程处理4个元素（提高指令级并行）
  for (size_t base = tid * 4; base < L; base += stride * 4) {
    const size_t remaining = min(L - base, 4UL);

    for (size_t i = 0; i < remaining; ++i) {
      const size_t idx = base + i;
      float diff       = x[idx] - y[idx];
      sum += diff * diff;
    }
  }

  // 块内归约
  __shared__ float smem[BLOCK_SIZE];
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
    __syncthreads();
  }

  if (threadIdx.x == 0) { atomicAdd(output, smem[0]); }
}

// 主机函数封装
float L2Sqr_CUDA(raft::resources const& handle, const float* x, const float* y, size_t L)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  float* d_output;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_output, sizeof(float), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_output, 0, sizeof(float), stream));

  // 计算主部分（16倍数部分）
  const size_t num16 = L - (L % 16);
  if (num16 > 0) {
    const int grid_size = (num16 + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
    l2sqr_main_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(x, y, d_output, num16);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // 处理尾部剩余元素
  if (L > num16) {
    const int tail_grid = (L - num16 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    l2sqr_tail_kernel<<<tail_grid, BLOCK_SIZE, 0, stream>>>(x, y, d_output, num16, L);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // 回传结果
  float result;
  RAFT_CUDA_TRY(cudaMemcpyAsync(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(handle);
  RAFT_CUDA_TRY(cudaFreeAsync(d_output, stream));

  return result;
}

////--------------------------------------------------------------------
//// Scalar helper: find minimum and maximum in a float array
////--------------------------------------------------------------------
// inline void data_range16_scalar(const float* __restrict__ q,
//                                 float& vl,
//                                 float& vr,
//                                 size_t L)
//{
//     vl =  std::numeric_limits<float>::max();
//     vr = -std::numeric_limits<float>::max();
//
//     for (size_t i = 0; i < L; ++i) {
//         float v = q[i];
//         vl = (v < vl) ? v : vl;
//         vr = (v > vr) ? v : vr;
//     }
// }
//
////--------------------------------------------------------------------
//// Portable quantizer (no AVX/SSE required)
////--------------------------------------------------------------------
// void high_acc_quantize16_scalar(int16_t* __restrict__ result,
//                                 const float*  __restrict__ q,
//                                 float& width,
//                                 size_t D)
//{
//     constexpr int BQ          = 14;                    // signed‑14‑bit
//     constexpr int32_t MAX_Q   =  (1 << (BQ - 1)) - 1;  // +8191
//     constexpr int32_t MIN_Q   = -(1 << (BQ - 1));      // -8192
//
//     // 1. range scan --------------------------------------------------
//     float vl, vr;
//     data_range16_scalar(q, vl, vr, D);
//     float vmax = std::max(std::abs(vl), std::abs(vr));
//
//     // 2. compute bin width ------------------------------------------
//     width = vmax / MAX_Q;               // == vmax / 8191
//     float inv_width = (width > 0.f) ? 1.0f / width : 0.0f;
//
//     // 3. element‑wise quantisation ----------------------------------
//     for (size_t i = 0; i < D; ++i)
//     {
//         float  scaled = q[i] * inv_width;
//
//         // original rounding rule: round half away from zero
//         float  biased = scaled + ((scaled >= 0.f) ? 0.5f : -0.5f);
//         int32_t q32   = static_cast<int32_t>(biased);
//
//         // saturate to signed‑14‑bit range (safety, avoids UB on cast)
//         q32 = std::clamp(q32, MIN_Q, MAX_Q);
//
//         result[i] = static_cast<int16_t>(q32);
//     }
// }

//--------------------------------------------------------------------
// Scalar helper: find minimum and maximum in a float array
//--------------------------------------------------------------------
inline void data_range16_scalar(const float* __restrict__ q, float& vl, float& vr, size_t L)
{
  vl = std::numeric_limits<float>::max();
  vr = -std::numeric_limits<float>::max();

  for (size_t i = 0; i < L; ++i) {
    float v = q[i];
    vl      = (v < vl) ? v : vl;
    vr      = (v > vr) ? v : vr;
  }
}

/*--------------------------------------------------------------------
 * Quantiser that matches `high_acc_quantize16`
 *
 *  • Rounding  : “nearest‑even” (default MXCSR mode used by
 *                _mm512_cvtps_epi32).
 *  • Saturation: to the *signed‑16‑bit* range (hardware behaviour of
 *                _mm512_cvtepi32_epi16), **not** to ±8191.
 *-------------------------------------------------------------------*/
void high_acc_quantize16_scalar(int16_t* __restrict__ result,
                                const float* __restrict__ q,
                                float& width,
                                size_t D)
{
  constexpr int BQ        = 14;                   // signed‑14‑bit
  constexpr int32_t MAX_Q = (1 << (BQ - 1)) - 1;  // 8191

  // 1) range scan --------------------------------------------------
  float vl, vr;
  data_range16_scalar(q, vl, vr, D);
  float vmax = std::max(std::abs(vl), std::abs(vr));

  // 2) compute bin width ------------------------------------------
  width           = vmax / MAX_Q;  // identical to SIMD path
  float inv_width = (width > 0.f) ? 1.0f / width : 0.0f;

  // 3) element‑wise quantisation ----------------------------------
  for (size_t i = 0; i < D; ++i) {
    float scaled = q[i] * inv_width;

    /*  The intrinsic `_mm512_cvtps_epi32` uses the current MXCSR
     *  rounding mode (default: round‑to‑nearest‑even).  `std::lrintf`
     *  gives the *same* rounding on most platforms *without* changing
     *  the global rounding mode.
     *
     *  – ties (exactly ±0.5) are rounded to the even integer
     *    →  1.5 → 2,   0.5 → 0,  -0.5 → 0,  -1.5 → -2
     */
    int32_t q32 = static_cast<int32_t>(std::lrintf(scaled));

    /*  Hardware conversion to int16 in the SIMD path saturates to
     *  INT16_MIN … INT16_MAX.  We reproduce that explicitly to avoid
     *  UB on the cast in scalar C++.
     */
    q32 = std::clamp(q32, INT16_MIN, INT16_MAX);

    result[i] = static_cast<int16_t>(q32);
  }
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
