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

#ifndef EXRABITQ_SPACE_CUDA_H
#define EXRABITQ_SPACE_CUDA_H

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

struct L2Functor {
  __host__ __device__ float operator()(const thrust::tuple<float, float>& t) const
  {
    float x    = thrust::get<0>(t);
    float y    = thrust::get<1>(t);
    float diff = x - y;
    return diff * diff;
  }
};

float L2SqrThrust(const float* h_x, const float* h_y, size_t N);

float L2SqrCPU_STL(const float* h_x, const float* h_y, size_t N);

float L2Sqr_CUDA(const float* x, const float* y, size_t L);

void high_acc_quantize16_scalar(int16_t* __restrict__ result,
                                const float* __restrict__ q,
                                float& width,
                                size_t D);

inline float normalize_query16_scalar(float* unit_q,  // out: normalised vector (length D)
                                      const float* __restrict__ q,
                                      const float* __restrict__ c,
                                      float norm,
                                      size_t D)
{
  constexpr float eps = 1e-5f;

  if (norm > eps) {
    const float inv_norm = 1.0f / norm;
    float sum            = 0.0f;

    for (size_t i = 0; i < D; ++i) {
      float u = (q[i] - c[i]) * inv_norm;  // (q - c) / norm
      //            float u = q[i]; // (q - c) / norm       // use this for float
      unit_q[i] = u;
      sum += u;  // running sum
    }
    return sum;  // same as _mm512_reduce_add_ps
  } else         // q‑c is (almost) the zero vector
  {
    float value = 1.0f / std::sqrt(static_cast<float>(D));  // 1 / √D
    std::fill(unit_q, unit_q + D, value);
    return static_cast<float>(D) * value;  // == √D
  }
}

inline float compute_sum_q(const float* __restrict__ q, size_t D)
{
  // jamxia edit
  // constexpr float eps = 1e-5f;
  float sum = 0.0f;
  for (size_t i = 0; i < D; ++i) {
    float u = q[i];  // (q - c) / norm       // use this for float
    sum += u;        // running sum
  }
  return sum;  // same as _mm512_reduce_add_ps
}

#endif  // EXRABITQ_SPACE_CUDA_H
