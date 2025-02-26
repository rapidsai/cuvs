/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuvs/preprocessing/quantize/binary.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/stats/mean.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/system/omp/execution_policy.h>

namespace cuvs::preprocessing::quantize::detail {

template <typename T>
RAFT_KERNEL sample_and_transpose_dataset_kernel(const T* const in_ptr,
                                                const uint32_t ldi,
                                                const size_t dataset_size,
                                                const uint32_t dim_chunk,
                                                const size_t num_samples,
                                                const size_t stride,
                                                T* const out_ptr,
                                                const uint32_t ldo)
{
  const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (size_t i = tid; i < num_samples * dim_chunk; i += gridDim.x * blockDim.x) {
    const auto sampled_mat_i = i % num_samples;
    const auto in_mat_i      = (sampled_mat_i * stride) % dataset_size;
    const auto mat_j         = i / num_samples;

    out_ptr[sampled_mat_i + mat_j * ldo] = in_ptr[in_mat_i * ldi + mat_j];
  }
}

template <typename T>
inline void sample_and_transpose_dataset(const T* const in_ptr,
                                         const uint32_t ldi,
                                         const size_t dataset_size,
                                         const uint32_t dim_chunk,
                                         const size_t num_samples,
                                         T* const out_ptr,
                                         const uint32_t ldo,
                                         cudaStream_t cuda_stream)
{
  constexpr uint32_t block_size = 256;
  const auto grid_size =
    std::min(raft::div_rounding_up_safe<size_t>(num_samples * dim_chunk, block_size), 2048lu);

  // Note that one of the candidates must be prime to the dataset size because the product of the
  // prime candidates exceeds the maximum of U64.
  size_t stride_prime_list[] = {611323lu, 611333lu, 611389lu, 611393};
  uint32_t prime_i           = 0;
  while (dataset_size % stride_prime_list[prime_i] == 0) {
    prime_i++;
  }

  sample_and_transpose_dataset_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
    in_ptr, ldi, dataset_size, dim_chunk, num_samples, stride_prime_list[prime_i], out_ptr, ldo);
}

template <class T>
_RAFT_HOST_DEVICE bool is_positive(const T& a)
{
  return a > 0;
}

template <>
_RAFT_HOST_DEVICE bool is_positive<half>(const half& a)
{
  return is_positive(static_cast<float>(a));
}

template <class T, uint32_t block_size, class pack_t = uint8_t>
RAFT_KERNEL binary_quantization_kernel(const T* const in_ptr,
                                       const uint32_t ldi,
                                       const T* const threshold_ptr,
                                       const size_t dataset_size,
                                       const uint32_t dataset_dim,
                                       pack_t* const out_ptr,
                                       const uint32_t ldo)
{
  constexpr uint32_t warp_size = 32;
  const uint32_t bits_per_pack = sizeof(pack_t) * 8;
  const auto output_dim        = raft::div_rounding_up_safe(dataset_dim, bits_per_pack);

  const auto vector_id = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
  if (vector_id >= dataset_size) { return; }

  __shared__ pack_t smem[block_size];
  auto local_smem = smem + (threadIdx.x / warp_size) * warp_size;

  const auto lane_id = threadIdx.x % warp_size;
  for (uint32_t j_offset = 0; j_offset < dataset_dim; j_offset += warp_size * bits_per_pack) {
    // Load dataset vector elements with coalesce access. The mapping of the vector element position
    // and the `pack` register is as follows:
    //
    // lane_id | LSB  (pack(u8))  MSB
    //       0 |  0, 32, 64, ..., 224
    //       1 |  1, 33, 65, ..., 225
    //       ...
    //      31 | 31, 63, 95, ..., 255
    pack_t pack = 0;
    for (uint32_t bit_offset = 0; bit_offset < bits_per_pack; bit_offset++) {
      const auto j = j_offset + lane_id + bit_offset * warp_size;
      if (j < dataset_dim) {
        auto v = in_ptr[static_cast<size_t>(vector_id) * ldi + j];
        if (threshold_ptr != nullptr) { v -= threshold_ptr[j]; }
        if (is_positive(v)) { pack |= (1u << bit_offset); }
      }
    }

    // Store the local result in smem so that the other threads in the same warp can read
    local_smem[lane_id] = pack;

    // Store the result with (a kind of) transposition so that the the coalesce access can be used.
    // The mapping of the result `pack` register bit position and (smem_index, bit_position) is as
    // follows:
    //
    // lane_id | LSB          (pack(u8))         MSB
    //       0 | ( 0,0), ( 1,0), ( 2,0), ..., ( 7,0)
    //       1 | ( 8,0), ( 9,0), (10,0), ..., (15,0)
    //       ...
    //       4 | ( 0,1), ( 1,1), ( 2,1), ..., ( 7,1)
    //       ...
    //      31 | (24,7), (25,7), (26,7), ..., (31,7)
    //
    // The `bit_position`-th bit of `local_smem[smem_index]` is copied to the corresponding `pack`
    // bit. By this mapping, the quantization result of 8*i-th ~ (8*(i+1)-1)-th vector elements is
    // stored by the lane_id=i thread.
    pack                    = 0;
    const auto smem_start_i = (lane_id % (warp_size / bits_per_pack)) * bits_per_pack;
    const auto mask         = 1u << (lane_id / (warp_size / bits_per_pack));
    for (uint32_t j = 0; j < bits_per_pack; j++) {
      if (local_smem[smem_start_i + j] & mask) { pack |= (1u << j); }
    }

    const auto out_j = j_offset / bits_per_pack + lane_id;
    if (out_j < output_dim) { out_ptr[vector_id * ldo + out_j] = pack; }
  }
}

template <typename T, typename QuantI = uint8_t>
void transform(raft::resources const& res,
               cuvs::preprocessing::quantize::binary::params params,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  cudaStream_t stream            = raft::resource::get_cuda_stream(res);
  const uint32_t bits_per_pack   = sizeof(QuantI) * 8;
  const uint32_t dataset_dim     = dataset.extent(1);
  const uint32_t out_dim         = out.extent(1);
  const size_t dataset_size      = dataset.extent(0);
  const size_t out_dataset_size  = out.extent(0);
  const uint32_t minimul_out_dim = raft::div_rounding_up_safe(dataset_dim, bits_per_pack);
  RAFT_EXPECTS(out_dim >= minimul_out_dim,
               "The quantized dataset dimension must be larger or equal to "
               "%u but is %u passed",
               minimul_out_dim,
               out_dim);
  RAFT_EXPECTS(out_dataset_size >= dataset_size,
               "The quantized dataset size must be larger or equal to "
               "the input dataset size (%lu) but is %lu passed",
               dataset_size,
               out_dataset_size);
  // stats::mean does not support F16
  RAFT_EXPECTS(!(std::is_same_v<T, half> && params.threshold == binary::set_bit_threshold::mean),
               "binary::transform does not support threshold == mean for FP16 dataset");

  auto mr = raft::resource::get_workspace_resource(res);
  auto threshold_vec =
    raft::make_device_mdarray<T, int64_t>(res, mr, raft::make_extents<int64_t>(0));

  T* threshold_ptr = nullptr;

  if (params.threshold == cuvs::preprocessing::quantize::binary::set_bit_threshold::mean ||
      params.threshold ==
        cuvs::preprocessing::quantize::binary::set_bit_threshold::sampling_median) {
    threshold_vec = raft::make_device_mdarray<T, std::int64_t>(
      res, mr, raft::make_extents<std::int64_t>(dataset_dim));
    threshold_ptr = threshold_vec.data_handle();
  }

  if (params.threshold == cuvs::preprocessing::quantize::binary::set_bit_threshold::mean) {
    // stats::mean does not support F16
    if constexpr (!std::is_same_v<T, half>) {
      raft::stats::mean(
        res,
        dataset,
        raft::make_device_vector_view<T, int64_t>(threshold_vec.data_handle(), dataset_dim),
        false);
    }
  }

  if (params.threshold ==
      cuvs::preprocessing::quantize::binary::set_bit_threshold::sampling_median) {
    // Make the number of samples odd so that the median is calculated by only sort and memcpy
    const size_t num_sampls =
      std::max(
        raft::div_rounding_up_safe(static_cast<size_t>(dataset_size * params.sampling_ratio), 2lu) *
          2lu,
        2lu) -
      1;
    const size_t data_size_per_vector = sizeof(T) * dataset_dim;
    const uint32_t max_dim_chunk =
      std::min(std::max(1lu, raft::resource::get_workspace_free_bytes(res) / data_size_per_vector),
               static_cast<std::size_t>(dataset_dim));

    auto sampled_dataset_chunk = raft::make_device_mdarray<T, int64_t>(
      res, mr, raft::make_extents<int64_t>(max_dim_chunk, num_sampls));
    for (uint32_t dim_offset = 0; dim_offset < dataset_dim; dim_offset += max_dim_chunk) {
      const auto dim_chunk = std::min(max_dim_chunk, dataset_dim - dim_offset);

      sample_and_transpose_dataset<T>(dataset.data_handle() + dim_offset,
                                      dataset_dim,
                                      dataset_size,
                                      dim_chunk,
                                      num_sampls,
                                      sampled_dataset_chunk.data_handle(),
                                      num_sampls,
                                      raft::resource::get_cuda_stream(res));
      for (uint32_t i = 0; i < dim_chunk; i++) {
        auto start_ptr = sampled_dataset_chunk.data_handle() + i * num_sampls;
        thrust::sort(thrust::device, start_ptr, start_ptr + num_sampls);
      }

      RAFT_CUDA_TRY(cudaMemcpy2DAsync(threshold_ptr + dim_offset,
                                      sizeof(T),
                                      sampled_dataset_chunk.data_handle() + (num_sampls - 1) / 2,
                                      num_sampls * sizeof(T),
                                      sizeof(T),
                                      dim_chunk,
                                      cudaMemcpyDefault,
                                      raft::resource::get_cuda_stream(res)));
    }
  }

  constexpr uint32_t warp_size    = 32;
  constexpr uint32_t block_size   = 256;
  constexpr uint32_t vecs_per_cta = block_size / warp_size;
  const auto grid_size =
    raft::div_rounding_up_safe(dataset_size, static_cast<size_t>(vecs_per_cta));

  binary_quantization_kernel<T, block_size>
    <<<grid_size, block_size, 0, stream>>>(dataset.data_handle(),
                                           dataset.stride(0),
                                           threshold_ptr,
                                           dataset_size,
                                           dataset_dim,
                                           out.data_handle(),
                                           out.stride(0));
}

template <typename T, typename QuantI = uint8_t>
void transform(raft::resources const& res,
               cuvs::preprocessing::quantize::binary::params params,
               raft::host_matrix_view<const T, int64_t> dataset,
               raft::host_matrix_view<QuantI, int64_t> out)
{
  const uint32_t bits_per_pack   = sizeof(QuantI) * 8;
  const uint32_t dataset_dim     = dataset.extent(1);
  const uint32_t out_dim         = out.extent(1);
  const size_t dataset_size      = dataset.extent(0);
  const size_t out_dataset_size  = out.extent(0);
  const uint32_t minimul_out_dim = raft::div_rounding_up_safe(dataset_dim, bits_per_pack);
  RAFT_EXPECTS(out_dim >= minimul_out_dim,
               "The quantized dataset dimension must be larger or equal to "
               "%u but is %u passed",
               minimul_out_dim,
               out_dim);
  RAFT_EXPECTS(out_dataset_size >= dataset_size,
               "The quantized dataset size must be larger or equal to "
               "the input dataset size (%lu) but is %lu passed",
               dataset_size,
               out_dataset_size);

  auto threshold_vec = raft::make_host_vector<T, int64_t>(0);

  T* threshold_ptr = nullptr;

  if (params.threshold == cuvs::preprocessing::quantize::binary::set_bit_threshold::mean ||
      params.threshold ==
        cuvs::preprocessing::quantize::binary::set_bit_threshold::sampling_median) {
    threshold_vec = raft::make_host_vector<T, int64_t>(dataset_dim);
    threshold_ptr = threshold_vec.data_handle();
  }

  if (params.threshold == cuvs::preprocessing::quantize::binary::set_bit_threshold::mean) {
    for (uint32_t j = 0; j < dataset_dim; j++) {
      threshold_ptr[j] = 0;
    }
    if constexpr (!std::is_same_v<T, half>) {
#pragma omp parallel for reduction(+ : threshold_ptr[ : dataset_dim])
      for (size_t i = 0; i < dataset_size; i++) {
        for (uint32_t j = 0; j < dataset_dim; j++) {
          threshold_ptr[j] += dataset.data_handle()[i * dataset_dim + j];
        }
      }
      for (uint32_t j = 0; j < dataset_dim; j++) {
        threshold_ptr[j] /= dataset_size;
      }
    } else {
      // Use f32 array to compute the mean since omp reduction does not support f16
      auto threshold_vec_f32 = raft::make_host_vector<float, int64_t>(dataset_dim);
      auto threshold_f32_ptr = threshold_vec_f32.data_handle();
#pragma omp parallel for reduction(+ : threshold_f32_ptr[ : dataset_dim])
      for (size_t i = 0; i < dataset_size; i++) {
        for (uint32_t j = 0; j < dataset_dim; j++) {
          threshold_f32_ptr[j] += static_cast<float>(dataset.data_handle()[i * dataset_dim + j]);
        }
      }
      for (uint32_t j = 0; j < dataset_dim; j++) {
        threshold_ptr[j] = static_cast<half>(threshold_f32_ptr[j] / dataset_size);
      }
    }
  }

  if (params.threshold ==
      cuvs::preprocessing::quantize::binary::set_bit_threshold::sampling_median) {
    // Make the number of samples odd so that the median is calculated by only sort and memcpy
    const size_t num_sampls =
      std::max(
        raft::div_rounding_up_safe(static_cast<size_t>(dataset_size * params.sampling_ratio), 2lu) *
          2lu,
        2lu) -
      1;

    // Calculate stride
    size_t stride_prime_list[] = {611323lu, 611333lu, 611389lu, 611393};
    uint32_t prime_i           = 0;
    while (dataset_size % stride_prime_list[prime_i] == 0) {
      prime_i++;
    }
    const auto stride = stride_prime_list[prime_i];

    // Transposed
    auto sampled_dataset = raft::make_host_matrix<T, int64_t>(dataset_dim, num_sampls);
#pragma omp parallel for
    for (size_t out_i = 0; out_i < num_sampls; out_i++) {
      const auto in_i = (out_i * stride) % dataset_size;
      for (uint32_t j = 0; j < dataset_dim; j++) {
        sampled_dataset.data_handle()[j * num_sampls + out_i] =
          dataset.data_handle()[in_i * dataset_dim + j];
      }
    }

#pragma omp parallel for
    for (uint32_t j = 0; j < dataset_dim; j++) {
      auto start_ptr = sampled_dataset.data_handle() + j * num_sampls;
      std::sort(start_ptr, start_ptr + num_sampls);
      threshold_ptr[j] = start_ptr[(num_sampls - 1) / 2];
    }
  }

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < dataset_size; ++i) {
    for (uint32_t out_j = 0; out_j < minimul_out_dim; ++out_j) {
      QuantI pack = 0;
      for (uint32_t pack_j = 0; pack_j < bits_per_pack; ++pack_j) {
        const uint32_t in_j = out_j * bits_per_pack + pack_j;
        if (in_j < dataset_dim) {
          if (threshold_ptr == nullptr) {
            if (is_positive(dataset(i, in_j))) { pack |= (1u << pack_j); }
          } else {
            if (is_positive(dataset(i, in_j) - threshold_ptr[in_j])) { pack |= (1u << pack_j); }
          }
        }
      }
      out(i, out_j) = pack;
    }
  }
}
}  // namespace cuvs::preprocessing::quantize::detail
