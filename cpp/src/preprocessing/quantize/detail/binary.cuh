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

#include <cuvs/preprocessing/quantize/scalar.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/system/omp/execution_policy.h>

namespace cuvs::preprocessing::quantize::detail {

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
        const auto v = in_ptr[vector_id * ldi + j];
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
               "the input dataset size (%u) but is %u passed",
               dataset_size,
               out_dataset_size);

  constexpr uint32_t warp_size    = 32;
  constexpr uint32_t block_size   = 256;
  constexpr uint32_t vecs_per_cta = block_size / warp_size;
  const auto grid_size =
    raft::div_rounding_up_safe(dataset_size, static_cast<size_t>(vecs_per_cta));

  binary_quantization_kernel<T, block_size>
    <<<grid_size, block_size, 0, stream>>>(dataset.data_handle(),
                                           dataset.stride(0),
                                           dataset_size,
                                           dataset_dim,
                                           out.data_handle(),
                                           out.stride(0));
}

template <typename T, typename QuantI = uint8_t>
void transform(raft::resources const& res,
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
               "the input dataset size (%u) but is %u passed",
               dataset_size,
               out_dataset_size);

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < dataset_size; ++i) {
    for (uint32_t out_j = 0; out_j < minimul_out_dim; ++out_j) {
      QuantI pack = 0;
      for (uint32_t pack_j = 0; pack_j < bits_per_pack; ++pack_j) {
        const uint32_t in_j = out_j * bits_per_pack + pack_j;
        if (in_j < dataset_dim) {
          if (is_positive(dataset(i, in_j))) { pack |= (1u << pack_j); }
        }
      }
      out(i, out_j) = pack;
    }
  }
}
}  // namespace cuvs::preprocessing::quantize::detail
