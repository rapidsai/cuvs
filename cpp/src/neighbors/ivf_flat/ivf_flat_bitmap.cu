/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_flat.hpp>

#include <raft/core/bitmap.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace cuvs::neighbors::ivf_flat {
namespace detail {

__global__ void repack_bitmap_kernel(raft::core::bitmap_view<const uint32_t, int64_t> input,
                                     uint8_t* output,
                                     int64_t output_size,
                                     int64_t encoded_dim)
{
  for (auto offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < output_size;
       offset += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto row          = offset / encoded_dim;
    auto output_col   = offset % encoded_dim;
    auto input_col    = output_col * 8;
    uint8_t byte      = 0;
    auto remaining    = input.get_n_cols() - input_col;
    auto bits_to_copy = min(int64_t{8}, remaining);
    for (int64_t bit = 0; bit < bits_to_copy; ++bit) {
      if (input.test(row, input_col + bit)) { byte |= uint8_t{1} << bit; }
    }
    output[offset] = byte;
  }
}

auto repack_bitmap(raft::resources const& handle,
                   cuvs::device_bitmap_dataset_view<int64_t> dataset)
  -> raft::device_matrix<uint8_t, int64_t, raft::row_major>
{
  auto input       = dataset.data_const_view();
  auto n_rows      = input.get_n_rows();
  auto logical_dim = input.get_n_cols();
  RAFT_EXPECTS(n_rows >= 0, "Bitmap dataset row count must be non-negative.");
  RAFT_EXPECTS(logical_dim > 0, "Bitmap dataset dimension must be positive.");
  RAFT_EXPECTS(logical_dim <= std::numeric_limits<uint32_t>::max(),
               "Bitmap dataset dimension exceeds the supported range.");

  auto encoded_dim = (logical_dim - 1) / 8 + 1;
  RAFT_EXPECTS(n_rows <= std::numeric_limits<int64_t>::max() / encoded_dim,
               "Bitmap dataset size exceeds the supported range.");
  auto output = raft::make_device_matrix<uint8_t, int64_t>(handle, n_rows, encoded_dim);

  auto output_size = n_rows * encoded_dim;
  if (output_size > 0) {
    constexpr int threads_per_block = 256;
    auto n_blocks                   = static_cast<int>(
      std::min<int64_t>((output_size + threads_per_block - 1) / threads_per_block, 65535));
    repack_bitmap_kernel<<<n_blocks,
                           threads_per_block,
                           0,
                           raft::resource::get_cuda_stream(handle).get()>>>(
      input, output.data_handle(), output_size, encoded_dim);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  return output;
}

void check_compatible(const index<uint8_t, int64_t>& idx,
                      cuvs::device_bitmap_dataset_view<int64_t> dataset)
{
  RAFT_EXPECTS(idx.binary_index(), "Bitmap datasets require a BitwiseHamming index.");
  RAFT_EXPECTS(dataset.data_const_view().get_n_cols() == idx.dim(),
               "Bitmap dataset dimension must match the index logical dimension.");
}

}  // namespace detail

auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           cuvs::device_bitmap_dataset_view<int64_t> dataset)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>
{
  RAFT_EXPECTS(index_params.metric == cuvs::distance::DistanceType::BitwiseHamming,
               "Bitmap datasets require the BitwiseHamming metric.");
  auto packed = detail::repack_bitmap(handle, dataset);
  auto idx    = build(handle, index_params, raft::make_const_mdspan(packed.view()));
  detail::index_access::set_dim(idx, dataset.dim());
  return idx;
}

auto extend(raft::resources const& handle,
            cuvs::device_bitmap_dataset_view<int64_t> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>
{
  detail::check_compatible(idx, new_vectors);
  auto packed = detail::repack_bitmap(handle, new_vectors);
  return extend(handle, raft::make_const_mdspan(packed.view()), new_indices, idx);
}

void extend(raft::resources const& handle,
            cuvs::device_bitmap_dataset_view<int64_t> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* idx)
{
  RAFT_EXPECTS(idx != nullptr, "Index pointer must not be null.");
  detail::check_compatible(*idx, new_vectors);
  auto packed = detail::repack_bitmap(handle, new_vectors);
  extend(handle, raft::make_const_mdspan(packed.view()), new_indices, idx);
}

void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index,
            cuvs::device_bitmap_dataset_view<int64_t> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  detail::check_compatible(index, queries);
  auto packed = detail::repack_bitmap(handle, queries);
  search(handle,
         params,
         index,
         raft::make_const_mdspan(packed.view()),
         neighbors,
         distances,
         sample_filter);
}

}  // namespace cuvs::neighbors::ivf_flat
