/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_binary.hpp>

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <limits>

namespace cuvs::neighbors::ivf_binary {
namespace {

uint32_t encoded_length(uint32_t dim)
{
  RAFT_EXPECTS(dim > 0, "The dimension of a binary index must be positive");
  RAFT_EXPECTS(dim <= std::numeric_limits<uint32_t>::max() / 8,
               "The binary dimension is too large for the per-bit center statistics");
  return (dim + 7) / 8;
}

template <typename T>
void zero_fill(raft::resources const& res, T* ptr, size_t n)
{
  if (n != 0) {
    RAFT_CUDA_TRY(
      cudaMemsetAsync(ptr, 0, n * sizeof(T), raft::resource::get_cuda_stream(res).get()));
  }
}

}  // namespace

index::index(raft::resources const& res)
  : dim_{0},
    adaptive_centers_{false},
    quantizer_{res},
    centers_{raft::make_device_matrix<uint8_t, int64_t>(res, 0, 0)},
    center_counts_{raft::make_device_matrix<uint32_t, int64_t>(res, 0, 0)},
    lists_{res}
{
}

index::index(raft::resources const& res, const index_params& params, uint32_t dim)
  : dim_{dim},
    adaptive_centers_{params.adaptive_centers},
    quantizer_{res},
    centers_{raft::make_device_matrix<uint8_t, int64_t>(res, params.n_lists, encoded_length(dim))},
    center_counts_{raft::make_device_matrix<uint32_t, int64_t>(
      res,
      params.adaptive_centers ? params.n_lists : 0,
      params.adaptive_centers ? int64_t{8} * encoded_length(dim) : 0)},
    lists_{res,
           cuvs::distance::DistanceType::BitwiseHamming,
           params.n_lists,
           false,
           params.conservative_memory_allocation,
           encoded_length(dim)}
{
  zero_fill(res, center_counts_.data_handle(), center_counts_.size());
  // The lists reserve floating-point centers that are not used by the binary index.
  zero_fill(res, lists_.centers().data_handle(), lists_.centers().size());
}

uint32_t index::dim() const noexcept { return dim_; }

uint32_t index::encoded_dim() const noexcept { return static_cast<uint32_t>(centers_.extent(1)); }

uint32_t index::n_lists() const noexcept { return lists_.n_lists(); }

int64_t index::size() const noexcept { return lists_.size(); }

bool index::adaptive_centers() const noexcept { return adaptive_centers_; }

bool index::conservative_memory_allocation() const noexcept
{
  return lists_.conservative_memory_allocation();
}

raft::device_vector_view<const uint32_t, uint32_t> index::list_sizes() const noexcept
{
  return lists_.list_sizes();
}

raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> index::centers() const noexcept
{
  return centers_.view();
}

raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> index::center_counts()
  const noexcept
{
  return center_counts_.view();
}

const preprocessing::quantize::binary::quantizer<float>& index::quantizer() const noexcept
{
  return quantizer_;
}

}  // namespace cuvs::neighbors::ivf_binary
