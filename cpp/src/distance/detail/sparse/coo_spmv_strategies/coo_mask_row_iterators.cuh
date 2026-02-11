/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../common.hpp"
#include "../utils.cuh"

#include <raft/util/cuda_dev_essentials.cuh>  // raft::ceildiv

#include <rmm/device_uvector.hpp>

#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cuvs::distance::detail::sparse {

template <typename ValueIdx>
class mask_row_it {
 public:
  mask_row_it(const ValueIdx* full_indptr_,
              const ValueIdx& n_rows_,
              ValueIdx* mask_row_idx_ = nullptr)
    : full_indptr(full_indptr_), mask_row_idx(mask_row_idx_), n_rows(n_rows_)
  {
  }

  __device__ inline auto get_row_idx(const int& n_blocks_nnz_b) -> ValueIdx
  {
    if (mask_row_idx != nullptr) {
      return mask_row_idx[blockIdx.x / n_blocks_nnz_b];
    } else {
      return blockIdx.x / n_blocks_nnz_b;
    }
  }

  __device__ inline void get_row_offsets(const ValueIdx& row_idx,
                                         ValueIdx& start_offset,
                                         ValueIdx& stop_offset,
                                         const ValueIdx& n_blocks_nnz_b,
                                         bool& first_a_chunk,
                                         bool& last_a_chunk)
  {
    start_offset = full_indptr[row_idx];
    stop_offset  = full_indptr[row_idx + 1] - 1;
  }

  __device__ constexpr inline void get_indices_boundary(const ValueIdx* indices,
                                                        ValueIdx& indices_len,
                                                        ValueIdx& start_offset,
                                                        ValueIdx& stop_offset,
                                                        ValueIdx& start_index,
                                                        ValueIdx& stop_index,
                                                        bool& first_a_chunk,
                                                        bool& last_a_chunk)
  {
    // do nothing;
  }

  __device__ constexpr inline auto check_indices_bounds(ValueIdx& start_index_a,
                                                        ValueIdx& stop_index_a,
                                                        ValueIdx& index_b) -> bool
  {
    return true;
  }

  const ValueIdx *full_indptr, &n_rows;
  ValueIdx* mask_row_idx;
};

template <typename ValueIdx>
RAFT_KERNEL fill_chunk_indices_kernel(ValueIdx* n_chunks_per_row,
                                      ValueIdx* chunk_indices,
                                      ValueIdx n_rows)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_rows) {
    auto start = n_chunks_per_row[tid];
    auto end   = n_chunks_per_row[tid + 1];

#pragma unroll
    for (int i = start; i < end; i++) {
      chunk_indices[i] = tid;
    }
  }
}

template <typename ValueIdx>
class chunked_mask_row_it : public mask_row_it<ValueIdx> {
 public:
  chunked_mask_row_it(const ValueIdx* full_indptr_,
                      const ValueIdx& n_rows_,
                      ValueIdx* mask_row_idx_,
                      int row_chunk_size_,
                      const ValueIdx* n_chunks_per_row_,
                      const ValueIdx* chunk_indices_,
                      const cudaStream_t stream_)
    : mask_row_it<ValueIdx>(full_indptr_, n_rows_, mask_row_idx_),
      row_chunk_size(row_chunk_size_),
      n_chunks_per_row(n_chunks_per_row_),
      chunk_indices(chunk_indices_),
      stream(stream_)
  {
  }

  static void init(const ValueIdx* indptr,
                   const ValueIdx* mask_row_idx,
                   const ValueIdx& n_rows,
                   const int row_chunk_size,
                   rmm::device_uvector<ValueIdx>& n_chunks_per_row,
                   rmm::device_uvector<ValueIdx>& chunk_indices,
                   cudaStream_t stream)
  {
    auto policy = rmm::exec_policy(stream);

    constexpr ValueIdx kFirstElement = 0;
    n_chunks_per_row.set_element_async(0, kFirstElement, stream);
    n_chunks_per_row_functor chunk_functor(indptr, row_chunk_size);
    thrust::transform(
      policy, mask_row_idx, mask_row_idx + n_rows, n_chunks_per_row.begin() + 1, chunk_functor);

    thrust::inclusive_scan(
      policy, n_chunks_per_row.begin() + 1, n_chunks_per_row.end(), n_chunks_per_row.begin() + 1);

    raft::update_host(&total_row_blocks, n_chunks_per_row.data() + n_rows, 1, stream);

    fill_chunk_indices(n_rows, n_chunks_per_row, chunk_indices, stream);
  }

  __device__ inline auto get_row_idx(const int& n_blocks_nnz_b) -> ValueIdx
  {
    return this->mask_row_idx[chunk_indices[blockIdx.x / n_blocks_nnz_b]];
  }

  __device__ inline void get_row_offsets(const ValueIdx& row_idx,
                                         ValueIdx& start_offset,
                                         ValueIdx& stop_offset,
                                         const int& n_blocks_nnz_b,
                                         bool& first_a_chunk,
                                         bool& last_a_chunk)
  {
    auto chunk_index    = blockIdx.x / n_blocks_nnz_b;
    auto chunk_val      = chunk_indices[chunk_index];
    auto prev_n_chunks  = n_chunks_per_row[chunk_val];
    auto relative_chunk = chunk_index - prev_n_chunks;
    first_a_chunk       = relative_chunk == 0;

    start_offset = this->full_indptr[row_idx] + relative_chunk * row_chunk_size;
    stop_offset  = start_offset + row_chunk_size;

    auto final_stop_offset = this->full_indptr[row_idx + 1];

    last_a_chunk = stop_offset >= final_stop_offset;
    stop_offset  = last_a_chunk ? final_stop_offset - 1 : stop_offset - 1;
  }

  __device__ inline void get_indices_boundary(const ValueIdx* indices,
                                              ValueIdx& row_idx,
                                              ValueIdx& start_offset,
                                              ValueIdx& stop_offset,
                                              ValueIdx& start_index,
                                              ValueIdx& stop_index,
                                              bool& first_a_chunk,
                                              bool& last_a_chunk)
  {
    start_index = first_a_chunk ? start_index : indices[start_offset - 1] + 1;
    stop_index  = last_a_chunk ? stop_index : indices[stop_offset];
  }

  __device__ inline auto check_indices_bounds(ValueIdx& start_index_a,
                                              ValueIdx& stop_index_a,
                                              ValueIdx& index_b) -> bool
  {
    return (index_b >= start_index_a && index_b <= stop_index_a);
  }

  inline static ValueIdx total_row_blocks = 0;
  const cudaStream_t stream;
  const ValueIdx *n_chunks_per_row, *chunk_indices;
  ValueIdx row_chunk_size;

  struct n_chunks_per_row_functor {
   public:
    n_chunks_per_row_functor(const ValueIdx* indptr_, ValueIdx row_chunk_size_)
      : indptr(indptr_), row_chunk_size(row_chunk_size_)
    {
    }

    __host__ __device__ auto operator()(const ValueIdx& i) -> ValueIdx
    {
      auto degree = indptr[i + 1] - indptr[i];
      return raft::ceildiv(degree, static_cast<ValueIdx>(row_chunk_size));
    }

    const ValueIdx* indptr;
    ValueIdx row_chunk_size;
  };

 private:
  static void fill_chunk_indices(const ValueIdx& n_rows,
                                 rmm::device_uvector<ValueIdx>& n_chunks_per_row,
                                 rmm::device_uvector<ValueIdx>& chunk_indices,
                                 cudaStream_t stream)
  {
    auto n_threads = std::min(n_rows, 256);
    auto n_blocks  = raft::ceildiv(n_rows, static_cast<ValueIdx>(n_threads));

    chunk_indices.resize(total_row_blocks, stream);

    fill_chunk_indices_kernel<ValueIdx>
      <<<n_blocks, n_threads, 0, stream>>>(n_chunks_per_row.data(), chunk_indices.data(), n_rows);
  }
};

}  // namespace cuvs::distance::detail::sparse
