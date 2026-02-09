/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "base_strategy.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

namespace cuvs {
namespace distance {
namespace detail {
namespace sparse {

template <typename value_idx, typename value_t, int tpb>
class dense_smem_strategy : public coo_spmv_strategy<value_idx, value_t, tpb> {
 public:
  using map_type = value_t*;

  explicit dense_smem_strategy(const distances_config_t<value_idx, value_t>& config_)
    : coo_spmv_strategy<value_idx, value_t, tpb>(config_)
  {
  }

  inline static auto smem_per_block(int n_cols) -> int
  {
    return (n_cols * sizeof(value_t)) + ((1024 / raft::warp_size()) * sizeof(value_t));
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch(value_t* out_dists,
                value_idx* coo_rows_b,
                product_f product_func,
                accum_f accum_func,
                write_f write_func,
                int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * 1024);
    auto n_blocks         = this->config.a_nrows * n_blocks_per_row;

    mask_row_it<value_idx> a_indptr(this->config.a_indptr, this->config.a_nrows);

    this->_dispatch_base(*this,
                         this->config.b_ncols,
                         a_indptr,
                         out_dists,
                         coo_rows_b,
                         product_func,
                         accum_func,
                         write_func,
                         chunk_size,
                         n_blocks,
                         n_blocks_per_row);
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch_rev(value_t* out_dists,
                    value_idx* coo_rows_a,
                    product_f product_func,
                    accum_f accum_func,
                    write_f write_func,
                    int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * 1024);
    auto n_blocks         = this->config.b_nrows * n_blocks_per_row;

    mask_row_it<value_idx> b_indptr(this->config.b_indptr, this->config.b_nrows);

    this->_dispatch_base_rev(*this,
                             this->config.a_ncols,
                             b_indptr,
                             out_dists,
                             coo_rows_a,
                             product_func,
                             accum_func,
                             write_func,
                             chunk_size,
                             n_blocks,
                             n_blocks_per_row);
  }

  __device__ inline auto init_map(void* storage, const value_idx& cache_size) -> map_type
  {
    auto cache = static_cast<value_t*>(storage);
    for (int k = threadIdx.x; k < cache_size; k += blockDim.x) {
      cache[k] = 0.0;
    }
    return cache;
  }

  __device__ inline void insert(map_type& cache, const value_idx& key, const value_t& value)
  {
    cache[key] = value;
  }

  __device__ inline auto find(map_type& cache, const value_idx& key) -> value_t
  {
    return cache[key];
  }
};

}  // namespace sparse
}  // namespace detail
}  // namespace distance
}  // namespace cuvs
