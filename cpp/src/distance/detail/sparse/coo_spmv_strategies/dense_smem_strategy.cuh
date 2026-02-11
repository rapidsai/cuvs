/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "base_strategy.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

namespace cuvs::distance::detail::sparse {

template <typename ValueIdx, typename ValueT, int tpb>
class dense_smem_strategy : public coo_spmv_strategy<ValueIdx, ValueT, tpb> {
 public:
  using map_type = ValueT*;

  explicit dense_smem_strategy(const distances_config_t<ValueIdx, ValueT>& config_)
    : coo_spmv_strategy<ValueIdx, ValueT, tpb>(config_)
  {
  }

  inline static auto smem_per_block(int n_cols) -> int
  {
    return (n_cols * sizeof(ValueT)) + ((1024 / raft::warp_size()) * sizeof(ValueT));
  }

  template <typename ProductF, typename AccumF, typename WriteF>
  void dispatch(ValueT* out_dists,
                ValueIdx* coo_rows_b,
                ProductF product_func,
                AccumF accum_func,
                WriteF write_func,
                int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config_.b_nnz, chunk_size * 1024);
    auto n_blocks         = this->config_.a_nrows * n_blocks_per_row;

    mask_row_it<ValueIdx> a_indptr(this->config_.a_indptr, this->config_.a_nrows);

    this->dispatch_base(*this,
                        this->config_.b_ncols,
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

  template <typename ProductF, typename AccumF, typename WriteF>
  void dispatch_rev(ValueT* out_dists,
                    ValueIdx* coo_rows_a,
                    ProductF product_func,
                    AccumF accum_func,
                    WriteF write_func,
                    int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config_.a_nnz, chunk_size * 1024);
    auto n_blocks         = this->config_.b_nrows * n_blocks_per_row;

    mask_row_it<ValueIdx> b_indptr(this->config_.b_indptr, this->config_.b_nrows);

    this->dispatch_base_rev(*this,
                            this->config_.a_ncols,
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

  __device__ inline auto init_map(void* storage, const ValueIdx& cache_size) -> map_type
  {
    auto cache = static_cast<ValueT*>(storage);
    for (int k = threadIdx.x; k < cache_size; k += blockDim.x) {
      cache[k] = 0.0;
    }
    return cache;
  }

  __device__ inline void insert(map_type& cache, const ValueIdx& key, const ValueT& value)
  {
    cache[key] = value;
  }

  __device__ inline auto find(map_type& cache, const ValueIdx& key) -> ValueT { return cache[key]; }
};

}  // namespace cuvs::distance::detail::sparse
