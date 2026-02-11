/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../common.hpp"
#include "../coo_spmv_kernel.cuh"
#include "../utils.cuh"
#include "coo_mask_row_iterators.cuh"

#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_uvector.hpp>

namespace cuvs::distance::detail::sparse {

template <typename ValueIdx, typename value_t, int tpb>  // NOLINT(readability-identifier-naming)
class coo_spmv_strategy {
 public:
  explicit coo_spmv_strategy(const distances_config_t<ValueIdx, value_t>& config_)
    : config_(config_)
  {
    smem_ = raft::getSharedMemPerBlock();
  }

  template <typename StrategyT,
            typename IndptrIt,
            typename ProductF,
            typename AccumF,
            typename WriteF>
  void dispatch_base(StrategyT& strategy,
                     int smem_dim,
                     IndptrIt& a_indptr,
                     value_t* out_dists,
                     ValueIdx* coo_rows_b,
                     ProductF product_func,
                     AccumF accum_func,
                     WriteF write_func,
                     int chunk_size,
                     int n_blocks,
                     int n_blocks_per_row)
  {
    RAFT_CUDA_TRY(cudaFuncSetCacheConfig(balanced_coo_generalized_spmv_kernel<StrategyT,
                                                                              IndptrIt,
                                                                              ValueIdx,
                                                                              value_t,
                                                                              false,
                                                                              tpb,
                                                                              ProductF,
                                                                              AccumF,
                                                                              WriteF>,
                                         cudaFuncCachePreferShared));

    balanced_coo_generalized_spmv_kernel<StrategyT, IndptrIt, ValueIdx, value_t, false, tpb>
      <<<n_blocks, tpb, smem_, raft::resource::get_cuda_stream(config_.handle)>>>(strategy,
                                                                                  a_indptr,
                                                                                  config_.a_indices,
                                                                                  config_.a_data,
                                                                                  config_.a_nnz,
                                                                                  coo_rows_b,
                                                                                  config_.b_indices,
                                                                                  config_.b_data,
                                                                                  config_.a_nrows,
                                                                                  config_.b_nrows,
                                                                                  smem_dim,
                                                                                  config_.b_nnz,
                                                                                  out_dists,
                                                                                  n_blocks_per_row,
                                                                                  chunk_size,
                                                                                  config_.b_ncols,
                                                                                  product_func,
                                                                                  accum_func,
                                                                                  write_func);
  }

  template <typename StrategyT,
            typename IndptrIt,
            typename ProductF,
            typename AccumF,
            typename WriteF>
  void dispatch_base_rev(StrategyT& strategy,
                         int smem_dim,
                         IndptrIt& b_indptr,
                         value_t* out_dists,
                         ValueIdx* coo_rows_a,
                         ProductF product_func,
                         AccumF accum_func,
                         WriteF write_func,
                         int chunk_size,
                         int n_blocks,
                         int n_blocks_per_row)
  {
    RAFT_CUDA_TRY(cudaFuncSetCacheConfig(balanced_coo_generalized_spmv_kernel<StrategyT,
                                                                              IndptrIt,
                                                                              ValueIdx,
                                                                              value_t,
                                                                              true,
                                                                              tpb,
                                                                              ProductF,
                                                                              AccumF,
                                                                              WriteF>,
                                         cudaFuncCachePreferShared));

    balanced_coo_generalized_spmv_kernel<StrategyT, IndptrIt, ValueIdx, value_t, true, tpb>
      <<<n_blocks, tpb, smem_, raft::resource::get_cuda_stream(config_.handle)>>>(strategy,
                                                                                  b_indptr,
                                                                                  config_.b_indices,
                                                                                  config_.b_data,
                                                                                  config_.b_nnz,
                                                                                  coo_rows_a,
                                                                                  config_.a_indices,
                                                                                  config_.a_data,
                                                                                  config_.b_nrows,
                                                                                  config_.a_nrows,
                                                                                  smem_dim,
                                                                                  config_.a_nnz,
                                                                                  out_dists,
                                                                                  n_blocks_per_row,
                                                                                  chunk_size,
                                                                                  config_.a_ncols,
                                                                                  product_func,
                                                                                  accum_func,
                                                                                  write_func);
  }

 protected:
  int smem_;
  const distances_config_t<ValueIdx, value_t>& config_;
};

}  // namespace cuvs::distance::detail::sparse
