/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.hpp"
#include "coo_spmv.cuh"

#include <raft/core/operators.cuh>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/linalg/transpose.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <climits>

#include <nvfunctional>

namespace cuvs::distance::detail::sparse {

template <typename ValueIdx, typename ValueT>  // NOLINT(readability-identifier-naming)
class ip_distances_t : public distances_t<ValueT> {
 public:
  /**
   * Computes simple sparse inner product distances as sum(x_y * y_k)
   * @param[in] config specifies inputs, outputs, and sizes
   */
  explicit ip_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config), coo_rows_b_(config.b_nnz, raft::resource::get_cuda_stream(config.handle))
  {
    raft::sparse::convert::csr_to_coo(config_->b_indptr,
                                      config_->b_nrows,
                                      coo_rows_b_.data(),
                                      config_->b_nnz,
                                      raft::resource::get_cuda_stream(config_->handle));
  }

  /**
   * Performs pairwise distance computation and computes output distances
   * @param out_distances dense output matrix (size a_nrows * b_nrows)
   */
  void compute(ValueT* out_distances)
  {
    /**
     * Compute pairwise distances and return dense matrix in row-major format
     */
    balanced_coo_pairwise_generalized_spmv<ValueIdx, ValueT>(out_distances,
                                                             *config_,
                                                             coo_rows_b_.data(),
                                                             raft::mul_op(),
                                                             raft::add_op(),
                                                             raft::atomic_add_op());
  }

  auto b_rows_coo() -> ValueIdx* { return coo_rows_b_.data(); }

  auto b_data_coo() -> ValueT* { return config_->b_data; }

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
  rmm::device_uvector<ValueIdx> coo_rows_b_;
};

}  // namespace cuvs::distance::detail::sparse
   // END namespace detail
   // END namespace distance
   // END namespace cuvs
