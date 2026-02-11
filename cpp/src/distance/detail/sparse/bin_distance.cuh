/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.hpp"
#include "ip_distance.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <climits>

#include <nvfunctional>

namespace cuvs::distance::detail::sparse {
// @TODO: Move this into sparse prims (coo_norm)
template <typename ValueIdx, typename ValueT>  // NOLINT(readability-identifier-naming)
RAFT_KERNEL compute_binary_row_norm_kernel(ValueT* out,
                                           const ValueIdx* __restrict__ coo_rows,
                                           const ValueT* __restrict__ data,
                                           ValueIdx nnz)
{
  ValueIdx i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nnz) {
    // We do conditional here only because it's
    // possible there could be some stray zeros in
    // the sparse structure and removing them would be
    // more expensive.
    atomicAdd(&out[coo_rows[i]], data[i] == 1.0);
  }
}

template <typename ValueIdx,
          typename ValueT,
          typename ExpansionF>  // NOLINT(readability-identifier-naming)
RAFT_KERNEL compute_binary_warp_kernel(ValueT* __restrict__ C,
                                       const ValueT* __restrict__ q_norms,
                                       const ValueT* __restrict__ r_norms,
                                       ValueIdx n_rows,
                                       ValueIdx n_cols,
                                       ExpansionF expansion_func)
{
  std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  ValueIdx i      = tid / n_cols;
  ValueIdx j      = tid % n_cols;

  if (i >= n_rows || j >= n_cols) return;

  ValueT q_norm             = q_norms[i];
  ValueT r_norm             = r_norms[j];
  ValueT dot                = C[(size_t)i * n_cols + j];
  C[(size_t)i * n_cols + j] = expansion_func(dot, q_norm, r_norm);
}

template <typename ValueIdx,
          typename ValueT,
          typename ExpansionF,
          int tpb = 1024>  // NOLINT(readability-identifier-naming)
void compute_binary(ValueT* C,
                    const ValueT* q_norms,
                    const ValueT* r_norms,
                    ValueIdx n_rows,
                    ValueIdx n_cols,
                    ExpansionF expansion_func,
                    cudaStream_t stream)
{
  int blocks = raft::ceildiv<size_t>((size_t)n_rows * n_cols, tpb);
  compute_binary_warp_kernel<<<blocks, tpb, 0, stream>>>(
    C, q_norms, r_norms, n_rows, n_cols, expansion_func);
}

template <typename ValueIdx,
          typename ValueT,
          typename ExpansionF,
          int tpb = 1024>  // NOLINT(readability-identifier-naming)
void compute_bin_distance(ValueT* out,
                          const ValueIdx* Q_coo_rows,
                          const ValueT* Q_data,
                          ValueIdx Q_nnz,
                          const ValueIdx* R_coo_rows,
                          const ValueT* R_data,
                          ValueIdx R_nnz,
                          ValueIdx m,
                          ValueIdx n,
                          cudaStream_t stream,
                          ExpansionF expansion_func)
{
  rmm::device_uvector<ValueT> q_norms(m, stream);
  rmm::device_uvector<ValueT> r_norms(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(q_norms.data(), 0, q_norms.size() * sizeof(ValueT)));
  RAFT_CUDA_TRY(cudaMemsetAsync(r_norms.data(), 0, r_norms.size() * sizeof(ValueT)));

  compute_binary_row_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    q_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_binary_row_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    r_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_binary(out, q_norms.data(), r_norms.data(), m, n, expansion_func, stream);
}

/**
 * Jaccard distance using the expanded form:
 * 1 - (sum(x_k * y_k) / ((sum(x_k) + sum(y_k)) - sum(x_k * y_k))
 */
template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class jaccard_expanded_distances_t : public distances_t<ValueT> {
 public:
  explicit jaccard_expanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config),
      workspace_(0, raft::resource::get_cuda_stream(config.handle)),
      ip_dists_(config)
  {
  }

  void compute(ValueT* out_dists)
  {
    ip_dists_.compute(out_dists);

    ValueIdx* b_indices = ip_dists_.b_rows_coo();
    ValueT* b_data      = ip_dists_.b_data_coo();

    rmm::device_uvector<ValueIdx> search_coo_rows(config_->a_nnz,
                                                  raft::resource::get_cuda_stream(config_->handle));
    raft::sparse::convert::csr_to_coo(config_->a_indptr,
                                      config_->a_nrows,
                                      search_coo_rows.data(),
                                      config_->a_nnz,
                                      raft::resource::get_cuda_stream(config_->handle));

    compute_bin_distance(
      out_dists,
      search_coo_rows.data(),
      config_->a_data,
      config_->a_nnz,
      b_indices,
      b_data,
      config_->b_nnz,
      config_->a_nrows,
      config_->b_nrows,
      raft::resource::get_cuda_stream(config_->handle),
      [] __device__ __host__(ValueT dot, ValueT q_norm, ValueT r_norm) -> ValueT {
        ValueT q_r_union = q_norm + r_norm;
        ValueT denom     = q_r_union - dot;

        ValueT jacc = ((denom != 0) * dot) / ((denom == 0) + denom);

        // flip the similarity when both rows are 0
        bool both_empty = q_r_union == 0;
        return 1 - ((!both_empty * jacc) + both_empty);
      });
  }

  ~jaccard_expanded_distances_t() = default;

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
  rmm::device_uvector<char> workspace_;
  ip_distances_t<ValueIdx, ValueT> ip_dists_;
};

/**
 * Dice distance using the expanded form:
 * 1 - ((2 * sum(x_k * y_k)) / (sum(x_k) + sum(y_k)))
 */
template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class dice_expanded_distances_t : public distances_t<ValueT> {
 public:
  explicit dice_expanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config),
      workspace_(0, raft::resource::get_cuda_stream(config.handle)),
      ip_dists_(config)
  {
  }

  void compute(ValueT* out_dists)
  {
    ip_dists_.compute(out_dists);

    ValueIdx* b_indices = ip_dists_.b_rows_coo();
    ValueT* b_data      = ip_dists_.b_data_coo();

    rmm::device_uvector<ValueIdx> search_coo_rows(config_->a_nnz,
                                                  raft::resource::get_cuda_stream(config_->handle));
    raft::sparse::convert::csr_to_coo(config_->a_indptr,
                                      config_->a_nrows,
                                      search_coo_rows.data(),
                                      config_->a_nnz,
                                      raft::resource::get_cuda_stream(config_->handle));

    compute_bin_distance(
      out_dists,
      search_coo_rows.data(),
      config_->a_data,
      config_->a_nnz,
      b_indices,
      b_data,
      config_->b_nnz,
      config_->a_nrows,
      config_->b_nrows,
      raft::resource::get_cuda_stream(config_->handle),
      [] __device__ __host__(ValueT dot, ValueT q_norm, ValueT r_norm) -> ValueT {
        ValueT q_r_union = q_norm + r_norm;
        ValueT dice      = (2 * dot) / q_r_union;
        bool both_empty  = q_r_union == 0;
        return 1 - ((!both_empty * dice) + both_empty);
      });
  }

  ~dice_expanded_distances_t() = default;

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
  rmm::device_uvector<char> workspace_;
  ip_distances_t<ValueIdx, ValueT> ip_dists_;
};

}  // namespace cuvs::distance::detail::sparse
   // END namespace detail
   // END namespace distance
   // END namespace cuvs
