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

#include "common.hpp"
#include "ip_distance.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <limits.h>

#include <nvfunctional>

namespace cuvs {
namespace distance {
namespace detail {
namespace sparse {
// @TODO: Move this into sparse prims (coo_norm)
template <typename value_idx, typename value_t>
RAFT_KERNEL compute_binary_row_norm_kernel(value_t* out,
                                           const value_idx* __restrict__ coo_rows,
                                           const value_t* __restrict__ data,
                                           value_idx nnz)
{
  value_idx i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nnz) {
    // We do conditional here only because it's
    // possible there could be some stray zeros in
    // the sparse structure and removing them would be
    // more expensive.
    atomicAdd(&out[coo_rows[i]], data[i] == 1.0);
  }
}

template <typename value_idx, typename value_t, typename expansion_f>
RAFT_KERNEL compute_binary_warp_kernel(value_t* __restrict__ C,
                                       const value_t* __restrict__ Q_norms,
                                       const value_t* __restrict__ R_norms,
                                       value_idx n_rows,
                                       value_idx n_cols,
                                       expansion_f expansion_func)
{
  std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  value_idx i     = tid / n_cols;
  value_idx j     = tid % n_cols;

  if (i >= n_rows || j >= n_cols) return;

  value_t q_norm            = Q_norms[i];
  value_t r_norm            = R_norms[j];
  value_t dot               = C[(size_t)i * n_cols + j];
  C[(size_t)i * n_cols + j] = expansion_func(dot, q_norm, r_norm);
}

template <typename value_idx, typename value_t, typename expansion_f, int tpb = 1024>
void compute_binary(value_t* C,
                    const value_t* Q_norms,
                    const value_t* R_norms,
                    value_idx n_rows,
                    value_idx n_cols,
                    expansion_f expansion_func,
                    cudaStream_t stream)
{
  int blocks = raft::ceildiv<size_t>((size_t)n_rows * n_cols, tpb);
  compute_binary_warp_kernel<<<blocks, tpb, 0, stream>>>(
    C, Q_norms, R_norms, n_rows, n_cols, expansion_func);
}

template <typename value_idx, typename value_t, typename expansion_f, int tpb = 1024>
void compute_bin_distance(value_t* out,
                          const value_idx* Q_coo_rows,
                          const value_t* Q_data,
                          value_idx Q_nnz,
                          const value_idx* R_coo_rows,
                          const value_t* R_data,
                          value_idx R_nnz,
                          value_idx m,
                          value_idx n,
                          cudaStream_t stream,
                          expansion_f expansion_func)
{
  rmm::device_uvector<value_t> Q_norms(m, stream);
  rmm::device_uvector<value_t> R_norms(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(Q_norms.data(), 0, Q_norms.size() * sizeof(value_t)));
  RAFT_CUDA_TRY(cudaMemsetAsync(R_norms.data(), 0, R_norms.size() * sizeof(value_t)));

  compute_binary_row_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    Q_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_binary_row_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    R_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_binary(out, Q_norms.data(), R_norms.data(), m, n, expansion_func, stream);
}

/**
 * Jaccard distance using the expanded form:
 * 1 - (sum(x_k * y_k) / ((sum(x_k) + sum(y_k)) - sum(x_k * y_k))
 */
template <typename value_idx = int, typename value_t = float>
class jaccard_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit jaccard_expanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config),
      workspace(0, raft::resource::get_cuda_stream(config.handle)),
      ip_dists(config)
  {
  }

  void compute(value_t* out_dists)
  {
    ip_dists.compute(out_dists);

    value_idx* b_indices = ip_dists.b_rows_coo();
    value_t* b_data      = ip_dists.b_data_coo();

    rmm::device_uvector<value_idx> search_coo_rows(
      config_->a_nnz, raft::resource::get_cuda_stream(config_->handle));
    raft::sparse::convert::csr_to_coo(config_->a_indptr,
                                      config_->a_nrows,
                                      search_coo_rows.data(),
                                      config_->a_nnz,
                                      raft::resource::get_cuda_stream(config_->handle));

    compute_bin_distance(out_dists,
                         search_coo_rows.data(),
                         config_->a_data,
                         config_->a_nnz,
                         b_indices,
                         b_data,
                         config_->b_nnz,
                         config_->a_nrows,
                         config_->b_nrows,
                         raft::resource::get_cuda_stream(config_->handle),
                         [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) {
                           value_t q_r_union = q_norm + r_norm;
                           value_t denom     = q_r_union - dot;

                           value_t jacc = ((denom != 0) * dot) / ((denom == 0) + denom);

                           // flip the similarity when both rows are 0
                           bool both_empty = q_r_union == 0;
                           return 1 - ((!both_empty * jacc) + both_empty);
                         });
  }

  ~jaccard_expanded_distances_t() = default;

 private:
  const distances_config_t<value_idx, value_t>* config_;
  rmm::device_uvector<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

/**
 * Dice distance using the expanded form:
 * 1 - ((2 * sum(x_k * y_k)) / (sum(x_k) + sum(y_k)))
 */
template <typename value_idx = int, typename value_t = float>
class dice_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit dice_expanded_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config),
      workspace(0, raft::resource::get_cuda_stream(config.handle)),
      ip_dists(config)
  {
  }

  void compute(value_t* out_dists)
  {
    ip_dists.compute(out_dists);

    value_idx* b_indices = ip_dists.b_rows_coo();
    value_t* b_data      = ip_dists.b_data_coo();

    rmm::device_uvector<value_idx> search_coo_rows(
      config_->a_nnz, raft::resource::get_cuda_stream(config_->handle));
    raft::sparse::convert::csr_to_coo(config_->a_indptr,
                                      config_->a_nrows,
                                      search_coo_rows.data(),
                                      config_->a_nnz,
                                      raft::resource::get_cuda_stream(config_->handle));

    compute_bin_distance(out_dists,
                         search_coo_rows.data(),
                         config_->a_data,
                         config_->a_nnz,
                         b_indices,
                         b_data,
                         config_->b_nnz,
                         config_->a_nrows,
                         config_->b_nrows,
                         raft::resource::get_cuda_stream(config_->handle),
                         [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) {
                           value_t q_r_union = q_norm + r_norm;
                           value_t dice      = (2 * dot) / q_r_union;
                           bool both_empty   = q_r_union == 0;
                           return 1 - ((!both_empty * dice) + both_empty);
                         });
  }

  ~dice_expanded_distances_t() = default;

 private:
  const distances_config_t<value_idx, value_t>* config_;
  rmm::device_uvector<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

}  // END namespace sparse
}  // END namespace detail
}  // END namespace distance
}  // END namespace cuvs
