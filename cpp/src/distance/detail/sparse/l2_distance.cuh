/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.hpp"
#include "ip_distance.cuh"
#include <cuvs/distance/distance.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/csr.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <nvfunctional>

namespace cuvs::distance::detail::sparse {

// @TODO: Move this into sparse prims (coo_norm)
template <typename ValueIdx, typename ValueT>
RAFT_KERNEL compute_row_norm_kernel(ValueT* out,
                                    const ValueIdx* __restrict__ coo_rows,
                                    const ValueT* __restrict__ data,
                                    ValueIdx nnz)
{
  ValueIdx i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nnz) { atomicAdd(&out[coo_rows[i]], data[i] * data[i]); }
}

template <typename ValueIdx, typename ValueT>
RAFT_KERNEL compute_row_sum_kernel(ValueT* out,
                                   const ValueIdx* __restrict__ coo_rows,
                                   const ValueT* __restrict__ data,
                                   ValueIdx nnz)
{
  ValueIdx i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nnz) { atomicAdd(&out[coo_rows[i]], data[i]); }
}

template <typename ValueIdx, typename ValueT, typename ExpansionF>
RAFT_KERNEL compute_euclidean_warp_kernel(ValueT* __restrict__ C,
                                          const ValueT* __restrict__ q_sq_norms,
                                          const ValueT* __restrict__ r_sq_norms,
                                          ValueIdx n_rows,
                                          ValueIdx n_cols,
                                          ExpansionF expansion_func)
{
  std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  ValueIdx i      = tid / n_cols;
  ValueIdx j      = tid % n_cols;

  if (i >= n_rows || j >= n_cols) return;

  ValueT dot = C[(size_t)i * n_cols + j];

  // e.g. Euclidean expansion func = -2.0 * dot + q_norm + r_norm
  ValueT val = expansion_func(dot, q_sq_norms[i], r_sq_norms[j]);

  // correct for small instabilities
  C[(size_t)i * n_cols + j] = val * (fabs(val) >= 0.0001);
}

template <typename ValueIdx, typename ValueT>
RAFT_KERNEL compute_correlation_warp_kernel(ValueT* __restrict__ C,
                                            const ValueT* __restrict__ q_sq_norms,
                                            const ValueT* __restrict__ r_sq_norms,
                                            const ValueT* __restrict__ q_norms,
                                            const ValueT* __restrict__ r_norms,
                                            ValueIdx n_rows,
                                            ValueIdx n_cols,
                                            ValueIdx n)
{
  std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  ValueIdx i      = tid / n_cols;
  ValueIdx j      = tid % n_cols;

  if (i >= n_rows || j >= n_cols) return;

  ValueT dot  = C[(size_t)i * n_cols + j];
  ValueT q_l1 = q_norms[i];
  ValueT r_l1 = r_norms[j];

  ValueT q_l2 = q_sq_norms[i];
  ValueT r_l2 = r_sq_norms[j];

  ValueT numer   = n * dot - (q_l1 * r_l1);
  ValueT q_denom = n * q_l2 - (q_l1 * q_l1);
  ValueT r_denom = n * r_l2 - (r_l1 * r_l1);

  ValueT val = 1 - (numer / raft::sqrt(q_denom * r_denom));

  // correct for small instabilities
  C[(size_t)i * n_cols + j] = val * (fabs(val) >= 0.0001);
}

template <typename ValueIdx, typename ValueT, int tpb = 256, typename ExpansionF>
void compute_euclidean(ValueT* C,
                       const ValueT* q_sq_norms,
                       const ValueT* r_sq_norms,
                       ValueIdx n_rows,
                       ValueIdx n_cols,
                       cudaStream_t stream,
                       ExpansionF expansion_func)
{
  int blocks = raft::ceildiv<size_t>((size_t)n_rows * n_cols, tpb);
  compute_euclidean_warp_kernel<<<blocks, tpb, 0, stream>>>(
    C, q_sq_norms, r_sq_norms, n_rows, n_cols, expansion_func);
}

template <typename ValueIdx, typename ValueT, int tpb = 256, typename ExpansionF>
void compute_l2(ValueT* out,
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
  rmm::device_uvector<ValueT> q_sq_norms(m, stream);
  rmm::device_uvector<ValueT> r_sq_norms(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(q_sq_norms.data(), 0, q_sq_norms.size() * sizeof(ValueT)));
  RAFT_CUDA_TRY(cudaMemsetAsync(r_sq_norms.data(), 0, r_sq_norms.size() * sizeof(ValueT)));

  compute_row_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    q_sq_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_row_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    r_sq_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_euclidean(out, q_sq_norms.data(), r_sq_norms.data(), m, n, stream, expansion_func);
}

template <typename ValueIdx, typename ValueT, int tpb = 256>
void compute_correlation(ValueT* C,
                         const ValueT* q_sq_norms,
                         const ValueT* r_sq_norms,
                         const ValueT* q_norms,
                         const ValueT* r_norms,
                         ValueIdx n_rows,
                         ValueIdx n_cols,
                         ValueIdx n,
                         cudaStream_t stream)
{
  int blocks = raft::ceildiv<size_t>((size_t)n_rows * n_cols, tpb);
  compute_correlation_warp_kernel<<<blocks, tpb, 0, stream>>>(
    C, q_sq_norms, r_sq_norms, q_norms, r_norms, n_rows, n_cols, n);
}

template <typename ValueIdx, typename ValueT, int tpb = 256>
void compute_corr(ValueT* out,
                  const ValueIdx* Q_coo_rows,
                  const ValueT* Q_data,
                  ValueIdx Q_nnz,
                  const ValueIdx* R_coo_rows,
                  const ValueT* R_data,
                  ValueIdx R_nnz,
                  ValueIdx m,
                  ValueIdx n,
                  ValueIdx n_cols,
                  cudaStream_t stream)
{
  // sum_sq for std dev
  rmm::device_uvector<ValueT> q_sq_norms(m, stream);
  rmm::device_uvector<ValueT> r_sq_norms(n, stream);

  // sum for mean
  rmm::device_uvector<ValueT> q_norms(m, stream);
  rmm::device_uvector<ValueT> r_norms(n, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(q_sq_norms.data(), 0, q_sq_norms.size() * sizeof(ValueT)));
  RAFT_CUDA_TRY(cudaMemsetAsync(r_sq_norms.data(), 0, r_sq_norms.size() * sizeof(ValueT)));

  RAFT_CUDA_TRY(cudaMemsetAsync(q_norms.data(), 0, q_norms.size() * sizeof(ValueT)));
  RAFT_CUDA_TRY(cudaMemsetAsync(r_norms.data(), 0, r_norms.size() * sizeof(ValueT)));

  compute_row_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    q_sq_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_row_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    r_sq_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_row_sum_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    q_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_row_sum_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    r_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_correlation(out,
                      q_sq_norms.data(),
                      r_sq_norms.data(),
                      q_norms.data(),
                      r_norms.data(),
                      m,
                      n,
                      n_cols,
                      stream);
}

/**
 * L2 distance using the expanded form: sum(x_k)^2 + sum(y_k)^2 - 2 * sum(x_k * y_k)
 * The expanded form is more efficient for sparse data.
 */
template <typename ValueIdx = int, typename ValueT = float>
class l2_expanded_distances_t : public distances_t<ValueT> {
 public:
  explicit l2_expanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config), ip_dists_(config)
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

    compute_l2(out_dists,
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
                 return -2 * dot + q_norm + r_norm;
               });
  }

  ~l2_expanded_distances_t() = default;

 protected:
  const distances_config_t<ValueIdx, ValueT>* config_;
  ip_distances_t<ValueIdx, ValueT> ip_dists_;
};

/**
 * L2 sqrt distance performing the sqrt operation after the distance computation
 * The expanded form is more efficient for sparse data.
 */
template <typename ValueIdx = int, typename ValueT = float>
class l2_sqrt_expanded_distances_t : public l2_expanded_distances_t<ValueIdx, ValueT> {
 public:
  explicit l2_sqrt_expanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : l2_expanded_distances_t<ValueIdx, ValueT>(config)
  {
  }

  void compute(ValueT* out_dists) override
  {
    l2_expanded_distances_t<ValueIdx, ValueT>::compute(out_dists);
    // Sqrt Post-processing
    raft::linalg::unaryOp<ValueT>(
      out_dists,
      out_dists,
      this->config_->a_nrows * this->config_->b_nrows,
      [] __device__(ValueT input) -> ValueT {
        int neg = input < 0 ? -1 : 1;
        return raft::sqrt(abs(input) * neg);
      },
      raft::resource::get_cuda_stream(this->config_->handle));
  }

  ~l2_sqrt_expanded_distances_t() = default;
};

template <typename ValueIdx, typename ValueT>
class correlation_expanded_distances_t : public distances_t<ValueT> {
 public:
  explicit correlation_expanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config), ip_dists_(config)
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

    compute_corr(out_dists,
                 search_coo_rows.data(),
                 config_->a_data,
                 config_->a_nnz,
                 b_indices,
                 b_data,
                 config_->b_nnz,
                 config_->a_nrows,
                 config_->b_nrows,
                 config_->b_ncols,
                 raft::resource::get_cuda_stream(config_->handle));
  }

  ~correlation_expanded_distances_t() = default;

 protected:
  const distances_config_t<ValueIdx, ValueT>* config_;
  ip_distances_t<ValueIdx, ValueT> ip_dists_;
};

/**
 * Cosine distance using the expanded form: 1 - ( sum(x_k * y_k) / (sqrt(sum(x_k)^2) *
 * sqrt(sum(y_k)^2))) The expanded form is more efficient for sparse data.
 */
template <typename ValueIdx = int, typename ValueT = float>
class cosine_expanded_distances_t : public distances_t<ValueT> {
 public:
  explicit cosine_expanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
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

    compute_l2(out_dists,
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
                 ValueT norms = raft::sqrt(q_norm) * raft::sqrt(r_norm);
                 // deal with potential for 0 in denominator by forcing 0/1 instead
                 ValueT cos = ((norms != 0) * dot) / ((norms == 0) + norms);

                 // flip the similarity when both rows are 0
                 bool both_empty = (q_norm == 0) && (r_norm == 0);
                 return 1 - ((!both_empty * cos) + both_empty);
               });
  }

  ~cosine_expanded_distances_t() = default;

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
  rmm::device_uvector<char> workspace_;
  ip_distances_t<ValueIdx, ValueT> ip_dists_;
};

/**
 * Hellinger distance using the expanded form: sqrt(1 - sum(sqrt(x_k) * sqrt(y_k)))
 * The expanded form is more efficient for sparse data.
 *
 * This distance computation modifies A and B by computing a sqrt
 * and then performing a `pow(x, 2)` to convert it back. Because of this,
 * it is possible that the values in A and B might differ slightly
 * after this is invoked.
 */
template <typename ValueIdx = int, typename ValueT = float>
class hellinger_expanded_distances_t : public distances_t<ValueT> {
 public:
  explicit hellinger_expanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config), workspace_(0, raft::resource::get_cuda_stream(config.handle))
  {
  }

  void compute(ValueT* out_dists)
  {
    rmm::device_uvector<ValueIdx> coo_rows(std::max(config_->b_nnz, config_->a_nnz),
                                           raft::resource::get_cuda_stream(config_->handle));

    raft::sparse::convert::csr_to_coo(config_->b_indptr,
                                      config_->b_nrows,
                                      coo_rows.data(),
                                      config_->b_nnz,
                                      raft::resource::get_cuda_stream(config_->handle));

    balanced_coo_pairwise_generalized_spmv<ValueIdx, ValueT>(
      out_dists,
      *config_,
      coo_rows.data(),
      [] __device__(ValueT a, ValueT b) -> ValueT { return raft::sqrt(a) * raft::sqrt(b); },
      raft::add_op(),
      raft::atomic_add_op());

    raft::linalg::unaryOp<ValueT>(
      out_dists,
      out_dists,
      config_->a_nrows * config_->b_nrows,
      [=] __device__(ValueT input) -> ValueT {
        // Adjust to replace NaN in sqrt with 0 if input to sqrt is negative
        bool rectifier = (1 - input) > 0;
        return raft::sqrt(rectifier * (1 - input));
      },
      raft::resource::get_cuda_stream(config_->handle));
  }

  ~hellinger_expanded_distances_t() = default;

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
  rmm::device_uvector<char> workspace_;
};

template <typename ValueIdx = int, typename ValueT = float>
class russelrao_expanded_distances_t : public distances_t<ValueT> {
 public:
  explicit russelrao_expanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config),
      workspace_(0, raft::resource::get_cuda_stream(config.handle)),
      ip_dists_(config)
  {
  }

  void compute(ValueT* out_dists)
  {
    ip_dists_.compute(out_dists);

    ValueT n_cols     = config_->a_ncols;
    ValueT n_cols_inv = 1.0 / n_cols;
    raft::linalg::unaryOp<ValueT>(
      out_dists,
      out_dists,
      config_->a_nrows * config_->b_nrows,
      [=] __device__(ValueT input) -> ValueT { return (n_cols - input) * n_cols_inv; },
      raft::resource::get_cuda_stream(config_->handle));

    auto exec_policy = rmm::exec_policy(raft::resource::get_cuda_stream(config_->handle));
    auto diags       = thrust::counting_iterator<ValueIdx>(0);
    ValueIdx b_nrows = config_->b_nrows;
    thrust::for_each(
      exec_policy, diags, diags + config_->a_nrows, [=] __device__(ValueIdx input) -> void {
        out_dists[input * b_nrows + input] = 0.0;
      });
  }

  ~russelrao_expanded_distances_t() = default;

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
  rmm::device_uvector<char> workspace_;
  ip_distances_t<ValueIdx, ValueT> ip_dists_;
};

}  // namespace cuvs::distance::detail::sparse
   // END namespace detail
   // END namespace distance
   // END namespace cuvs
