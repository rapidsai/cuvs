/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.hpp"

#include <raft/core/operators.cuh>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/csr.hpp>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <climits>

#include <algorithm>
#include <nvfunctional>

namespace cuvs::distance::detail::sparse {

template <typename ValueIdx = int,
          typename ValueT   = float,  // NOLINT(readability-identifier-naming)
          typename ProductF,
          typename AccumF,
          typename WriteF>
void unexpanded_lp_distances(ValueT* out_dists,
                             const distances_config_t<ValueIdx, ValueT>* config_,
                             ProductF product_func,
                             AccumF accum_func,
                             WriteF write_func)
{
  rmm::device_uvector<ValueIdx> coo_rows(std::max(config_->b_nnz, config_->a_nnz),
                                         raft::resource::get_cuda_stream(config_->handle));

  raft::sparse::convert::csr_to_coo(config_->b_indptr,
                                    config_->b_nrows,
                                    coo_rows.data(),
                                    config_->b_nnz,
                                    raft::resource::get_cuda_stream(config_->handle));

  balanced_coo_pairwise_generalized_spmv<ValueIdx, ValueT>(
    out_dists, *config_, coo_rows.data(), product_func, accum_func, write_func);

  raft::sparse::convert::csr_to_coo(config_->a_indptr,
                                    config_->a_nrows,
                                    coo_rows.data(),
                                    config_->a_nnz,
                                    raft::resource::get_cuda_stream(config_->handle));

  balanced_coo_pairwise_generalized_spmv_rev<ValueIdx, ValueT>(
    out_dists, *config_, coo_rows.data(), product_func, accum_func, write_func);
}

/**
 * Computes L1 distances for sparse input. This does not have
 * an equivalent expanded form, so it is only executed in
 * an unexpanded form.
 * @tparam ValueIdx
 * @tparam ValueT
 */
template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class l1_unexpanded_distances_t : public distances_t<ValueT> {
 public:
  explicit l1_unexpanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config)
  {
  }

  void compute(ValueT* out_dists)
  {
    unexpanded_lp_distances<ValueIdx, ValueT>(
      out_dists, config_, raft::absdiff_op(), raft::add_op(), raft::atomic_add_op());
  }

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
};

template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class l2_unexpanded_distances_t : public distances_t<ValueT> {
 public:
  explicit l2_unexpanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config)
  {
  }

  void compute(ValueT* out_dists)
  {
    unexpanded_lp_distances<ValueIdx, ValueT>(
      out_dists, config_, raft::sqdiff_op(), raft::add_op(), raft::atomic_add_op());
  }

 protected:
  const distances_config_t<ValueIdx, ValueT>* config_;
};

template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class l2_sqrt_unexpanded_distances_t : public l2_unexpanded_distances_t<ValueIdx, ValueT> {
 public:
  explicit l2_sqrt_unexpanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : l2_unexpanded_distances_t<ValueIdx, ValueT>(config)
  {
  }

  void compute(ValueT* out_dists)
  {
    l2_unexpanded_distances_t<ValueIdx, ValueT>::compute(out_dists);

    uint64_t n =
      static_cast<uint64_t>(this->config_->a_nrows) * static_cast<uint64_t>(this->config_->b_nrows);
    // Sqrt Post-processing
    raft::linalg::unaryOp<ValueT>(
      out_dists,
      out_dists,
      n,
      [] __device__(ValueT input) {
        int neg = input < 0 ? -1 : 1;
        return raft::sqrt(abs(input) * neg);
      },
      raft::resource::get_cuda_stream(this->config_->handle));
  }
};

template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class linf_unexpanded_distances_t : public distances_t<ValueT> {
 public:
  explicit linf_unexpanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config)
  {
  }

  void compute(ValueT* out_dists)
  {
    unexpanded_lp_distances<ValueIdx, ValueT>(
      out_dists, config_, raft::absdiff_op(), raft::max_op(), raft::atomic_max_op());
  }

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
};

template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class canberra_unexpanded_distances_t : public distances_t<ValueT> {
 public:
  explicit canberra_unexpanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config)
  {
  }

  void compute(ValueT* out_dists)
  {
    unexpanded_lp_distances<ValueIdx, ValueT>(
      out_dists,
      config_,
      [] __device__(ValueT a, ValueT b) -> ValueT {
        ValueT d = fabs(a) + fabs(b);

        // deal with potential for 0 in denominator by
        // forcing 1/0 instead
        return ((d != 0) * fabs(a - b)) / (d + (d == 0));
      },
      raft::add_op(),
      raft::atomic_add_op());
  }

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
};

template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class lp_unexpanded_distances_t : public distances_t<ValueT> {
 public:
  explicit lp_unexpanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config, ValueT p_)
    : config_(&config), p_(p_)
  {
  }

  void compute(ValueT* out_dists)
  {
    unexpanded_lp_distances<ValueIdx, ValueT>(
      out_dists,
      config_,
      raft::compose_op(raft::pow_const_op<ValueT>(p_), raft::sub_op()),
      raft::add_op(),
      raft::atomic_add_op());

    uint64_t n =
      static_cast<uint64_t>(this->config_->a_nrows) * static_cast<uint64_t>(this->config_->b_nrows);
    ValueT one_over_p = ValueT{1} / p_;
    raft::linalg::unaryOp<ValueT>(out_dists,
                                  out_dists,
                                  n,
                                  raft::pow_const_op<ValueT>(one_over_p),
                                  raft::resource::get_cuda_stream(config_->handle));
  }

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
  ValueT p_;
};

template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class hamming_unexpanded_distances_t : public distances_t<ValueT> {
 public:
  explicit hamming_unexpanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config)
  {
  }

  void compute(ValueT* out_dists)
  {
    unexpanded_lp_distances<ValueIdx, ValueT>(
      out_dists, config_, raft::notequal_op(), raft::add_op(), raft::atomic_add_op());

    uint64_t n = static_cast<uint64_t>(config_->a_nrows) * static_cast<uint64_t>(config_->b_nrows);
    ValueT n_cols = 1.0 / config_->a_ncols;
    raft::linalg::unaryOp<ValueT>(out_dists,
                                  out_dists,
                                  n,
                                  raft::mul_const_op<ValueT>(n_cols),
                                  raft::resource::get_cuda_stream(config_->handle));
  }

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
};

template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class jensen_shannon_unexpanded_distances_t : public distances_t<ValueT> {
 public:
  explicit jensen_shannon_unexpanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config)
  {
  }

  void compute(ValueT* out_dists)
  {
    unexpanded_lp_distances<ValueIdx, ValueT>(
      out_dists,
      config_,
      [] __device__(ValueT a, ValueT b) -> ValueT {
        ValueT m    = 0.5f * (a + b);
        bool a_zero = a == 0;
        bool b_zero = b == 0;

        ValueT x = (!a_zero * m) / (a_zero + a);
        ValueT y = (!b_zero * m) / (b_zero + b);

        bool x_zero = x == 0;
        bool y_zero = y == 0;

        return (-a * (!x_zero * log(x + x_zero))) + (-b * (!y_zero * log(y + y_zero)));
      },
      raft::add_op(),
      raft::atomic_add_op());

    uint64_t n =
      static_cast<uint64_t>(this->config_->a_nrows) * static_cast<uint64_t>(this->config_->b_nrows);
    raft::linalg::unaryOp<ValueT>(
      out_dists,
      out_dists,
      n,
      [=] __device__(ValueT input) { return raft::sqrt(0.5 * input); },
      raft::resource::get_cuda_stream(config_->handle));
  }

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
};

template <typename ValueIdx = int,
          typename ValueT   = float>  // NOLINT(readability-identifier-naming)
class kl_divergence_unexpanded_distances_t : public distances_t<ValueT> {
 public:
  explicit kl_divergence_unexpanded_distances_t(const distances_config_t<ValueIdx, ValueT>& config)
    : config_(&config)
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
      [] __device__(ValueT a, ValueT b) -> ValueT { return a * log(a / b); },
      raft::add_op(),
      raft::atomic_add_op());

    uint64_t n =
      static_cast<uint64_t>(this->config_->a_nrows) * static_cast<uint64_t>(this->config_->b_nrows);
    raft::linalg::unaryOp<ValueT>(out_dists,
                                  out_dists,
                                  n,
                                  raft::mul_const_op<ValueT>(0.5),
                                  raft::resource::get_cuda_stream(config_->handle));
  }

 private:
  const distances_config_t<ValueIdx, ValueT>* config_;
};

}  // namespace cuvs::distance::detail::sparse
   // END namespace detail
   // END namespace distance
   // END namespace cuvs
