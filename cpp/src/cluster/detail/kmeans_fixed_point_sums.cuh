/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>

/**
 * Deterministic per-cluster column sums by exact fixed-point accumulation.
 *
 * Each value is converted to a 64-bit integer with a per-column power-of-two scale, and the
 * integers are added with atomics. Integer addition is associative, so the result does not depend
 * on the order of the atomics (or on the GPU). The scale is chosen from the largest absolute value
 * of each column so that n_rows values cannot overflow: each value is rounded to a multiple of
 * about max|column| * n_rows * 2^-62, which is far below the rounding error of a float sum.
 *
 * This needs no sort, and with the scales computed once for a dataset that is summed repeatedly
 * (as in the balanced k-means EM loop), each sum takes two kernels: the same launch count as the
 * atomic floating-point sums it replaces. That matters for balanced k-means, which runs many small
 * problems where launch overhead dominates.
 */
namespace cuvs::cluster::kmeans::detail::fixed_point_sums {

/** Exponent e such that |x| * 2^e < 2^(61 - log2_rows) whenever |x| <= max_abs. */
__host__ __device__ inline int column_exponent(double max_abs, int log2_rows)
{
  int exp2 = 0;
  // max_abs < 2^exp2 (frexp gives exp2 = 0 for 0)
  frexp(max_abs, &exp2);
  return 61 - log2_rows - exp2;
}

template <typename MathT>
__device__ inline int64_t to_fixed(MathT x, int exponent)
{
  if constexpr (std::is_same_v<MathT, float>) {
    return __float2ll_rn(ldexpf(x, exponent));
  } else {
    return __double2ll_rn(ldexp(static_cast<double>(x), exponent));
  }
}

/** Per-column maximum of |X| (exact; the order of the atomics does not matter for a maximum). */
template <typename MathT, typename DataIterator>
RAFT_KERNEL column_max_abs_kernel(DataIterator X, int64_t n_rows, int64_t n_cols, MathT* col_max)
{
  using BitsT         = std::conditional_t<sizeof(MathT) == 4, unsigned int, unsigned long long>;
  const int64_t j     = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t chunk = raft::ceildiv<int64_t>(n_rows, gridDim.y);
  const int64_t begin = static_cast<int64_t>(blockIdx.y) * chunk;
  const int64_t end   = std::min<int64_t>(begin + chunk, n_rows);
  if (j >= n_cols) { return; }
  MathT m = MathT{0};
  for (int64_t row = begin; row < end; ++row) {
    const auto x = static_cast<MathT>(X[row * n_cols + j]);
    m            = std::max<MathT>(m, x < MathT{0} ? -x : x);
  }
  // Non-negative floating-point values compare like their bit patterns.
  if constexpr (sizeof(MathT) == 4) {
    atomicMax(reinterpret_cast<BitsT*>(col_max + j), __float_as_uint(m));
  } else {
    atomicMax(reinterpret_cast<BitsT*>(col_max + j), static_cast<BitsT>(__double_as_longlong(m)));
  }
}

template <typename MathT, typename DataIterator, typename LabelsIterator>
RAFT_KERNEL accumulate_kernel(DataIterator X,
                              LabelsIterator labels,
                              int64_t n_rows,
                              int64_t n_cols,
                              int64_t n_clusters,
                              const MathT* col_max,
                              int log2_rows,
                              bool use_shared,
                              unsigned long long* acc)
{
  extern __shared__ unsigned long long fixed_point_smem[];
  unsigned long long* shared_acc = fixed_point_smem;
  int* exponents =
    reinterpret_cast<int*>(fixed_point_smem + (use_shared ? n_clusters * n_cols : 0));
  const int64_t cells = n_clusters * n_cols;
  for (int64_t j = threadIdx.x; j < n_cols; j += blockDim.x) {
    exponents[j] = column_exponent(static_cast<double>(col_max[j]), log2_rows);
  }
  if (use_shared) {
    for (int64_t i = threadIdx.x; i < cells; i += blockDim.x) {
      shared_acc[i] = 0;
    }
  }
  __syncthreads();
  unsigned long long* dst = use_shared ? shared_acc : acc;
  const int64_t total     = n_rows * n_cols;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = idx / n_cols;
    const int64_t j   = idx - row * n_cols;
    const auto c      = static_cast<int64_t>(labels[row]);
    const auto t      = to_fixed<MathT>(static_cast<MathT>(X[idx]), exponents[j]);
    // Two's complement wrap-around makes the unsigned atomic add a signed one.
    atomicAdd(dst + c * n_cols + j, static_cast<unsigned long long>(t));
  }
  if (use_shared) {
    __syncthreads();
    for (int64_t i = threadIdx.x; i < cells; i += blockDim.x) {
      if (shared_acc[i] != 0) { atomicAdd(acc + i, shared_acc[i]); }
    }
  }
}

/** Converts the accumulators to MathT, writes or adds them to `sums`, and zeroes them again. */
template <typename MathT>
RAFT_KERNEL convert_kernel(unsigned long long* acc,
                           int64_t n_clusters,
                           int64_t n_cols,
                           const MathT* col_max,
                           int log2_rows,
                           bool reset_sums,
                           MathT* sums)
{
  const int64_t cells = n_clusters * n_cols;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < cells;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t j  = i % n_cols;
    const auto fixed = static_cast<int64_t>(acc[i]);
    acc[i]           = 0;
    const int exp    = column_exponent(static_cast<double>(col_max[j]), log2_rows);
    const auto value = static_cast<MathT>(ldexp(static_cast<double>(fixed), -exp));
    sums[i]          = reset_sums ? value : sums[i] + value;
  }
}

/**
 * @brief Repeated deterministic per-cluster column sums over one dataset.
 *
 * `set_scale` computes the per-column scales of a dataset; `sums_by_label` then sums the rows of
 * that dataset (or of any subset of at most the same number of rows, with values within the same
 * column ranges) per label. The accumulators are zeroed after every sum, so consecutive sums need
 * no memset.
 *
 * @tparam MathT Accumulation and output type (float or double)
 */
template <typename MathT>
class accumulator {
 public:
  accumulator(raft::resources const& handle,
              int64_t n_cols,
              int64_t n_clusters,
              rmm::device_async_resource_ref mr)
    : n_cols_{n_cols},
      n_clusters_{n_clusters},
      col_max_(n_cols, raft::resource::get_cuda_stream(handle), mr),
      acc_(n_cols * n_clusters, raft::resource::get_cuda_stream(handle), mr)
  {
    auto stream = raft::resource::get_cuda_stream(handle);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(acc_.data(), 0, acc_.size() * sizeof(unsigned long long), stream.get()));
  }

  /** Computes the per-column scales of the [n_rows x n_cols] row-major dataset X. */
  template <typename DataIterator>
  void set_scale(raft::resources const& handle, DataIterator X, int64_t n_rows)
  {
    auto stream = raft::resource::get_cuda_stream(handle).get();
    log2_rows_  = 0;
    while ((int64_t{1} << log2_rows_) < std::max<int64_t>(n_rows, 1)) {
      ++log2_rows_;
    }
    max_rows_ = n_rows;
    RAFT_CUDA_TRY(cudaMemsetAsync(col_max_.data(), 0, n_cols_ * sizeof(MathT), stream));
    if (n_rows == 0 || n_cols_ == 0) { return; }
    constexpr int kThreads = 128;
    const auto row_chunks =
      static_cast<unsigned>(std::min<int64_t>(raft::ceildiv<int64_t>(n_rows, 1024), int64_t{1024}));
    const dim3 grid(static_cast<unsigned>(raft::ceildiv<int64_t>(n_cols_, kThreads)), row_chunks);
    column_max_abs_kernel<<<grid, kThreads, 0, stream>>>(X, n_rows, n_cols_, col_max_.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  /**
   * sums[c, :] (+)= sum of the rows i of X with labels[i] == c.
   *
   * @param[in] X       Row-major data [n_rows x n_cols], within the ranges given to set_scale
   * @param[in] labels  Label of each row, in [0, n_clusters)
   * @param[in] n_rows  Number of rows, at most the number given to set_scale
   * @param[inout] sums Row-major output [n_clusters x n_cols]
   * @param[in] reset_sums Overwrite `sums` if true, add to it otherwise
   */
  template <typename DataIterator, typename LabelsIterator>
  void sums_by_label(raft::resources const& handle,
                     DataIterator X,
                     LabelsIterator labels,
                     int64_t n_rows,
                     MathT* sums,
                     bool reset_sums)
  {
    RAFT_EXPECTS(n_rows <= max_rows_, "fixed_point_sums: more rows than given to set_scale");
    auto stream         = raft::resource::get_cuda_stream(handle).get();
    const int64_t cells = n_cols_ * n_clusters_;
    if (cells == 0) { return; }
    constexpr int kThreads    = 256;
    constexpr size_t kMaxSmem = 40 * 1024;
    const size_t exp_bytes    = sizeof(int) * n_cols_;
    const bool use_shared     = cells * sizeof(unsigned long long) + exp_bytes <= kMaxSmem;
    const size_t smem         = (use_shared ? cells * sizeof(unsigned long long) : 0) + exp_bytes;
    if (n_rows > 0) {
      const int64_t sms = raft::resource::get_device_properties(handle).multiProcessorCount;
      const auto grid   = static_cast<unsigned>(std::min<int64_t>(
        raft::ceildiv<int64_t>(n_rows * n_cols_, kThreads), use_shared ? 4 * sms : 64 * sms));
      accumulate_kernel<<<grid, kThreads, smem, stream>>>(X,
                                                          labels,
                                                          n_rows,
                                                          n_cols_,
                                                          n_clusters_,
                                                          col_max_.data(),
                                                          log2_rows_,
                                                          use_shared,
                                                          acc_.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    const auto grid =
      static_cast<unsigned>(std::min<int64_t>(raft::ceildiv<int64_t>(cells, kThreads), 4096));
    convert_kernel<<<grid, kThreads, 0, stream>>>(
      acc_.data(), n_clusters_, n_cols_, col_max_.data(), log2_rows_, reset_sums, sums);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

 private:
  int64_t n_cols_;
  int64_t n_clusters_;
  int64_t max_rows_ = 0;
  int log2_rows_    = 0;
  rmm::device_uvector<MathT> col_max_;
  rmm::device_uvector<unsigned long long> acc_;
};

}  // namespace cuvs::cluster::kmeans::detail::fixed_point_sums
