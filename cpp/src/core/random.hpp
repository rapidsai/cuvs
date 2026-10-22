/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>

namespace cuvs::detail {

using raft::random::RngState;

template <typename T, typename Len>
void uniform_impl(RngState& state, T* out, Len count, T a, T b, cudaStream_t stream);

template <typename T, typename Len>
void uniformInt_impl(RngState& state, T* out, Len count, T a, T b, cudaStream_t stream);

template <typename T, typename Len>
void normal_impl(RngState& state, T* out, Len count, T a, T b, cudaStream_t stream);

template <typename T, typename IdxT>
void sample_rows_impl(const raft::resources& res,
                      RngState state,
                      const T* input,
                      IdxT stride,
                      IdxT n_rows,
                      raft::device_matrix_view<T, IdxT> output);

raft::device_vector<int, int> sample_indices(const raft::resources& res,
                                             RngState& state,
                                             int n,
                                             int count);

raft::device_vector<int64_t, int64_t> sample_indices(const raft::resources& res,
                                                     RngState& state,
                                                     int64_t n,
                                                     int64_t count);

template <typename IdxT, typename MatIdxT = IdxT>
auto excess_subsample(const raft::resources& res, RngState& state, IdxT n, IdxT count)
{
  return sample_indices(res, state, n, count);
}

template <typename OutputValueType, typename IndexType>
void uniform(raft::resources const& handle,
             RngState& rng_state,
             raft::device_vector_view<OutputValueType, IndexType> out,
             OutputValueType start,
             OutputValueType end)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  uniform_impl(rng_state,
               out.data_handle(),
               out.extent(0),
               start,
               end,
               raft::resource::get_cuda_stream(handle).get());
}

template <typename OutType, typename LenType = int>
void uniform(raft::resources const& handle,
             RngState& rng_state,
             OutType* ptr,
             LenType len,
             OutType start,
             OutType end)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  uniform_impl(rng_state, ptr, len, start, end, raft::resource::get_cuda_stream(handle).get());
}

template <typename OutputValueType, typename IndexType>
void uniformInt(raft::resources const& handle,
                RngState& rng_state,
                raft::device_vector_view<OutputValueType, IndexType> out,
                OutputValueType start,
                OutputValueType end)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  static_assert(
    std::is_same<OutputValueType, typename std::remove_cv<OutputValueType>::type>::value,
    "uniformInt: The output vector must be a view of nonconst, "
    "so that we can write to it.");
  static_assert(std::is_integral<OutputValueType>::value,
                "uniformInt: The elements of the output vector must have integral type.");
  uniformInt_impl(rng_state,
                  out.data_handle(),
                  out.extent(0),
                  start,
                  end,
                  raft::resource::get_cuda_stream(handle).get());
}

template <typename OutType, typename LenType = int>
void uniformInt(raft::resources const& handle,
                RngState& rng_state,
                OutType* ptr,
                LenType len,
                OutType start,
                OutType end)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  uniformInt_impl(rng_state, ptr, len, start, end, raft::resource::get_cuda_stream(handle).get());
}

template <typename OutputValueType, typename IndexType>
void normal(raft::resources const& handle,
            RngState& rng_state,
            raft::device_vector_view<OutputValueType, IndexType> out,
            OutputValueType mu,
            OutputValueType sigma)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  normal_impl(rng_state,
              out.data_handle(),
              out.extent(0),
              mu,
              sigma,
              raft::resource::get_cuda_stream(handle).get());
}

template <typename OutType, typename LenType = int>
void normal(raft::resources const& handle,
            RngState& rng_state,
            OutType* ptr,
            LenType len,
            OutType mu,
            OutType sigma)
{
  if (raft::resource::get_dry_run_flag(handle)) { return; }
  normal_impl(rng_state, ptr, len, mu, sigma, raft::resource::get_cuda_stream(handle).get());
}

template <typename T, typename IdxT, typename accessor, typename layout>
void sample_rows(raft::resources const& res,
                 RngState random_state,
                 raft::mdspan<const T, raft::matrix_extent<IdxT>, layout, accessor> dataset,
                 raft::device_matrix_view<T, IdxT> output)
{
  static_assert(std::is_same_v<layout, raft::row_major> ||
                std::is_same_v<layout, raft::layout_stride>);
  RAFT_EXPECTS(dataset.extent(1) == output.extent(1),
               "dataset dims must match, but received %ld vs %ld",
               static_cast<long>(dataset.extent(1)),
               static_cast<long>(output.extent(1)));
  sample_rows_impl<T, IdxT>(
    res, random_state, dataset.data_handle(), dataset.stride(0), dataset.extent(0), output);
}

template <typename T, typename IdxT, typename accessor, typename layout>
raft::device_matrix<T, IdxT> sample_rows(
  raft::resources const& res,
  RngState random_state,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, layout, accessor> dataset,
  IdxT n_samples)
{
  static_assert(std::is_same_v<layout, raft::row_major> ||
                std::is_same_v<layout, raft::layout_stride>);
  auto output = raft::make_device_matrix<T, IdxT>(res, n_samples, dataset.extent(1));
  cuvs::detail::sample_rows(res, random_state, dataset, output.view());
  return output;
}

}  // namespace cuvs::detail
