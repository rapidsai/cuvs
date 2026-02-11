/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/stats/silhouette_score.hpp>

#include "./detail/batched/silhouette_score.cuh"
#include "./detail/silhouette_score.cuh"

namespace cuvs {
namespace stats {
namespace {
template <typename ValueT, typename LabelT, typename idx_t>
auto _silhouette_score(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueT, idx_t, raft::row_major> X_in,
  raft::device_vector_view<const LabelT, idx_t> labels,
  std::optional<raft::device_vector_view<ValueT, idx_t>> silhouette_score_per_sample,
  idx_t n_unique_labels,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded) -> ValueT
{
  RAFT_EXPECTS(labels.extent(0) == X_in.extent(0), "Size mismatch between labels and data");

  ValueT* silhouette_score_per_sample_ptr = nullptr;
  if (silhouette_score_per_sample.has_value()) {
    silhouette_score_per_sample_ptr = silhouette_score_per_sample.value().data_handle();
    RAFT_EXPECTS(silhouette_score_per_sample.value().extent(0) == X_in.extent(0),
                 "Size mismatch between silhouette_score_per_sample and data");
  }
  return detail::silhouette_score(handle,
                                  X_in.data_handle(),
                                  X_in.extent(0),
                                  X_in.extent(1),
                                  labels.data_handle(),
                                  n_unique_labels,
                                  silhouette_score_per_sample_ptr,
                                  raft::resource::get_cuda_stream(handle),
                                  metric);
}

template <typename ValueT, typename LabelT, typename idx_t>
auto _silhouette_score_batched(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueT, idx_t, raft::row_major> X,
  raft::device_vector_view<const LabelT, idx_t> labels,
  std::optional<raft::device_vector_view<ValueT, idx_t>> silhouette_score_per_sample,
  idx_t n_unique_labels,
  idx_t batch_size,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded) -> ValueT
{
  static_assert(std::is_integral_v<idx_t>,
                "silhouette_score_batched: The index type "
                "of each mdspan argument must be an integral type.");
  static_assert(std::is_integral_v<LabelT>,
                "silhouette_score_batched: The label type must be an integral type.");
  RAFT_EXPECTS(labels.extent(0) == X.extent(0), "Size mismatch between labels and data");

  ValueT* scores_ptr = nullptr;
  if (silhouette_score_per_sample.has_value()) {
    scores_ptr = silhouette_score_per_sample.value().data_handle();
    RAFT_EXPECTS(silhouette_score_per_sample.value().extent(0) == X.extent(0),
                 "Size mismatch between silhouette_score_per_sample and data");
  }
  return cuvs::stats::batched::detail::silhouette_score<ValueT, int, LabelT>(handle,
                                                                             X.data_handle(),
                                                                             X.extent(0),
                                                                             X.extent(1),
                                                                             labels.data_handle(),
                                                                             n_unique_labels,
                                                                             scores_ptr,
                                                                             batch_size,
                                                                             metric);
}
}  // namespace

auto silhouette_score(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X_in,
  raft::device_vector_view<const int, int64_t> labels,
  std::optional<raft::device_vector_view<float, int64_t>> silhouette_score_per_sample,
  int64_t n_unique_labels,
  cuvs::distance::DistanceType metric) -> float
{
  return _silhouette_score(
    handle, X_in, labels, silhouette_score_per_sample, n_unique_labels, metric);
}

auto silhouette_score(
  raft::resources const& handle,
  raft::device_matrix_view<const double, int64_t, raft::row_major> X_in,
  raft::device_vector_view<const int, int64_t> labels,
  std::optional<raft::device_vector_view<double, int64_t>> silhouette_score_per_sample,
  int64_t n_unique_labels,
  cuvs::distance::DistanceType metric) -> double
{
  return _silhouette_score(
    handle, X_in, labels, silhouette_score_per_sample, n_unique_labels, metric);
}

auto silhouette_score_batched(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X,
  raft::device_vector_view<const int, int64_t> labels,
  std::optional<raft::device_vector_view<float, int64_t>> silhouette_score_per_sample,
  int64_t n_unique_labels,
  int64_t batch_size,
  cuvs::distance::DistanceType metric) -> float
{
  return _silhouette_score_batched(
    handle, X, labels, silhouette_score_per_sample, n_unique_labels, batch_size, metric);
}

auto silhouette_score_batched(
  raft::resources const& handle,
  raft::device_matrix_view<const double, int64_t, raft::row_major> X,
  raft::device_vector_view<const int, int64_t> labels,
  std::optional<raft::device_vector_view<double, int64_t>> silhouette_score_per_sample,
  int64_t n_unique_labels,
  int64_t batch_size,
  cuvs::distance::DistanceType metric) -> double
{
  return _silhouette_score_batched(
    handle, X, labels, silhouette_score_per_sample, n_unique_labels, batch_size, metric);
}
};  // namespace stats
};  // namespace cuvs
