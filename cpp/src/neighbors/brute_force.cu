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

#include "./detail/knn_brute_force.cuh"

#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/copy.hpp>

namespace cuvs::neighbors::brute_force {
template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                       std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
  raft::resource::sync_stream(res);
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                       std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
                       std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    dataset_view_(dataset_view),
    norms_view_(norms_view),
    metric_arg_(metric_arg)
{
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
                       std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(
      raft::make_device_matrix<T, int64_t>(res, dataset_view.extent(0), dataset_view.extent(1))),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  // currently we don't support col_major inside tiled_brute_force_knn, because
  // of limitations of the pairwise_distance API:
  // 1) paiwise_distance takes a single 'isRowMajor' parameter - and we have
  // multiple options here (both dataset and queries)
  // 2) because of tiling, we need to be able to set a custom stride in the PW
  // api, which isn't supported
  // Instead, transpose the input matrices if they are passed as col-major.
  // (note: we're doing the transpose here to avoid doing per query)
  raft::linalg::transpose(res,
                          const_cast<T*>(dataset_view.data_handle()),
                          dataset_.data_handle(),
                          dataset_view.extent(0),
                          dataset_view.extent(1),
                          raft::resource::get_cuda_stream(res));
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
                       std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(
      raft::make_device_matrix<T, int64_t>(res, dataset_view.extent(0), dataset_view.extent(1))),
    norms_view_(norms_view),
    metric_arg_(metric_arg)
{
  // currently we don't support col_major inside tiled_brute_force_knn, because
  // of limitations of the pairwise_distance API:
  // 1) paiwise_distance takes a single 'isRowMajor' parameter - and we have
  // multiple options here (both dataset and queries)
  // 2) because of tiling, we need to be able to set a custom stride in the PW
  // api, which isn't supported
  // Instead, transpose the input matrices if they are passed as col-major.
  // (note: we're doing the transpose here to avoid doing per query)
  raft::linalg::transpose(res,
                          const_cast<T*>(dataset_view.data_handle()),
                          dataset_.data_handle(),
                          dataset_view.extent(0),
                          dataset_view.extent(1),
                          raft::resource::get_cuda_stream(res));
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

template <typename T, typename DistT>
void index<T, DistT>::update_dataset(
  raft::resources const& res, raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_view_ = dataset;
}

template <typename T, typename DistT>
void index<T, DistT>::update_dataset(
  raft::resources const& res, raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_ = raft::make_device_matrix<T, int64_t>(res, dataset.extent(0), dataset.extent(1));
  raft::copy(res, dataset_.view(), dataset);
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

#define CUVS_INST_BFKNN(T, DistT)                                                                 \
  auto build(raft::resources const& res,                                                          \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,                 \
             cuvs::distance::DistanceType metric,                                                 \
             DistT metric_arg)                                                                    \
    ->cuvs::neighbors::brute_force::index<T, DistT>                                               \
  {                                                                                               \
    return detail::build<T, DistT>(res, dataset, metric, metric_arg);                             \
  }                                                                                               \
  auto build(raft::resources const& res,                                                          \
             raft::device_matrix_view<const T, int64_t, raft::col_major> dataset,                 \
             cuvs::distance::DistanceType metric,                                                 \
             DistT metric_arg)                                                                    \
    ->cuvs::neighbors::brute_force::index<T, DistT>                                               \
  {                                                                                               \
    return detail::build<T, DistT>(res, dataset, metric, metric_arg);                             \
  }                                                                                               \
                                                                                                  \
  void search(                                                                                    \
    raft::resources const& res,                                                                   \
    const cuvs::neighbors::brute_force::index<T, DistT>& idx,                                     \
    raft::device_matrix_view<const T, int64_t, raft::row_major> queries,                          \
    raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,                        \
    raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,                          \
    std::optional<cuvs::core::bitmap_view<const uint32_t, int64_t>> sample_filter = std::nullopt) \
  {                                                                                               \
    if (!sample_filter.has_value()) {                                                             \
      detail::brute_force_search<T, int64_t, DistT>(res, idx, queries, neighbors, distances);     \
    } else {                                                                                      \
      detail::brute_force_search_filtered<T, int64_t, uint32_t, DistT>(                           \
        res, idx, queries, *sample_filter, neighbors, distances);                                 \
    }                                                                                             \
  }                                                                                               \
  void search(                                                                                    \
    raft::resources const& res,                                                                   \
    const cuvs::neighbors::brute_force::index<T, DistT>& idx,                                     \
    raft::device_matrix_view<const T, int64_t, raft::col_major> queries,                          \
    raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,                        \
    raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,                          \
    std::optional<cuvs::core::bitmap_view<const uint32_t, int64_t>> sample_filter = std::nullopt) \
  {                                                                                               \
    if (!sample_filter.has_value()) {                                                             \
      detail::brute_force_search<T, int64_t, DistT>(res, idx, queries, neighbors, distances);     \
    } else {                                                                                      \
      RAFT_FAIL("filtered search isn't available with col_major queries yet");                    \
    }                                                                                             \
  }                                                                                               \
                                                                                                  \
  template struct cuvs::neighbors::brute_force::index<T, DistT>;

CUVS_INST_BFKNN(float, float);
CUVS_INST_BFKNN(half, float);

#undef CUVS_INST_BFKNN

}  // namespace cuvs::neighbors::brute_force
