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
template <typename T>
index<T>::index(raft::resources const& res,
                raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                std::optional<raft::device_vector<T, int64_t>>&& norms,
                cuvs::distance::DistanceType metric,
                T metric_arg)
  : ann::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
  raft::resource::sync_stream(res);
}

template <typename T>
index<T>::index(raft::resources const& res,
                raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                std::optional<raft::device_vector<T, int64_t>>&& norms,
                cuvs::distance::DistanceType metric,
                T metric_arg)
  : ann::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
}

template <typename T>
index<T>::index(raft::resources const& res,
                raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
                std::optional<raft::device_vector_view<const T, int64_t>> norms_view,
                cuvs::distance::DistanceType metric,
                T metric_arg)
  : ann::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    dataset_view_(dataset_view),
    norms_view_(norms_view),
    metric_arg_(metric_arg)
{
}

template <typename T>
void index<T>::update_dataset(raft::resources const& res,
                              raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_view_ = dataset;
}

template <typename T>
void index<T>::update_dataset(raft::resources const& res,
                              raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_ = raft::make_device_matrix<T, int64_t>(res, dataset.extent(0), dataset.extent(1));
  raft::copy(res, dataset_.view(), dataset);
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

#define CUVS_INST_BFKNN(T)                                                           \
  auto build(raft::resources const& res,                                             \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,    \
             cuvs::distance::DistanceType metric,                                    \
             T metric_arg)                                                           \
    ->cuvs::neighbors::brute_force::index<T>                                         \
  {                                                                                  \
    return detail::build<T>(res, dataset, metric, metric_arg);                       \
  }                                                                                  \
                                                                                     \
  void search(raft::resources const& res,                                            \
              const cuvs::neighbors::brute_force::index<T>& idx,                     \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,   \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors, \
              raft::device_matrix_view<T, int64_t, raft::row_major> distances)       \
  {                                                                                  \
    detail::brute_force_search<T, int64_t>(res, idx, queries, neighbors, distances); \
  }                                                                                  \
                                                                                     \
  template struct cuvs::neighbors::brute_force::index<T>;

CUVS_INST_BFKNN(float);

#undef CUVS_INST_BFKNN

}  // namespace cuvs::neighbors::brute_force
