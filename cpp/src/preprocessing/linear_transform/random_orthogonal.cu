/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/random_orthogonal.cuh"

#include <cuvs/preprocessing/linear_transform/random_orthogonal.hpp>

namespace cuvs::preprocessing::linear_transform::random_orthogonal {

#define CUVS_INST_TRANSFORMATION(T)                                                \
  auto train(raft::resources const& res,                                           \
             const params params,                                                  \
             raft::device_matrix_view<const T, int64_t> dataset) -> transformer<T> \
  {                                                                                \
    return detail::train(res, params, dataset);                                    \
  }                                                                                \
  auto train(raft::resources const& res,                                           \
             const params params,                                                  \
             raft::host_matrix_view<const T, int64_t> dataset) -> transformer<T>   \
  {                                                                                \
    return detail::train(res, params, dataset);                                    \
  }                                                                                \
  void transform(raft::resources const& res,                                       \
                 const transformer<T>& transformer,                                \
                 raft::device_matrix_view<const T, int64_t> dataset,               \
                 raft::device_matrix_view<T, int64_t> out)                         \
  {                                                                                \
    detail::transform(res, transformer, dataset, out);                             \
  }                                                                                \
  void transform(raft::resources const& res,                                       \
                 const transformer<T>& transformer,                                \
                 raft::host_matrix_view<const T, int64_t> dataset,                 \
                 raft::host_matrix_view<T, int64_t> out)                           \
  {                                                                                \
    detail::transform(res, transformer, dataset, out);                             \
  }                                                                                \
  void inverse_transform(raft::resources const& res,                               \
                         const transformer<T>& transformer,                        \
                         raft::device_matrix_view<const T, int64_t> dataset,       \
                         raft::device_matrix_view<T, int64_t> out)                 \
  {                                                                                \
    detail::inverse_transform(res, transformer, dataset, out);                     \
  }                                                                                \
  void inverse_transform(raft::resources const& res,                               \
                         const transformer<T>& transformer,                        \
                         raft::host_matrix_view<const T, int64_t> dataset,         \
                         raft::host_matrix_view<T, int64_t> out)                   \
  {                                                                                \
    detail::inverse_transform(res, transformer, dataset, out);                     \
  }                                                                                \
  template struct transformer<T>;

CUVS_INST_TRANSFORMATION(double);
CUVS_INST_TRANSFORMATION(float);
CUVS_INST_TRANSFORMATION(half);

#undef CUVS_INST_TRANSFORMATION

}  // namespace cuvs::preprocessing::linear_transform::random_orthogonal
