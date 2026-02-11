/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/resources.hpp>

namespace cuvs {  // NOLINT(modernize-concat-nested-namespaces)
namespace distance {
namespace detail {
namespace sparse {

template <typename value_idx, typename ValueT>  // NOLINT(readability-identifier-naming)
struct distances_config_t {
  explicit distances_config_t(raft::resources const& handle_) : handle(handle_) {}

  // left side
  value_idx a_nrows;
  value_idx a_ncols;
  value_idx a_nnz;
  value_idx* a_indptr;
  value_idx* a_indices;
  ValueT* a_data;

  // right side
  value_idx b_nrows;
  value_idx b_ncols;
  value_idx b_nnz;
  value_idx* b_indptr;
  value_idx* b_indices;
  ValueT* b_data;

  raft::resources const& handle;
};

template <typename ValueT>  // NOLINT(readability-identifier-naming)
class distances_t {
 public:
  virtual void compute(ValueT* out) {}
  virtual ~distances_t() = default;
};

}  // namespace sparse
}  // namespace detail
}  // namespace distance
}  // namespace cuvs
